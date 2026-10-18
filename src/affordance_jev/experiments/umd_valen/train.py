"""Two-stage Valen SFT: frozen-feature head warmup, then language LoRA + head.

Only the 76 training records are read. No test iterator/checkpoint selection exists.
"""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file,load_file
from transformers import AutoModel

from ...paths import training_config
from ...common import read,write,read_jsonl,sha


def configure(config):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])


def load_model(config, device):
    model=AutoModel.from_pretrained(config['model_path'],trust_remote_code=True,local_files_only=True,
                                  dtype=torch.bfloat16,attn_implementation=config['attention']).to(device)
    model.requires_grad_(False)
    model.head.float().requires_grad_(True)
    model.eval()
    return model


def compile_records(model,config,output,rank):
    rows=read_jsonl(config['records'])
    if len(rows)!=76 or len({r['group_id'] for r in rows})!=76:
        raise ValueError('Expected exactly the 76 original training instances')
    states=[]
    for i,row in enumerate(rows):
        compiler=model.make_compiler(media_root=row['media_root'],execution='shared_state')
        state=compiler.compile(row,labeled_only=True)
        if not state.questions or any(q.target is None for q in state.questions):
            raise ValueError(f'Missing training target: {row["sample_id"]}')
        # Labels may select supervised readouts, but must never alter input tensors.
        unlabeled=compiler.compile({'request':row['request']})
        if state.inputs.keys()!=unlabeled.inputs.keys() or any(
                not torch.equal(v,unlabeled.inputs[k]) for k,v in state.inputs.items()):
            raise AssertionError('Targets leaked into compiled model inputs')
        states.append(state)
    if rank==0:
        write(output/'compiled_training_audit.json',{
            'records_sha256':sha(config['records']),'images':len(rows),
            'regions':sum(len(s.questions) for s in states),
            'max_input_tokens':max(s.compute_tokens for s in states),
            'targets_do_not_change_inputs':True,
            'sample_ids':[r['sample_id'] for r in rows],
            'input_tokens':[s.compute_tokens for s in states]})
    return states


def loss_for(logits,state):
    losses=[]
    for q,z in zip(state.questions,logits):
        target=torch.tensor(q.target,device=z.device,dtype=torch.float32)
        losses.append(-(target*z.float().log_softmax(-1)).sum())
    return torch.stack(losses).mean()


def schedule(size,total,seed):
    rng=random.Random(seed)
    order=[]
    while len(order)<total:
        chunk=list(range(size))
        rng.shuffle(chunk)
        order.extend(chunk)
    return order[:total]


def save_head(model,path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.head.state_dict().items()},path/'head.safetensors')


def add_lora(model,config):
    from peft import LoraConfig,inject_adapter_in_model
    targets=[name for name,module in model.backbone.named_modules()
             if name.startswith('language_model.layers.') and isinstance(module,nn.Linear)
             and name.rsplit('.',1)[-1] in config['lora_target_suffixes']]
    if not targets: raise ValueError('No language-model LoRA targets')
    lc=LoraConfig(r=config['lora_r'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],
                  target_modules=targets,bias='none')
    inject_adapter_in_model(lc,model.backbone)
    model.head.requires_grad_(True)
    # Explicitly keep the vision tower and merger frozen.
    assert not any(p.requires_grad for p in model.backbone.visual.parameters())
    for name,p in model.backbone.named_parameters():
        if p.requires_grad and 'lora_' not in name: raise ValueError(f'Unexpected trained base parameter: {name}')
    if config['gradient_checkpointing']:
        model.backbone.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return lc,targets


class LossModel(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
    def forward(self,state):
        return loss_for(self.model.forward_state(state),state)


def append(path,value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def train(config,stage,smoke_steps=0):
    rank=int(os.environ.get('RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    configure(config)
    if world>1:
        torch.distributed.init_process_group('nccl',device_id=device)
    output=Path(config['output']) / ('smoke' if smoke_steps else stage)
    output.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(config['checkpoint_dir'])
    if rank==0:
        write(output/'config.json',config)
        write(output/'environment.json',{'python':os.sys.executable,'world_size':world,
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','numpy','Pillow']},
              'gpu':torch.cuda.get_device_name()})
    model=load_model(config,device)
    states=compile_records(model,config,output,rank)
    started=time.perf_counter()
    if stage=='warmup':
        if world!=1: raise ValueError('Head warmup uses one GPU with cached frozen features')
        features=[]
        with torch.no_grad():
            for i,state in enumerate(states,1):
                features.append(model.extract_state_features(state))
                if i%10==0 or i==len(states): print(f'FEATURES {i}/{len(states)}',flush=True)
        optimizer=torch.optim.AdamW(model.head.parameters(),lr=config['head_learning_rate'],weight_decay=config['weight_decay'])
        steps=smoke_steps or config['head_steps']
        batch=config['effective_images_per_step']
        order=schedule(len(states),steps*batch,config['seed'])
        for step in range(1,steps+1):
            optimizer.zero_grad(set_to_none=True)
            values=[]
            for index in order[(step-1)*batch:step*batch]:
                logits=[model.score_features(f) for f in features[index]]
                loss=loss_for(logits,states[index])
                (loss/batch).backward()
                values.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.head.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step()
            if step%10==0 or step==steps:
                value={'stage':'warmup','step':step,'total':steps,'loss':float(np.mean(values)),
                       'elapsed_seconds':time.perf_counter()-started,'status':'running'}
                append(output/'loss.jsonl',value)
                write(output/'status.json',value)
                print(f'WARMUP {step}/{steps} loss={value["loss"]:.6f}',flush=True)
        save_head(model,checkpoint/'warmup')
        write(output/'status.json',{**value,'status':'complete','checkpoint':str(checkpoint/'warmup')})
        return
    warmup=checkpoint/'warmup/head.safetensors'
    if warmup.exists(): model.head.load_state_dict(load_file(warmup),strict=True)
    elif not smoke_steps: raise FileNotFoundError(warmup)
    lc,targets=add_lora(model,config)
    model.train()
    model.backbone.visual.eval()
    torch.manual_seed(config['seed']+rank)
    trainable=[p for p in model.parameters() if p.requires_grad]
    lora=[p for n,p in model.backbone.named_parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([
        {'params':model.head.parameters(),'lr':config['head_learning_rate']},
        {'params':lora,'lr':config['lora_learning_rate']}],weight_decay=config['weight_decay'])
    if rank==0:
        write(output/'trainable_parameters.json',{'head':sum(p.numel() for p in model.head.parameters()),
              'lora':sum(p.numel() for p in lora),'targets':targets,'vision_frozen':True,
              'backbone_dtype':str(next(model.backbone.parameters()).dtype),
              'head_dtypes':sorted({str(p.dtype) for p in model.head.parameters()})})
    network=LossModel(model)
    if world>1:
        network=torch.nn.parallel.DistributedDataParallel(network,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    steps=smoke_steps or config['joint_steps']
    batch=config['effective_images_per_step']
    if batch%world: raise ValueError('Effective image batch must be divisible by world size')
    accumulate=batch//world
    order=schedule(len(states),steps*batch,config['seed']+1)
    if smoke_steps:
        biggest=max(range(len(states)),key=lambda i:states[i].compute_tokens)
        order=[biggest]*len(order)
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,steps+1):
        tick=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_value=torch.zeros((),device=device)
        for micro in range(accumulate):
            index=order[(step-1)*batch+micro*world+rank]
            sync=network.no_sync() if world>1 and micro<accumulate-1 else nullcontext()
            with sync:
                loss=network(states[index])
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss on {index}')
                (loss/accumulate).backward()
            loss_value+=loss.detach()/accumulate
        norm=torch.nn.utils.clip_grad_norm_(trainable,config['gradient_clip_norm'],error_if_nonfinite=True)
        if step==1:
            if not any(p.grad is not None and bool(p.grad.abs().max()>0) for p in lora):
                raise AssertionError('LoRA received no nonzero gradients')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise AssertionError('Frozen parameter has a gradient')
            if rank==0: write(output/'gradient_audit.json',{'frozen_parameters_have_no_grad':True,'lora_receives_gradient':True})
        optimizer.step()
        if step==1 or step%10==0 or step==steps:
            if world>1: torch.distributed.all_reduce(loss_value)
            if rank==0:
                elapsed=time.perf_counter()-started
                value={'stage':'joint','step':step,'total':steps,'loss':float(loss_value/world),
                       'gradient_norm':float(norm),'elapsed_seconds':elapsed,
                       'step_seconds':time.perf_counter()-tick,'eta_seconds':elapsed/step*(steps-step),
                       'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'status':'running'}
                append(output/'loss.jsonl',value)
                write(output/'status.json',value)
                print(f'JOINT {step}/{steps} loss={value["loss"]:.6f} elapsed={elapsed:.1f}s ETA={value["eta_seconds"]:.1f}s',flush=True)
        if (step%250==0 or step==steps) and rank==0:
            from peft import get_peft_model_state_dict
            dest=(output/'checkpoint') if smoke_steps else checkpoint/('final' if step==steps else f'step_{step:04}')
            save_head(model,dest)
            (dest/'adapter').mkdir(exist_ok=True)
            lc.save_pretrained(dest/'adapter')
            save_file({k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(model.backbone).items()},
                      dest/'adapter/adapter_model.safetensors')
            write(dest/'training_config.json',config)
            write(dest/'status.json',{'stage':'joint','step':step,'total':steps,'seed':config['seed'],
                  'selection':'fixed final step' if step==steps else 'recovery only, not test-selected',
                  'training_records_sha256':sha(config['records'])})
            if not smoke_steps:
                torch.save({'optimizer':optimizer.state_dict(),'step':step},dest/'optimizer.pt')
            write(output/'status.json',{**value,'status':'complete' if step==steps else 'running','checkpoint':str(dest)})
    if world>1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['warmup','joint'],required=True)
    parser.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args()
    train(training_config(args.config),args.stage,args.smoke_steps)


if __name__=='__main__': main()
