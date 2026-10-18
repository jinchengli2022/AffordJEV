"""Fixed-budget A/B/C training; no evaluation dataset is opened here."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import os
from pathlib import Path
import time

import torch
from safetensors.torch import save_file,load_file

from ....paths import training_config
from ....common import read,write,read_jsonl,sha
from ..train import configure,load_model,add_lora,save_head,schedule,append
from .model import ABCModel,region_geometry


def compile_training(model, config, output, rank):
    rows = read_jsonl(config['records'])
    assert len(rows) == len({r['group_id'] for r in rows}) == 76
    views = read_jsonl(config['views']) if config.get('views') else []
    audit = []
    def compile_one(row):
        compiler = model.make_compiler(media_root=row['media_root'],execution='shared_state')
        state = compiler.compile(row,labeled_only=True)
        unlabeled = compiler.compile({'request':row['request']})
        assert state.inputs.keys() == unlabeled.inputs.keys()
        assert all(torch.equal(v,unlabeled.inputs[k]) for k,v in state.inputs.items())
        assert len(state.questions) == len(row['regions'])
        geometry = region_geometry(row['media_root'],state,model.config) if config['method'].startswith('c') else None
        audit.append({'sample_id':row['sample_id'],'variant':row.get('variant'),
                      'tokens':state.compute_tokens,'geometry':None if geometry is None else geometry['audit']})
        return state,row,geometry
    originals = [compile_one(row) for row in rows]
    by_id = {r['sample_id']:i for i,r in enumerate(rows)}
    transformed = {i:[] for i in range(len(rows))}
    for j,row in enumerate(views,1):
        assert row['sample_id'] in by_id
        transformed[by_id[row['sample_id']]].append(compile_one(row))
        if rank==0 and j%76==0: print(f'COMPILED VIEWS {j}/{len(views)}',flush=True)
    if views: assert all(len(v)==4 for v in transformed.values())
    if rank==0:
        write(output/'compiled_audit.json',{'training_images':76,'views':len(views),
              'records_sha256':sha(config['records']),
              'views_sha256':sha(config['views']) if views else None,
              'targets_do_not_change_inputs':True,'test_read_during_training':False,
              'max_tokens':max(a['tokens'] for a in audit),'records':audit})
    return originals,transformed


def save(network, lora_config, optimizer, config, dest, step, final):
    from peft import get_peft_model_state_dict
    dest.mkdir(parents=True,exist_ok=True)
    save_head(network.model,dest)
    (dest/'adapter').mkdir(exist_ok=True)
    lora_config.save_pretrained(dest/'adapter')
    save_file({k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(network.model.backbone).items()},
              dest/'adapter/adapter_model.safetensors')
    extras = {k:v.detach().cpu().contiguous() for k,v in network.state_dict().items() if not k.startswith('model.')}
    if extras: save_file(extras,dest/'extra_heads.safetensors')
    write(dest/'training_config.json',config)
    write(dest/'status.json',{'step':step,'total':config['joint_steps'],'method':config['method'],
          'selection':'fixed final step' if final else 'recovery only, never test-selected',
          'training_records_sha256':sha(config['records'])})
    torch.save({'optimizer':optimizer.state_dict(),'step':step},dest/'optimizer.pt')


def train(config, smoke_steps=0):
    rank=int(os.environ.get('RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    configure(config)
    if world>1: torch.distributed.init_process_group('nccl',device_id=device)
    output=Path(config['output'])/('smoke' if smoke_steps else 'joint')
    if (output/'status.json').exists(): raise FileExistsError(output/'status.json')
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    model=load_model(config,device)
    originals,views=compile_training(model,config,output,rank)
    model.head.load_state_dict(load_file(config['warmup_head']),strict=True)
    lc,targets=add_lora(model,config)
    network=ABCModel(model,config).to(device)
    network.train()
    model.backbone.visual.eval()
    lora=[p for p in model.backbone.parameters() if p.requires_grad]
    heads=[p for n,p in network.named_parameters() if p.requires_grad and not n.startswith('model.backbone.')]
    trainable=lora+heads
    optimizer=torch.optim.AdamW([{'params':heads,'lr':config['head_learning_rate']},
                                 {'params':lora,'lr':config['lora_learning_rate']}],weight_decay=config['weight_decay'])
    if rank==0:
        write(output/'config.json',config)
        write(output/'environment.json',{'python':os.sys.executable,'gpu':torch.cuda.get_device_name(),
              'world_size':world,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','numpy','Pillow']}})
        write(output/'trainable_parameters.json',{'lora':sum(p.numel() for p in lora),
              'all_heads':sum(p.numel() for p in heads),'lora_targets':targets,'vision_frozen':True,
              'extra':sum(p.numel() for n,p in network.named_parameters() if p.requires_grad and not n.startswith('model.')),
              'warmup_head_sha256':sha(config['warmup_head'])})
    wrapped=network
    if world>1:
        wrapped=torch.nn.parallel.DistributedDataParallel(network,device_ids=[local],broadcast_buffers=False)
    steps=smoke_steps or config['joint_steps']
    batch=config['effective_images_per_step']
    assert batch%world==0
    accumulate=batch//world
    order=schedule(76,steps*batch,config['seed']+1)
    if smoke_steps:
        largest=max(range(76),key=lambda i:originals[i][0].compute_tokens+
                    max([s[0].compute_tokens for s in views[i]] or [0]))
        order=[largest]*len(order)
    torch.manual_seed(config['seed']+rank)
    torch.cuda.reset_peak_memory_stats()
    loop_started=time.perf_counter()
    for step in range(1,steps+1):
        tick=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        values=torch.zeros(6,device=device)
        for micro in range(accumulate):
            index=order[(step-1)*batch+micro*world+rank]
            view=views[index][(step+index)%4] if views[index] else None
            sync=wrapped.no_sync() if world>1 and micro<accumulate-1 else nullcontext()
            with sync:
                loss,parts=wrapped(originals[index],view)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss: {index}')
                (loss/accumulate).backward()
            values+=torch.cat([loss.detach()[None],parts])/accumulate
        norm=torch.nn.utils.clip_grad_norm_(trainable,config['gradient_clip_norm'],error_if_nonfinite=True)
        if step==1:
            audits={'frozen_parameters_have_no_grad':all(p.grad is None for p in network.parameters() if not p.requires_grad),
                    'lora_receives_nonzero_gradient':any(p.grad is not None and bool(p.grad.abs().max()>0) for p in lora),
                    'heads_receive_nonzero_gradient':any(p.grad is not None and bool(p.grad.abs().max()>0) for p in heads)}
            if network.auxiliary is not None or network.region_head is not None:
                module=network.auxiliary if network.auxiliary is not None else network.region_head
                audits['extra_head_receives_nonzero_gradient']=any(p.grad is not None and bool(p.grad.abs().max()>0) for p in module.parameters())
            assert all(audits.values()),audits
            if rank==0: write(output/'gradient_audit.json',audits)
        optimizer.step()
        if step==1 or step%10==0 or step==steps:
            if world>1: torch.distributed.all_reduce(values)
            if rank==0:
                values=(values/world).tolist()
                elapsed=time.perf_counter()-loop_started
                status={'status':'running','method':config['method'],'step':step,'total':steps,
                        **dict(zip(['loss','ce','pair','invariance','composition','consistency'],values)),
                        'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-started,
                        'training_seconds':elapsed,'step_seconds':time.perf_counter()-tick,
                        'eta_seconds':elapsed/step*(steps-step),
                        'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
                append(output/'loss.jsonl',status)
                write(output/'status.json',status)
                print(f"{config['method'].upper()} {step}/{steps} loss={status['loss']:.5f} CE={status['ce']:.5f} ETA={status['eta_seconds']:.0f}s",flush=True)
        if (step==steps or step==500) and rank==0:
            dest=(output/'checkpoint') if smoke_steps else Path(config['checkpoint_dir'])/('final' if step==steps else 'step_0500')
            save(network,lc,optimizer,config,dest,step,step==steps)
            write(output/'status.json',{**status,'status':'complete' if step==steps else 'running','checkpoint':str(dest)})
    if world>1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args()
    train(training_config(args.config),args.smoke_steps)
