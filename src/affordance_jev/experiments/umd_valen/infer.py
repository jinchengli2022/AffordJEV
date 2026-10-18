"""Run fixed-candidate Valen variants and preserve inspectable inputs/outputs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from ...common import read,write,read_jsonl,sha,load_mask,save_mask
from ...decision.valen import Valen
from ...pipeline.artifacts import merge_results,verify_prepared
from ...decision.regions import decode,merge
from ...visualization import overlay,report


def compact_merge(sample):
    """Keep every numeric output and one overlay; avoid eight duplicate RGB PNGs."""
    started=time.perf_counter()
    verify_prepared(sample)
    response=read(sample/'valen_response.json')
    assert response['request_sha256']==sha(sample/'valen_request.json')
    candidates=read(sample/'candidates.json')['candidates']
    definitions=read(sample/'affordances_snapshot.json')
    foreground=load_mask(sample/'foreground/object_mask.png')
    regions=[load_mask(sample/r['mask'],foreground.shape) for r in candidates]
    decisions=decode(response,candidates,definitions)
    merged,class_map=merge(regions,decisions,foreground,definitions,whole_fraction=None)
    result=sample/'result'
    result.mkdir(exist_ok=True)
    Image.fromarray(class_map).save(result/'class_map.png')
    masks=[]
    for c in definitions['classes']:
        save_mask(result/f"masks/{c['name']}.png",merged[c['name']])
        masks.append({'class':c['name'],'id':c['id'],'mask':f"masks/{c['name']}.png",
                      'pixels':int(merged[c['name']].sum()),
                      'region_ids':[d['region_id'] for d in decisions if d['final_choice']==c['name']]})
    rgb=Image.open(sample/'rgb.png').convert('RGB')
    overlay(rgb,[merged[c['name']] for c in definitions['classes']],
            colors=[c['color'] for c in definitions['classes']],
            labels=[c['name'] for c in definitions['classes']]).save(result/'affordance_overlay.png')
    write(sample/'decisions.json',{'mode':'single_label','regions':decisions,
          'note':'Probabilities are model scores, not measured accuracy.'})
    write(result/'masks.json',{'size_wh':list(rgb.size),'classes':masks,'class_map':'class_map.png',
          'coordinate_system':'original RGB pixels','background_unassigned_none_id':0,
          'none_mask':'masks/none.png','unassigned_foreground_mask':'../regions/unassigned_foreground.png'})
    union=np.logical_or.reduce(regions) if regions else np.zeros_like(foreground)
    write(sample/'audit.json',{'region_count':len(regions),'overlap_pixels':0,'outside_object_pixels':0,
          'class_map_matches_masks':True,'all_candidate_pixels_accounted_for':True,
          'reference_annotations_used':False,'whole_object_fraction':None,
          'predicted_foreground_coverage':float(union.sum()/max(1,foreground.sum()))})
    write(sample/'timings.json',{'merge_and_save_seconds':time.perf_counter()-started,
          'partglee':read(sample/'partglee_runtime.json'),'prepare':read(sample/'prepare_runtime.json'),
          'valen':read(sample/'valen_runtime.json'),
          'note':'Staged offline inference; use separate warmed benchmark for latency.'})
    write(sample/'run.json',{'status':'complete','reference_annotations_used':False})
    report(sample,'推理完成；未使用 GT')


def load_adapter(model, checkpoint):
    from peft import LoraConfig,inject_adapter_in_model,set_peft_model_state_dict
    from safetensors.torch import load_file
    checkpoint=Path(checkpoint)
    config=LoraConfig.from_pretrained(checkpoint/'adapter')
    inject_adapter_in_model(config,model.backbone)
    result=set_peft_model_state_dict(model.backbone,load_file(checkpoint/'adapter/adapter_model.safetensors'))
    if result.unexpected_keys or any('lora_' in k for k in result.missing_keys):
        raise RuntimeError(f'Adapter checkpoint mismatch: {result}')
    model.head.load_state_dict(load_file(checkpoint/'head.safetensors'),strict=True)
    model.eval()
    if (checkpoint/'training_config.json').exists():
        from .abc.inference import install_region_head
        install_region_head(model,checkpoint)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--merge-workers',type=int,default=6)
    parser.add_argument('--merge-only',action='store_true')
    parser.add_argument('--full-artifacts',action='store_true')
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    all_jobs=read_jsonl(root/'model_jobs.jsonl')
    jobs=all_jobs[args.shard_index::args.shard_count]
    suffix=f'_{args.shard_index}' if args.shard_count>1 else ''
    config=read(root/'config_snapshot.json')['valen']
    state=read(root/'run.json')
    if args.shard_count==1: write(root/'run.json',{**state,'status':'running'})
    started=time.perf_counter()
    model=None if args.merge_only else Valen(jobs[0]['valen_model'],config)
    if args.adapter and model is not None:
        load_adapter(model.model,args.adapter)
        checkpoint_info={'checkpoint':str(Path(args.adapter).resolve()),
              'head_sha256':sha(Path(args.adapter)/'head.safetensors'),
              'adapter_sha256':sha(Path(args.adapter)/'adapter/adapter_model.safetensors')}
        for name in ['extra_heads.safetensors','training_config.json']:
            path=Path(args.adapter)/name
            if path.exists(): checkpoint_info[name+'_sha256']=sha(path)
        write(root/f'checkpoint_used{suffix}.json',checkpoint_info)
    if model is not None:
        write(root/f'model_load{suffix}.json',{'seconds':model.load_seconds,'model':model.model_path,'adapter':args.adapter})
    for i,job in enumerate(jobs,1):
        sample=Path(job['output'])
        current=read(sample/'run.json')['status']
        if current in ['classified','complete']: continue
        if args.merge_only:
            raise ValueError(f'Missing classified input for merge-only: {sample}: {current}')
        try:
            model.predict(sample)
            write(sample/'run.json',{'status':'classified','reference_annotations_used':False})
        except Exception as exc:
            traceback.print_exc()
            write(sample/'run.json',{'status':'failed','stage':'decision','error':str(exc)})
        if i%25==0 or i==len(jobs):
            write(root/f'progress{suffix}.json',{'stage':'inference','completed':i,'total':len(jobs),
                  'elapsed_seconds':time.perf_counter()-started})
            print(f'INFER {i}/{len(jobs)} {root}',flush=True)
    del model
    def finish(job):
        sample=Path(job['output'])
        if read(sample/'run.json')['status']=='classified':
            try:
                if args.full_artifacts: merge_results(sample)
                else: compact_merge(sample)
            except Exception as exc:
                traceback.print_exc()
                write(sample/'run.json',{'status':'failed','stage':'merge','error':str(exc)})
        return read(sample/'run.json')['status']
    counts=Counter()
    with ThreadPoolExecutor(max_workers=args.merge_workers) as pool:
        for i,status in enumerate(pool.map(finish,jobs),1):
            counts[status]+=1
            if i%100==0 or i==len(jobs):
                write(root/f'progress{suffix}.json',{'stage':'merge','completed':i,'total':len(jobs)})
                print(f'MERGE {i}/{len(jobs)} {root}',flush=True)
    write(root/('run.json' if args.shard_count==1 else f'worker_{args.shard_index}.json'),{**state,'status':'complete' if counts['complete']==len(jobs) else 'failed',
          'counts':dict(counts),'adapter':args.adapter,'batch_wall_seconds':time.perf_counter()-started})
    print(f'COMPLETE {dict(counts)} {root}',flush=True)


if __name__=='__main__': main()
