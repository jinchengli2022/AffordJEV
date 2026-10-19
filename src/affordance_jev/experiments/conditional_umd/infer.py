"""Inference workers consume RGB/category/action jobs, never annotations."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from ...common import read, write, read_jsonl, sha, load_mask, save_mask
from ...paths import PROJECT, VALEN_MODEL
from ...pipeline.run import sample_directory
from ...visualization import overlay
from .requests import METHODS, prepare_assets, prepare_query, decode


def image_groups(queries, shard_index, shard_count, limit=0):
    groups = defaultdict(list)
    for q in queries:
        assert set(q) == {'sample_id','query_id','rgb','rgb_sha256','size_wh','object_category','class_id','action'}
        groups[q['sample_id']].append(q)
    values = list(groups.values())
    if limit: values = values[:limit]
    return values[shard_index::shard_count]


def save_result(out, query, mask, *, error=None, timing=None, decisions=None):
    out.mkdir(parents=True, exist_ok=True)
    assert mask.shape == tuple(query['size_wh'][::-1])
    save_mask(out/'mask.png', mask)
    rgb = Image.open(query['rgb']).convert('RGB')
    overlay(rgb,[mask],colors=[[60,180,95]],labels=['']).save(out/'overlay.jpg',quality=90)
    write(out/'query.json',query)
    if decisions is not None: write(out/'decisions.json',decisions)
    if timing is not None: write(out/'timing.json',timing)
    write(out/'run.json',{'status':'failed' if error else 'complete','error':error,
          'mask_sha256':sha(out/'mask.png'), 'reference_annotations_used':False})


def valen(args):
    from ...decision.valen import Valen
    from ..umd_valen.infer import load_adapter
    root=Path(args.root).resolve()
    definitions=read(root/'definitions.json')
    groups=image_groups(read_jsonl(root/'test_queries.jsonl'),args.shard_index,args.shard_count,args.limit)
    model=Valen(str(VALEN_MODEL),read(root/'pipeline.json')['valen'])
    if args.adapter:
        assert args.methods == ['sft']
        load_adapter(model.model,args.adapter)
    for method in args.methods:
        target=root/'methods'/method
        write(target/f'model_load_{args.shard_index}.json',{'model':str(VALEN_MODEL),
              'seconds':model.load_seconds,'adapter':args.adapter,
              'adapter_sha256':sha(Path(args.adapter)/'adapter/adapter_model.safetensors') if args.adapter else None,
              'head_sha256':sha(Path(args.adapter)/'head.safetensors') if args.adapter else None})
        started=time.perf_counter()
        counts=Counter()
        for i,queries in enumerate(groups,1):
            sid=queries[0]['sample_id']
            source=sample_directory(root/'candidates_test',sid)
            sample=sample_directory(target,sid)
            state=read(source/'run.json')
            wait_started=time.monotonic()
            while state['status']=='pending' and args.wait_candidates:
                if time.monotonic()-wait_started>7200:
                    raise TimeoutError(f'PartGLEE candidate wait exceeded two hours: {sid}')
                time.sleep(5)
                state=read(source/'run.json')
            if state['status']=='pending':
                raise RuntimeError(f'Candidate job has not run yet: {sid}')
            prepared=state['status'] in ['prepared','complete']
            if prepared:
                assets=sample/'inputs'
                if not assets.exists():
                    candidates=prepare_assets(source,assets,method)
                else:
                    # Resuming only this new run is allowed; no historic prediction source.
                    assert read(assets/'source.json')['candidate_file_sha256']==sha(source/'candidates.json')
                    candidates=read(source/'candidates.json')['candidates']
                masks=[load_mask(source/c['mask']) for c in candidates]
            else:
                candidates,masks=[],[]
            for q in queries:
                out=sample/'queries'/q['action']
                if (out/'run.json').exists() and read(out/'run.json')['status'] in ['complete','failed']:
                    assert read(out/'query.json')==q
                    counts[read(out/'run.json')['status']]+=1
                    continue
                tick=time.perf_counter()
                mask=np.zeros(tuple(q['size_wh'][::-1]),bool)
                error=None
                decisions=[]
                runtimes=[]
                try:
                    if not prepared: raise RuntimeError(f'PartGLEE failed: {state.get("error")}')
                    if sha(q['rgb'])!=q['rgb_sha256']: raise ValueError('RGB checksum changed')
                    requests=prepare_query(assets,out,q,candidates,definitions,method)
                    for request in requests:
                        folder=Path(request['folder'])
                        model.predict(folder)
                        response=read(folder/'valen_response.json')
                        assert response.get('usage',{}).get('output_tokens',0)==0
                        decisions.extend(decode(response,request['candidates']))
                        runtimes.append(read(folder/'valen_runtime.json'))
                    assert len(decisions)==len(candidates)
                    chosen={d['region_id'] for d in decisions if d['selected']}
                    for c,m in zip(candidates,masks):
                        if c['id'] in chosen: mask |= m
                except Exception as exc:
                    traceback.print_exc()
                    error=f'{type(exc).__name__}: {exc}'
                    mask[:]=False
                timing={'valen_predict_seconds':sum(r['timings']['predict_seconds'] for r in runtimes),
                    'compiled_input_save_seconds':sum(r['timings']['compiled_input_save_seconds_inside_predict'] for r in runtimes),
                    'forward_calls':sum(r['request_questions']>0 for r in runtimes),
                    'request_prepare_and_predict_seconds':time.perf_counter()-tick,
                    'scope':'per operation; model loading and PartGLEE excluded; includes preprocessing and prompt audit saving'}
                save_result(out,q,mask,error=error,timing=timing,decisions=decisions)
                counts['failed' if error else 'complete']+=1
            if i%10==0 or i==len(groups):
                write(target/f'progress_{args.shard_index}.json',{'images_done':i,'images_total':len(groups),
                    'queries':dict(counts),'elapsed_seconds':time.perf_counter()-started})
                print(f'VALEN {method} shard={args.shard_index} images={i}/{len(groups)} queries={dict(counts)}',flush=True)
        write(target/f'worker_{args.shard_index}.json',{'status':'complete','queries':dict(counts),
            'images':len(groups),'seconds':time.perf_counter()-started,'shard_count':args.shard_count})


def r1(args):
    from ...baselines.affordance_r1.predictor import Predictor
    root=Path(args.root).resolve()
    target=root/'methods/r1'
    cfg=read(PROJECT/'configs/affordance_r1_umd.json')
    cfg.update(protocol='given object and requested operation; independent mask',
               query_selection='only actions in frozen test_queries.jsonl',overlap_rule=None)
    write(target/f'config_{args.shard_index}.json',cfg)
    groups=image_groups(read_jsonl(root/'test_queries.jsonl'),args.shard_index,args.shard_count,args.limit)
    predictor=Predictor(args.model,args.sam_checkpoint,cfg,read(root/'definitions.json'))
    started=time.perf_counter()
    counts=Counter()
    for begin in range(0,len(groups),cfg['image_batch']):
        batch=groups[begin:begin+cfg['image_batch']]
        images=[]
        for qs in batch:
            assert sha(qs[0]['rgb'])==qs[0]['rgb_sha256']
            images.append(Image.open(qs[0]['rgb']).convert('RGB'))
        results=predictor.predict_many(images,[qs[0]['object_category'] for qs in batch],
                                      actions_per_image=[[q['action'] for q in qs] for qs in batch])
        for qs,(class_map,logits,responses,timing) in zip(batch,results):
            assert class_map is None and len(responses)==len(qs)
            for q,response in zip(qs,responses):
                assert response['class_id']==q['class_id']
                out=sample_directory(target,q['sample_id'])/'queries'/q['action']
                write(out/'response.json',response)
                (out/'prompt.txt').write_text(response['prompt'],encoding='utf-8')
                save_result(out,q,logits[q['class_id']-1]>0,error=response['error'],
                    timing={**timing,'shared_operations_this_rgb':len(qs),
                            'seconds_per_query_amortized':timing['rgb_to_requested_masks_seconds']/len(qs)})
                counts['failed' if response['error'] else 'complete']+=1
        done=min(begin+len(batch),len(groups))
        if begin%30==0 or done==len(groups):
            write(target/f'progress_{args.shard_index}.json',{'images_done':done,'images_total':len(groups),
                'queries':dict(counts),'elapsed_seconds':time.perf_counter()-started})
            print(f'R1 shard={args.shard_index} {done}/{len(groups)} {dict(counts)}',flush=True)
    write(target/f'worker_{args.shard_index}.json',{'status':'complete','queries':dict(counts),
            'images':len(groups),'seconds':time.perf_counter()-started,'shard_count':args.shard_count})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['valen','r1'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--methods',nargs='+',choices=list(METHODS),default=['fill','outline','definition','centroid'])
    parser.add_argument('--adapter')
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--wait-candidates',action='store_true',help='Stream freshly prepared candidates from this run')
    parser.add_argument('--model',default=str(PROJECT.parent/'2d_affordance/model/pretrained/affordance-r1/huggingface'))
    parser.add_argument('--sam-checkpoint',default=str(PROJECT.parent/'2d_affordance/model/pretrained/sam2-hiera-large/sam2_hiera_large.pt'))
    args=parser.parse_args()
    assert 0<=args.shard_index<args.shard_count
    globals()[args.mode](args)


if __name__=='__main__':
    main()
