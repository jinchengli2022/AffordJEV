"""Controlled diagnosis of foreground, image preprocessing and proposal filtering.

All model jobs are RGB + previously predicted foreground only. Evaluation is
GT-only and returns certified intervals for the union oracle. Production model,
foreground selection and partition settings are never changed by this script.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from ..common import read, read_jsonl, write, write_jsonl, sha, load_mask, padded_bbox
from .raw_candidates import ROOT, VARIANTS, load_proposals, now


BASELINES = {
    'specific': ROOT/'outputs/umd_novel_test',
    'generic': ROOT/'outputs/umd_given_operation_20261018/candidates_test',
}
RAW = ROOT/'outputs/umd_raw_candidate_oracle_20261019/full'
STAGES = [
    'full_joint', 'full_parts', 'full_parts_fg', 'grey_full_parts_fg',
    'crop_parts_fg', 'quality_filter', 'component_filter', 'partition_no_cap', 'final_cap6',
]


def sample_folder(variant, sid):
    return BASELINES[variant]/'samples'/sid


def save_proposals(folder, masks, quality, metadata):
    folder.mkdir(parents=True,exist_ok=True)
    n,h,w = masks.shape
    np.savez_compressed(folder/'masks.npz',packed=np.packbits(masks.reshape(n,-1),axis=1),
                        shape=np.array([n,h,w],np.int32))
    write(folder/'metadata.json',{**metadata,'qualities':quality,'sha256':sha(folder/'masks.npz')})


def infer_worker(args):
    import torch
    from ..segmentation.partglee import PartGLEE
    torch.set_num_threads(2)
    torch.manual_seed(42)
    out=Path(args.output)
    jobs=read_jsonl(args.jobs)
    protocol=read(out/'protocol.json')
    adapter=PartGLEE(ROOT/'third_party/PartGLEE',ROOT/'models/partglee/PartGLEE_SwinL.pth',
                     protocol['vocabularies']['generic'],Path(args.jobs).parent)
    for number,job in enumerate(jobs,1):
        # The worker never reads annotations or operation queries.
        assert set(job)=={'sample_id','rgb','rgb_sha256','size_wh','foregrounds'}
        assert sha(job['rgb'])==job['rgb_sha256']
        rgb=Image.open(job['rgb']).convert('RGB')
        w,h=rgb.size
        for mode,variant,foreground_variant,crop in [
            ('specific_grey_full','specific','specific',False),
            ('generic_grey_full','generic','generic',False),
            ('generic_old_foreground_crop','generic','specific',True),
        ]:
            info=job['foregrounds'][foreground_variant]
            assert sha(info['mask'])==info['sha256']
            fg=load_mask(info['mask'],(h,w))
            grey=np.asarray(rgb).copy(); grey[~fg]=127
            box=padded_bbox(fg,.1) if crop else [0,0,w,h]
            x0,y0,x1,y1=box
            image=Image.fromarray(grey).crop(box)
            adapter.vocabulary=protocol['vocabularies'][variant]
            adapter.labels=adapter.vocabulary['objects']+adapter.vocabulary['parts']
            small,quality,labels,elapsed=adapter.predict(image,'parts')
            masks=np.zeros((50,h,w),bool); masks[:,y0:y1,x0:x1]=small
            save_proposals(out/'generated'/mode/job['sample_id'],masks,quality,{
                'sample_id':job['sample_id'],'mode':mode,'vocabulary':variant,
                'foreground_source':foreground_variant,'foreground_sha256':info['sha256'],
                'rgb_sha256':job['rgb_sha256'],'crop_xyxy':box,'labels':labels,
                'forward_seconds':elapsed,'gt_used':False,'local_proposal_filtering':False,
            })
        print(f"Inference {number}/{len(jobs)} {job['sample_id']} 3 controls saved",flush=True)
    write(Path(args.jobs).parent/'finished.json',{'status':'complete','images':len(jobs),'finished':now()})


def stacked(masks,shape):
    return np.stack(masks) if len(masks) else np.zeros((0,*shape),bool)


def candidate_stages(row,out,variant):
    """Reconstruct stages with unchanged cached masks, except named new controls."""
    from ..segmentation.partition import partition,remove_small
    sid=row['sample_id']; shape=tuple(row['size_wh'][::-1])
    folder=sample_folder(variant,sid)
    cfg=read(folder/'config_snapshot.json')
    assert cfg['partition']==read(ROOT/'configs/umd_pipeline_generic.json')['partition']
    fg=load_mask(folder/'foreground/object_mask.png',shape)
    meta=read(folder/'foreground/metadata.json')
    assert sha(folder/'foreground/object_mask.png')==meta['mask_sha256']
    rgb=np.asarray(Image.open(row['rgb']).convert('RGB'))
    assert np.array_equal(rgb,np.asarray(Image.open(folder/'rgb.png').convert('RGB')))
    full=load_proposals(RAW/'proposals'/variant/sid)
    with np.load(folder/'raw/proposals.npz',allow_pickle=False) as data:
        crop=data['masks'].astype(bool); quality=data['qualities'].astype(np.float32)
    assert crop.shape==(50,*shape)
    grey=load_proposals(out/'generated'/f'{variant}_grey_full'/sid)
    minimum=max(cfg['partition']['minimum_component_pixels'],int(fg.sum()*cfg['partition']['minimum_component_fraction']))
    # Filter steps are independently exposed; pixel assignment uses production code.
    clipped=crop&fg
    passed=np.flatnonzero(quality>=cfg['partition']['minimum_quality'])
    cleaned=[remove_small(clipped[i],minimum) for i in passed]
    cleaned=[m for m in cleaned if m.sum()>=minimum]
    no_cap,_,_=partition(crop,quality,fg,{**cfg['partition'],'max_regions':50})
    final,_,trace=partition(crop,quality,fg,cfg['partition'])
    label_map=np.zeros(shape,np.uint16)
    for i,m in enumerate(final,1): label_map[m]=i
    assert np.array_equal(label_map,np.asarray(Image.open(folder/'regions/label_map.png'))),sid
    historical_trace=read(folder/'regions/filter_trace.json')
    assert trace['minimum_component_pixels']==historical_trace['minimum_component_pixels']
    stages={
        'full_parts':full[50:], 'full_parts_fg':full[50:]&fg,
        'grey_full_parts_fg':grey&fg, 'crop_parts_fg':clipped,
        'quality_filter':clipped[passed], 'component_filter':stacked(cleaned,shape),
        'partition_no_cap':stacked(no_cap,shape), 'final_cap6':stacked(final,shape),
        'crop_parts_unclipped':crop,
    }
    metadata={'minimum_component_pixels':minimum,'foreground_area':int(fg.sum()),
              'crop_xyxy':meta['crop_xyxy'],'final_reproduced':True,
              'candidate_counts':{k:len(v) for k,v in stages.items()}}
    if variant=='generic':
        old_fg=load_mask(sample_folder('specific',sid)/'foreground/object_mask.png',shape)
        new_folder=out/'generated/generic_old_foreground_crop'/sid
        repaired=load_proposals(new_folder)
        scores=read(new_folder/'metadata.json')['qualities']
        repaired_final,_,_=partition(repaired,scores,old_fg,cfg['partition'])
        stages['old_foreground_crop_raw']=repaired&old_fg
        stages['old_foreground_final']=stacked(repaired_final,shape)
    return stages,fg,metadata


def score_masks(masks,gt,budget):
    from ..evaluation.raw_oracle import pixel_atoms,union_oracle
    cids=[i for i in range(1,8) if np.any(gt==i)]
    coverage=masks.any(0)
    active=coverage&(gt>=0)
    if not active.any():
        return {cid:{'lower':0.,'upper':0.,'certified':True,'chosen':[],
                     'covered_gt_pixels':0,'gt_pixels':int((gt==cid).sum())} for cid in cids}
    # All-zero membership pixels never change any candidate union. Omit them
    # from the optimizer, but retain ALL GT pixels in each IoU denominator.
    membership,counts,_=pixel_atoms(masks[:,active][:,None,:],gt[active][None,:])
    totals={cid:int((gt==cid).sum()) for cid in cids}
    return {cid:{**union_oracle(membership,counts,cid,totals[cid],seconds=budget),
                 'covered_gt_pixels':int((coverage&(gt==cid)).sum()),'gt_pixels':totals[cid]}
            for cid in cids}


def evaluate_image(row,out,budget):
    import cv2
    from ..datasets.umd import CLASS_NAMES,load_annotation
    cv2.setNumThreads(1)
    out=Path(out); sid=row['sample_id']
    assert sha(row['annotation'])==row['annotation_sha256']
    gt,kind=load_annotation(row['annotation'],row['size_wh'][::-1]); assert kind=='manual'
    query_rows=[]; image_rows=[]
    cached=read(RAW/'evaluation/samples'/sid/'metrics.json')
    for variant in VARIANTS:
        stages,fg,meta=candidate_stages(row,out,variant)
        box=np.zeros_like(fg); x0,y0,x1,y1=meta['crop_xyxy']; box[y0:y1,x0:x1]=True
        image_rows.append({'sample_id':sid,'variant':variant,**meta,
            'gt_pixels':int((gt>0).sum()),'foreground_covered_gt':int((fg&(gt>0)).sum()),
            'crop_bbox_covered_gt':int((box&(gt>0)).sum())})
        results={stage:score_masks(masks,gt,budget) for stage,masks in stages.items()}
        for prior in [q for q in cached['queries'] if q['variant']==variant]:
            cid=prior['class_id']
            all_stages={'full_joint':prior['union_oracle'],**{k:v[cid] for k,v in results.items()}}
            query_rows.append({'sample_id':sid,'query_id':prior['query_id'],'variant':variant,
                'class_id':cid,'action':CLASS_NAMES[cid],'gt_pixels':int((gt==cid).sum()),
                'foreground_recall':float((fg&(gt==cid)).sum()/(gt==cid).sum()),
                'crop_bbox_recall':float((box&(gt==cid)).sum()/(gt==cid).sum()),'stages':all_stages})
    write(out/'evaluation/samples'/sid/'metrics.json',{'images':image_rows,'queries':query_rows})
    return image_rows,query_rows


def aggregate(images,queries):
    result={}
    for variant in VARIANTS:
        ims=[r for r in images if r['variant']==variant]
        qs=[q for q in queries if q['variant']==variant]
        names=list(qs[0]['stages'])
        def one(rows):
            return {name:{'lower':float(np.mean([q['stages'][name]['lower'] for q in rows])),
                          'upper':float(np.mean([q['stages'][name]['upper'] for q in rows])),
                          'certified_queries':sum(q['stages'][name]['certified'] for q in rows),
                          'coverage':sum(q['stages'][name].get('covered_gt_pixels',0) for q in rows)/sum(q['gt_pixels'] for q in rows)
                          if name!='full_joint' else None}
                    for name in names}
        result[variant]={
            'images':len(ims),'queries':len(qs),'stages':one(qs),
            'foreground_gt_coverage':sum(r['foreground_covered_gt'] for r in ims)/sum(r['gt_pixels'] for r in ims),
            'crop_bbox_gt_coverage':sum(r['crop_bbox_covered_gt'] for r in ims)/sum(r['gt_pixels'] for r in ims),
            'mean_foreground_query_recall':float(np.mean([q['foreground_recall'] for q in qs])),
            'mean_small_component_threshold':float(np.mean([r['minimum_component_pixels'] for r in ims])),
            'mean_candidates':{name:float(np.mean([r['candidate_counts'][name] for r in ims])) for name in ims[0]['candidate_counts']},
            'by_category':{cat:one([q for q in qs if q['sample_id'].split('/')[0].rsplit('_',1)[0]==cat])
                for cat in sorted({q['sample_id'].split('/')[0].rsplit('_',1)[0] for q in qs})},
        }
    return result


def run(args):
    out=Path(args.output).resolve()
    if out.exists(): raise FileExistsError(out)
    manifest=ROOT/'data/UMD/manifests/umd_novel_test.jsonl'
    rows=read_jsonl(manifest)
    if args.smoke:
        selected={}
        for row in rows: selected.setdefault(row['object_instance'].rsplit('_',1)[0],row)
        rows=list(selected.values())
    out.mkdir(parents=True)
    protocol={'started':now(),'manifest':str(manifest),'manifest_sha256':sha(manifest),
        'images':len(rows),'stage_order':STAGES,'oracle_seconds':args.oracle_seconds,
        'vocabularies':read(RAW/'protocol.json')['vocabularies'],
        'historical_sources':{v:str(p) for v,p in BASELINES.items()},'full_raw_source':str(RAW),
        'new_controls':['full-frame foreground-greyed RGB (each vocabulary)',
                        'generic vocabulary + specific-run foreground + unchanged crop/partition'],
        'gt_used_in_inference':False,
        'interpretation':'Sequential conditional effects, not independent additive causal contributions. Foreground replacement includes its downstream interactions.',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/affordance_jev/evaluation/raw_oracle.py',ROOT/'src/affordance_jev/segmentation/partition.py',ROOT/'src/affordance_jev/segmentation/partglee.py']}}
    write(out/'protocol.json',protocol); write_jsonl(out/'manifest.jsonl',rows)
    jobs=[]
    for row in rows:
        jobs.append({**{k:row[k] for k in ['sample_id','rgb','rgb_sha256','size_wh']},
            'foregrounds':{v:{'mask':str(sample_folder(v,row['sample_id'])/'foreground/object_mask.png'),
                'sha256':sha(sample_folder(v,row['sample_id'])/'foreground/object_mask.png')} for v in VARIANTS}})
    started=time.perf_counter(); processes=[]
    write(out/'status.json',{'status':'running','stage':'inference','images':len(rows),'pid':os.getpid(),'updated':now()})
    try:
        devices=args.gpus.split(',')
        for i,gpu in enumerate(devices):
            selected=jobs[i::len(devices)]
            if not selected: continue
            path=out/f'workers/{i}/jobs.jsonl'; write_jsonl(path,selected)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
            with open(path.parent/'run.log','w') as log:
                processes.append(subprocess.Popen([sys.executable,'-m','affordance_jev.experiments.candidate_stage_diagnosis','worker',
                    '--output',str(out),'--jobs',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in processes]
        if any(codes): raise RuntimeError(f'Inference workers failed: {codes}')
        print(f'Inference complete, {time.perf_counter()-started:.1f}s',flush=True)
        write(out/'status.json',{'status':'running','stage':'evaluation','images':len(rows),'pid':os.getpid(),'updated':now()})
        images=[];queries=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(evaluate_image,row,str(out),args.oracle_seconds) for row in rows]
            for number,future in enumerate(as_completed(futures),1):
                ims,qs=future.result(); images.extend(ims);queries.extend(qs)
                if number%100==0 or number==len(rows):
                    print(f'Oracle {number}/{len(rows)}, elapsed {time.perf_counter()-started:.1f}s',flush=True)
                    write(out/'status.json',{'status':'running','stage':'evaluation','images':len(rows),'completed':number,'pid':os.getpid(),'updated':now()})
        images.sort(key=lambda r:(r['sample_id'],r['variant'])); queries.sort(key=lambda r:(r['query_id'],r['variant']))
        write_jsonl(out/'evaluation/per_image.jsonl',images);write_jsonl(out/'evaluation/per_query.jsonl',queries)
        expected=[q for q in read_jsonl(RAW/'queries.jsonl') if q['sample_id'] in {r['sample_id'] for r in rows}]
        for variant in VARIANTS:
            actual=[q for q in queries if q['variant']==variant]
            assert len(actual)==len(expected) and {q['query_id'] for q in actual}=={q['query_id'] for q in expected}
        summary={'status':'complete','images':len(rows),'queries':len(expected),'finished':now(),
            'wall_seconds':time.perf_counter()-started,'metrics':aggregate(images,queries)}
        write(out/'evaluation/metrics.json',summary)
        write(out/'status.json',{k:v for k,v in summary.items() if k!='metrics'})
        print({v:summary['metrics'][v]['stages'] for v in VARIANTS},flush=True)
    except BaseException as exc:
        for process in processes:
            if process.poll() is None: process.terminate()
        write(out/'status.json',{'status':'failed','error':str(exc),'updated':now()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('run'); p.add_argument('--output',required=True);p.add_argument('--gpus',default='0,1,2,3,4,5,6')
    p.add_argument('--workers',type=int,default=24);p.add_argument('--oracle-seconds',type=float,default=3.);p.add_argument('--smoke',action='store_true')
    p=sub.add_parser('worker');p.add_argument('--output',required=True);p.add_argument('--jobs',required=True)
    args=parser.parse_args(); (run if args.mode=='run' else infer_worker)(args)
