"""Warmed serial RGB-to-mask timing with both models resident on one GPU.

Two Conda workers communicate locally. Only the three actual Valen input images
are materialized; visualization/result dumps and weight loading are excluded.
"""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from ...common import read,write,read_jsonl,sha,padded_bbox
from ...decision.regions import request_for,decode,merge
from ...paths import PROJECT, PARTGLEE_ROOT, PARTGLEE_WEIGHTS, VALEN_MODEL, conda_python


def segmentation_worker(args):
    import torch
    from ...segmentation.partglee import PartGLEE
    from ...segmentation.foreground import select_object
    from ...segmentation.partition import partition
    from ...visualization import overlay,sheet
    config=read(PROJECT/'configs/umd_pipeline.json')
    scratch=Path(args.scratch)
    scratch.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    with redirect_stdout(sys.stderr):
        adapter=PartGLEE(PARTGLEE_ROOT,PARTGLEE_WEIGHTS,
                         config['vocabulary'],scratch)
    print('READY',flush=True)
    for line in sys.stdin:
        job=json.loads(line)
        if job.get('stop'): break
        started=time.perf_counter()
        rgb=Image.open(job['rgb']).convert('RGB')
        with redirect_stdout(sys.stderr):
            object_masks,scores,labels,_=adapter.predict(rgb,'objects')
            index,_=select_object(object_masks,scores,config['foreground'])
            foreground=object_masks[index]
            crop=padded_bbox(foreground,config['foreground']['crop_padding_fraction'])
            x0,y0,x1,y1=crop
            masked=np.array(rgb)
            masked[~foreground]=config['foreground']['background_rgb']
            local,quality,labels,_=adapter.predict(Image.fromarray(masked).crop(crop),'parts')
        masks=np.zeros((len(local),rgb.height,rgb.width),bool)
        masks[:,y0:y1,x0:x1]=local
        regions,_,_=partition(masks,quality,foreground,config['partition'])
        alpha=0 if job['style']=='outline' else .45
        (scratch/'foreground').mkdir(exist_ok=True)
        (scratch/'regions').mkdir(exist_ok=True)
        rgb.crop(crop).save(scratch/'foreground/rgb_crop.png')
        overlay(rgb,regions,fill_alpha=alpha).crop(crop).save(scratch/'regions/numbered_crop.png')
        if job['style']=='centroid':
            from .centroid import render_queries
            render_queries(scratch,rgb.crop(crop),regions,crop,{},
                           [{'id':i} for i in range(1,len(regions)+1)],save_audit=False)
        else:
            panels=[(f'Region {i}',overlay(rgb,[m],labels=[str(i)],fill_alpha=alpha).crop(crop))
                    for i,m in enumerate(regions,1)]
            sheet(panels,scratch/'regions/region_sheet.jpg',columns=3)
        # Binary geometry is passed to the merger, not the Valen visual backbone.
        np.savez(scratch/'geometry.npz',foreground=foreground,crop_xyxy=np.asarray(crop),
                 masks=np.stack(regions) if regions else np.zeros((0,rgb.height,rgb.width),bool))
        torch.cuda.synchronize()
        print('RESULT '+json.dumps({'regions':len(regions),'seconds':time.perf_counter()-started}),flush=True)


def stats(values):
    return {'mean':float(np.mean(values)),'p50':float(np.percentile(values,50)),
            'p95':float(np.percentile(values,95))}


def main_benchmark(args):
    import torch
    from ...decision.valen import Valen
    from .infer import load_adapter
    root=Path(args.root).resolve()
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'metrics.json').exists(): raise FileExistsError(output/'metrics.json')
    scratch=output/'scratch'
    definitions=read(root/'affordances_snapshot.json')
    config=read(root/'config_snapshot.json')
    style=read(root/'run.json').get('style','fill')
    if style not in ['fill','outline','centroid']:
        raise ValueError(f'Unknown visual input style: {style}')
    valen_config={**config['valen'],'save_compiled_inputs':False}
    model=Valen(str(VALEN_MODEL),valen_config)
    if args.adapter: load_adapter(model.model,args.adapter)
    environment={**os.environ,'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    log=(output/'segmentation_worker.log').open('w')
    worker=subprocess.Popen([str(args.partglee_python),'-m',
        'affordance_jev.experiments.umd_valen.benchmark','--segmentation-worker','--scratch',str(scratch)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=environment)
    def receive(prefix):
        while True:
            line=worker.stdout.readline()
            if not line: raise RuntimeError(f'Segmentation worker exited: {worker.poll()}; see {output}')
            if line.startswith(prefix): return line[len(prefix):].strip()
    receive('READY')
    rows=read_jsonl(root/'evaluation_manifest.jsonl')
    selected=[rows[i] for i in np.linspace(0,len(rows)-1,args.samples,dtype=int)]
    measurements=[]
    try:
        for index,row in enumerate(selected[:5]+selected):
            torch.cuda.synchronize()
            started=time.perf_counter()
            worker.stdin.write(json.dumps({'rgb':row['rgb'],'style':style})+'\n')
            worker.stdin.flush()
            segmentation=json.loads(receive('RESULT '))
            candidates=[{'id':i} for i in range(1,segmentation['regions']+1)]
            if style=='centroid':
                from .centroid import centroid_request
                requests=[centroid_request(c['id'],definitions) for c in candidates]
            else:
                request=request_for(candidates,definitions,include_sheet=True)
            decision_started=time.perf_counter()
            if candidates and style=='centroid':
                answers={}
                for candidate,request in zip(candidates,requests):
                    result=model.model.predict(request,media_root=str(scratch/f"centroid_queries/{candidate['id']:02}"),
                                               execution='shared_state',temperature=1.0)
                    answers.update(result['answers'])
                response={'answers':answers}
            elif candidates:
                response=model.model.predict(request,media_root=str(scratch),execution='shared_state',temperature=1.0)
            else: response={'answers':{}}
            torch.cuda.synchronize()
            valen_seconds=time.perf_counter()-decision_started
            with np.load(scratch/'geometry.npz',allow_pickle=False) as data:
                masks,foreground=data['masks'],data['foreground']
            decisions=decode(response,candidates,definitions)
            _,prediction=merge(masks,decisions,foreground,definitions,whole_fraction=None)
            total_seconds=time.perf_counter()-started
            if index<5: continue
            previous=np.asarray(Image.open(root/'samples'/row['sample_id']/'result/class_map.png'))
            cached_regions=np.asarray(Image.open(root/'samples'/row['sample_id']/'regions/label_map.png'))
            live_regions=np.zeros(foreground.shape,np.uint16)
            for region_id,mask in enumerate(masks,1): live_regions[mask]=region_id
            m={'sample_id':row['sample_id'],'regions':len(candidates),'end_to_end_seconds':total_seconds,
               'partglee_and_input_preparation_seconds':segmentation['seconds'],
               'valen_seconds':valen_seconds,'matches_saved_prediction':bool(np.array_equal(prediction,previous)),
               'prediction_pixel_disagreements':int((prediction!=previous).sum()),
               'candidate_pixel_disagreements':int((live_regions!=cached_regions).sum()),
               'live_choices':[d['final_choice'] for d in decisions],
               'cached_choices':[d['final_choice'] for d in read(root/'samples'/row['sample_id']/'decisions.json')['regions']]}
            paths=['foreground/rgb_crop.png','regions/numbered_crop.png']
            if style=='centroid':
                paths += [f"centroid_queries/{c['id']:02}/centroid.png" for c in candidates]
                m['request_matches_saved']=all(request==read(root/'samples'/row['sample_id']/f"centroid_queries/{c['id']:02}/valen_request.json")
                                               for c,request in zip(candidates,requests))
                m['valen_forward_calls']=len(candidates)
            else:
                paths.append('regions/region_sheet.jpg')
                m['request_matches_saved']=request==read(root/'samples'/row['sample_id']/'valen_request.json')
            m['live_input_sha256']={p:sha(scratch/p) for p in paths}
            m['cached_input_sha256']={p:sha(root/'samples'/row['sample_id']/p) for p in paths}
            measurements.append(m)
            if len(measurements)%20==0:
                print(f'BENCH {len(measurements)}/{len(selected)} {output}',flush=True)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n')
            worker.stdin.flush()
            try: worker.wait(timeout=30)
            except subprocess.TimeoutExpired: worker.terminate()
        log.close()
    summary={'samples':len(measurements),'warmup_images':5,'batch':1,'same_gpu_for_both_models':True,
             'gpu':torch.cuda.get_device_name(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
             'style':style,'adapter':args.adapter,'source_root':str(root),
             'scope':'RGB read -> PartGLEE object/part inference -> partition -> input images -> all Valen queries -> merged class map; includes local IPC and image transfer; excludes weight load and result/viz saving',
             'valen_calls_per_rgb':'one per candidate' if style=='centroid' else 'one shared-state call',
             'matching_saved_predictions':sum(m['matches_saved_prediction'] for m in measurements),
             'total_prediction_pixel_disagreements':sum(m['prediction_pixel_disagreements'] for m in measurements),
             'total_candidate_pixel_disagreements':sum(m['candidate_pixel_disagreements'] for m in measurements),
             'timing_seconds':{k:stats([m[k] for m in measurements]) for k in [
                 'end_to_end_seconds','partglee_and_input_preparation_seconds','valen_seconds']},
             'manifest_sha256':sha(root/'evaluation_manifest.jsonl')}
    write(output/'metrics.json',summary)
    from ...common import write_jsonl
    write_jsonl(output/'per_image.jsonl',measurements)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--segmentation-worker',action='store_true')
    parser.add_argument('--scratch')
    parser.add_argument('--root')
    parser.add_argument('--output')
    parser.add_argument('--adapter')
    parser.add_argument('--samples',type=int,default=200)
    parser.add_argument('--partglee-python',default=str(conda_python('partglee')))
    args=parser.parse_args()
    if args.segmentation_worker: segmentation_worker(args)
    else: main_benchmark(args)


if __name__=='__main__': main()
