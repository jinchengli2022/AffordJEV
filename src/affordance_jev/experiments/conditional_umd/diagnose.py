"""Separate candidate geometry from decisions using frozen outputs and crossed runs."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ...common import read, write, read_jsonl, write_jsonl, sha
from ...datasets.umd import CLASS_NAMES, load_annotation
from ...baselines.affordance_r1.compare import query_score, summarize_queries
from ...evaluation.diagnose import best_union
from ...paths import PROJECT


OLD = PROJECT/'outputs/umd_visual_prompt_ablation/fill'


def geometry(root):
    root=Path(root).resolve()
    dest=root/'diagnosis'
    sources={'specific':OLD,'generic':root/'candidates_test'}
    per_image=[]; per_query=[]; region_rows=[]
    majority_scores=defaultdict(list)
    binary_counts={m:np.zeros((2,2),np.int64) for m in ['fill','outline','centroid','definition','sft']}
    pixel_counts={m:defaultdict(int) for m in binary_counts}
    rows=read_jsonl(root/'test_manifest.jsonl')
    allowed={(q['sample_id'],q['class_id']) for q in read_jsonl(root/'test_queries.jsonl')}
    for index,row in enumerate(rows,1):
        gt,kind=load_annotation(row['annotation'],tuple(row['size_wh'][::-1]))
        assert kind=='manual' and sha(row['annotation'])==row['annotation_sha256']
        positive=gt>0
        category=row['object_instance'].rsplit('_',1)[0]
        for source,folder in sources.items():
            sample=folder/'samples'/row['sample_id']
            regions=read(sample/'candidates.json')['candidates']
            labels=np.asarray(Image.open(sample/'regions/label_map.png'))
            foreground=np.asarray(Image.open(sample/'foreground/object_mask.png'))>0
            assert gt.shape==labels.shape==foreground.shape
            assert not ((labels>0)&~foreground).any()
            n=len(regions)
            assert [r['id'] for r in regions]==list(range(1,n+1))
            assert labels.max()<=n
            valid=gt>=0
            counts=np.bincount(labels[valid].astype(np.int64)*8+gt[valid],minlength=(n+1)*8).reshape(n+1,8)
            for r in regions:
                assert int((labels==r['id']).sum())==r['area']
            hist=counts[1:]
            major=np.r_[0,hist.argmax(1)].astype(np.uint8)
            pred=major[labels]
            per_image.append({'sample_id':row['sample_id'],'category':category,'source':source,
                'candidates':n,'gt_pixels':int(positive.sum()),
                'foreground_gt_pixels':int((positive&foreground).sum()),
                'candidate_gt_pixels':int((positive&(labels>0)).sum()),
                'foreground_pixels':int(foreground.sum()),'candidate_pixels':int((labels>0).sum())})
            for r,c in zip(regions,hist):
                region_rows.append({'sample_id':row['sample_id'],'source':source,'region_id':r['id'],
                                    'gt_counts':c.tolist()})
            for cid in range(1,8):
                if (row['sample_id'],cid) not in allowed: continue
                target=gt==cid; gt_pixels=int(target.sum())
                best,chosen=best_union(hist,gt_pixels,cid)
                majority_scores[source].append(query_score(row['sample_id'],cid,gt,pred==cid))
                per_query.append({'sample_id':row['sample_id'],'category':category,'class_id':cid,
                    'source':source,'gt_pixels':gt_pixels,
                    'foreground_gt_pixels':int((target&foreground).sum()),
                    'candidate_gt_pixels':int(hist[:,cid].sum()),
                    'best_subset_iou':best,'best_subset_regions':chosen})
                if source=='generic':
                    for method in binary_counts:
                        out=root/f'methods/{method}/samples'/row['sample_id']/'queries'/CLASS_NAMES[cid]
                        decisions=read(out/'decisions.json')
                        assert len(decisions)==n
                        selected=np.zeros(n,bool)
                        for d in decisions: selected[d['region_id']-1]=d['selected']
                        wanted=major[1:]==cid
                        binary_counts[method]+=np.bincount(wanted.astype(int)*2+selected,minlength=4).reshape(2,2)
                        h=hist[selected].sum(0)
                        px=pixel_counts[method]
                        px['gt_pixels']+=gt_pixels
                        px['gt_outside_candidates']+=gt_pixels-int(hist[:,cid].sum())
                        px['gt_in_rejected_candidates']+=int(hist[~selected,cid].sum())
                        px['true_positive']+=int(h[cid])
                        px['background_false_positive']+=int(h[0])
                        px['other_function_false_positive']+=int(h.sum()-h[0]-h[cid])
        if index%200==0: print(f'GEOMETRY {index}/{len(rows)}',flush=True)
    result={'task':'retrospective GT-only diagnostic; not model inference or checkpoint selection',
            'images':len(rows),'queries_per_source':len(allowed),'sources':{},'binary_decisions':{}}
    for source in sources:
        images=[x for x in per_image if x['source']==source]
        qs=[x for x in per_query if x['source']==source]
        gt_sum=sum(x['gt_pixels'] for x in images)
        result['sources'][source]={
            'foreground_gt_recall':sum(x['foreground_gt_pixels'] for x in images)/gt_sum,
            'candidate_gt_recall':sum(x['candidate_gt_pixels'] for x in images)/gt_sum,
            'mean_candidates':float(np.mean([x['candidates'] for x in images])),
            'majority_assignment':summarize_queries(majority_scores[source]),
            'best_subset_gIoU_upper_bound':float(np.mean([q['best_subset_iou'] for q in qs])),
            'categories':{}}
        for category in sorted({x['category'] for x in images}):
            ii=[x for x in images if x['category']==category]
            qq=[x for x in qs if x['category']==category]
            g=sum(x['gt_pixels'] for x in ii)
            result['sources'][source]['categories'][category]={
                'images':len(ii),'foreground_gt_recall':sum(x['foreground_gt_pixels'] for x in ii)/g,
                'candidate_gt_recall':sum(x['candidate_gt_pixels'] for x in ii)/g,
                'best_subset_gIoU_upper_bound':float(np.mean([x['best_subset_iou'] for x in qq]))}
    for method,cm in binary_counts.items():
        tn,fp,fn,tp=cm.ravel().tolist()
        result['binary_decisions'][method]={'gt_majority_binary_confusion_rows_gt_cols_prediction':cm.tolist(),
            'precision_vs_majority':tp/max(tp+fp,1),'recall_vs_majority':tp/max(tp+fn,1),
            'pixel_error_decomposition':dict(pixel_counts[method])}
    write_jsonl(dest/'geometry_per_image.jsonl',per_image)
    write_jsonl(dest/'geometry_per_query.jsonl',per_query)
    write_jsonl(dest/'region_gt_counts.jsonl',region_rows)
    write(dest/'geometry.json',result)
    print({s:{k:v for k,v in r.items() if k not in ['categories','majority_assignment']}
           for s,r in result['sources'].items()},flush=True)


def score_multiclass(root, predictions, output):
    root, predictions, output=Path(root),Path(predictions),Path(output)
    rows=read_jsonl(root/'test_manifest.jsonl')
    expected={(q['sample_id'],q['class_id']) for q in read_jsonl(root/'test_queries.jsonl')}
    scores=[]
    for row in rows:
        gt,_=load_annotation(row['annotation'],tuple(row['size_wh'][::-1]))
        sample=predictions/'samples'/row['sample_id']
        assert read(sample/'run.json')['status']=='complete'
        pred=np.asarray(Image.open(sample/'result/class_map.png'))
        for cid in range(1,8):
            if (row['sample_id'],cid) in expected:
                scores.append(query_score(row['sample_id'],cid,gt,pred==cid))
    assert len(scores)==5232
    write_jsonl(output/'per_query.jsonl',scores)
    write(output/'metrics.json',summarize_queries(scores))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    geometry(parser.parse_args().root)
