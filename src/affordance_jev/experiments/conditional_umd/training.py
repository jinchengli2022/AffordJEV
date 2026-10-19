"""Build per-image groups of conditional binary supervision on the same 76 RGBs."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ...common import read, write, read_jsonl, write_jsonl, load_mask, sha
from ...datasets.umd import load_annotation
from ...paths import PROJECT, VALEN_MODEL
from ...pipeline.run import sample_directory
from .requests import prepare_assets, prepare_query


def prepare(root):
    root=Path(root).resolve()
    rows=read_jsonl(root/'train_manifest.jsonl')
    groups=defaultdict(list)
    for q in read_jsonl(root/'train_queries.jsonl'):
        groups[q['sample_id']].append(q)
    records=[]
    counts=Counter()
    audit=[]
    definitions=read(root/'definitions.json')
    for row in rows:
        source=sample_directory(root/'candidates_train',row['sample_id'])
        if read(source/'run.json')['status']!='prepared':
            raise ValueError(f'Training candidate preparation failed: {source}')
        sample=sample_directory(root/'training/inputs',row['sample_id'])
        candidates=prepare_assets(source,sample/'assets','sft')
        assert sha(row['annotation'])==row['annotation_sha256']
        gt,kind=load_annotation(row['annotation'],tuple(row['size_wh'][::-1]))
        labels={}
        for c in candidates:
            mask=load_mask(source/c['mask'],gt.shape)
            valid=gt[mask & (gt>=0)]
            if len(valid):
                hist=np.bincount(valid,minlength=8)
                labels[c['id']]=int(hist.argmax())
                audit.append({'sample_id':row['sample_id'],'region_id':c['id'],
                              'majority_id':labels[c['id']],'counts':hist.tolist()})
        if not labels:
            raise ValueError(f'No valid training candidate labels: {row["sample_id"]}')
        operations=[]
        for q in groups[row['sample_id']]:
            out=sample/'queries'/q['action']
            prepare_query(sample/'assets',out,q,candidates,definitions,'sft')
            targets={}
            for cid,majority in labels.items():
                yes=majority==q['class_id']
                targets[f'region_{cid:02}']={'probabilities':{'yes':float(yes),'no':float(not yes)}}
                counts['yes' if yes else 'no']+=1
            operations.append({'media_root':str(out),'request':read(out/'valen_request.json'),
                               'targets':targets,'action':q['action']})
        records.append({'sample_id':row['sample_id'],'group_id':row['object_instance'],
                        'operations':operations,'annotation_sha256':row['annotation_sha256'],'gt_type':kind})
    assert len(records)==76 and len({r['group_id'] for r in records})==76
    dest=root/'training/records.jsonl'
    write_jsonl(dest,records)
    write_jsonl(root/'training/majority_audit.jsonl',audit)
    write(root/'training/supervision_audit.json',{'images':len(records),
        'operations':sum(len(r['operations']) for r in records),'binary_targets':dict(counts),
        'records_sha256':sha(dest),'target_rule':'majority valid GT class, background included; lowest native ID breaks ties; yes iff majority equals requested action',
        'image_weighting':'mean over candidates, then operations, then four source images',
        'gt_used_only_in_targets':True,'candidate_gt_types':dict(Counter(r['gt_type'] for r in records))})
    config=read(PROJECT/'configs/valen_76_sft.json')
    config.update(model_path=str(VALEN_MODEL),records=str(dest),output=str(root/'training'),
        checkpoint_dir=str(PROJECT/'checkpoints'/root.name),task='conditional_binary',
        prompt_protocol='known category + target operation + dominant definition; yes/no',
        training_target='yes iff candidate GT-majority ID equals requested action; background counts; ignore -1',
        loss='mean binary CE over candidates, then operations per source image, then four images',
        base_initialization='original Valen-4B, not the previous SFT checkpoint')
    write(root/'training/config.json',config)
    print({'training_config':str(root/'training/config.json'),'targets':dict(counts)},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    prepare(parser.parse_args().root)
