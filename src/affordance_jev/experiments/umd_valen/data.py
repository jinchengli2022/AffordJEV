"""Prepare immutable RGB-only requests; attach supervision separately for train."""
import argparse
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ...common import read, write, read_jsonl, write_jsonl, sha, load_mask
from ...datasets.umd import load_annotation
from ...decision.regions import request_for
from ...pipeline.run import sample_directory
from ...visualization import overlay, sheet

from ...paths import PROJECT, PARTGLEE_ROOT, PARTGLEE_WEIGHTS, VALEN_MODEL


def link(src, dst):
    src, dst = Path(src).resolve(strict=True), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(os.path.relpath(src, dst.parent.resolve()), target_is_directory=src.is_dir())


def empty_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_candidates(output, manifest=None, test_manifest=None, config_path=None, definitions_path=None):
    output = empty_output(output)
    manifest = Path(manifest) if manifest else PROJECT / "data/UMD/manifests/ooal_novel_train_76.jsonl"
    rows = read_jsonl(manifest)
    tests = read_jsonl(test_manifest or PROJECT / "data/UMD/manifests/umd_novel_test.jsonl")
    assert len(rows) == len({r['object_instance'] for r in rows}) == 76
    for field in ['sample_id', 'object_instance', 'rgb_sha256']:
        assert not {r[field] for r in rows} & {r[field] for r in tests}
    config = read(config_path or PROJECT / "configs/umd_pipeline.json")
    definitions = read(definitions_path or PROJECT / "configs/umd_affordances.json")
    jobs = []
    for row in rows:
        assert sha(row['rgb']) == row['rgb_sha256']
        assert sha(row['annotation']) == row['annotation_sha256']
        sample = sample_directory(output, row['sample_id'])
        sample.mkdir(parents=True)
        Image.open(row['rgb']).convert('RGB').save(sample / 'rgb.png')
        write(sample / 'config_snapshot.json', config)
        write(sample / 'affordances_snapshot.json', definitions)
        job = dict(sample_id=row['sample_id'], source_rgb=row['rgb'], output=str(sample),
                   source_sha256=row['rgb_sha256'], rgb_sha256=sha(sample/'rgb.png'),
                   source_size_wh=row['size_wh'], reference_annotations_used=False,
                   partglee_root=str(PARTGLEE_ROOT),
                   partglee_weights=str(PARTGLEE_WEIGHTS),
                   valen_model=str(VALEN_MODEL))
        write(sample/'job.json', job)
        write(sample/'run.json', {'status':'pending', 'reference_annotations_used':False})
        jobs.append(job)
    write_jsonl(output/'training_manifest.jsonl', rows)
    write_jsonl(output/'model_jobs.jsonl', jobs)
    write(output/'data_audit.json', {'images':76, 'train_test_overlap':False,
          'train_manifest_sha256':sha(manifest), 'gt_types':dict(Counter(r['gt_type'] for r in rows)),
          'annotation_paths_in_model_jobs':False})
    print(output/'model_jobs.jsonl', flush=True)


def definitions_for_sft(definitions_path=None):
    definitions = read(definitions_path or PROJECT/'configs/umd_affordances.json')
    old = 'Select none if no supplied role fits or the region is an incompatible mixture.'
    assert old in definitions['decision_rules']
    definitions['decision_rules'] = definitions['decision_rules'].replace(old,
        'Select none for background or when no supplied role fits. '
        'For a region containing multiple functional parts, select the role covering the largest visible portion of that region.')
    for cls in definitions['classes']:
        if cls['name'] == 'none':
            cls['definition'] = ('The marked region is predominantly background or has no matching supplied functional role. '
                'A mixture of functional parts should be assigned its dominant role rather than automatically rejected. '
                'This is a rejection option, not an eighth UMD affordance.')
    return definitions


def clone_sample(source, destination, row, config, definitions, outline=False):
    source, destination = Path(source), Path(destination)
    state = read(source/'run.json')['status']
    if state not in ['prepared', 'complete']:
        raise ValueError(f'Unusable source {source}: {state}')
    destination.mkdir(parents=True)
    for name in ['rgb.png', 'foreground', 'candidates.json', 'partglee_runtime.json', 'prepare_runtime.json']:
        link(source/name, destination/name)
    (destination/'regions').mkdir()
    for name in ['masks', 'label_map.png', 'unassigned_foreground.png']:
        link(source/'regions'/name, destination/'regions'/name)
    candidates = read(source/'candidates.json')['candidates']
    for r in candidates:
        if sha(source/r['mask']) != r['mask_sha256']:
            raise ValueError(f'Candidate mask changed: {source/r["mask"]}')
    if outline:
        rgb = Image.open(source/'rgb.png').convert('RGB')
        crop = read(source/'foreground/metadata.json')['crop_xyxy']
        masks = [load_mask(source/r['mask']) for r in candidates]
        # This verifies the default rendering reproduces the cached baseline exactly.
        reconstructed = np.asarray(overlay(rgb, masks).crop(crop))
        assert np.array_equal(reconstructed, np.asarray(Image.open(source/'regions/numbered_crop.png')))
        overlay(rgb, masks, fill_alpha=0).crop(crop).save(destination/'regions/numbered_crop.png')
        panels = [(f"Region {r['id']}", overlay(rgb,[m],labels=[str(r['id'])],fill_alpha=0).crop(crop))
                  for r,m in zip(candidates,masks)]
        sheet(panels,destination/'regions/region_sheet.jpg',columns=3)
    else:
        for name in ['numbered_crop.png','region_sheet.jpg']:
            link(source/'regions'/name,destination/'regions'/name)
    write(destination/'config_snapshot.json',config)
    write(destination/'affordances_snapshot.json',definitions)
    request=request_for(candidates,definitions,include_sheet=True)
    write(destination/'valen_request.json',request)
    if outline:
        assert request == read(source/'valen_request.json'), 'Outline experiment changed text'
    assets=['rgb.png','foreground/object_mask.png','foreground/rgb_crop.png',
            'regions/numbered_crop.png','regions/region_sheet.jpg','config_snapshot.json',
            'affordances_snapshot.json','candidates.json','valen_request.json']+[r['mask'] for r in candidates]
    write(destination/'prepared_assets.json',{p:sha(destination/p) for p in assets})
    write(destination/'run.json',{'status':'prepared','reference_annotations_used':False})
    job=read(source/'job.json')
    job['output']=str(destination)
    job.update(partglee_root=str(PARTGLEE_ROOT), partglee_weights=str(PARTGLEE_WEIGHTS), valen_model=str(VALEN_MODEL))
    write(destination/'job.json',job)
    write(destination/'variant_audit.json',{'source_sample':str(source),'outline_only':outline,
          'candidates_sha256':sha(source/'candidates.json'),'candidate_masks_verified':len(candidates),
          'first_rgb_unchanged':True,'ground_truth_used_in_request':False})
    return job


def variant(source, output, manifest, style, protocol, training=False, config_path=None, definitions_path=None):
    source=Path(source).resolve(strict=True)
    output=empty_output(output)
    rows=read_jsonl(manifest)
    definitions=definitions_for_sft(definitions_path) if protocol=='dominant' else read(definitions_path or PROJECT/'configs/umd_affordances.json')
    config=deepcopy(read(config_path or PROJECT/'configs/umd_pipeline.json'))
    config['valen']['save_compiled_tensors']=False
    write(output/'config_snapshot.json',config)
    write(output/'affordances_snapshot.json',definitions)
    manifest_name='training_manifest.jsonl' if training else 'evaluation_manifest.jsonl'
    write_jsonl(output/manifest_name,rows)
    def prepare(row):
        return clone_sample(sample_directory(source,row['sample_id']),sample_directory(output,row['sample_id']),
                            row,config,definitions,style=='outline')
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[]
        for i,job in enumerate(pool.map(prepare,rows),1):
            jobs.append(job)
            if i%100==0 or i==len(rows): print(f'PREPARE {i}/{len(rows)} {output.name}',flush=True)
    write_jsonl(output/'model_jobs.jsonl',jobs)
    write(output/'run.json',{'status':'prepared','samples':len(rows),'style':style,'protocol':protocol,
          'evaluation_manifest_sha256':sha(output/manifest_name),'source_candidates':str(source),
          'reference_annotations_used_in_inference':False,'whole_object_prohibition':False})
    if training:
        names={c['id']:c['name'] for c in definitions['classes']}
        records,counts=[],Counter()
        for row in rows:
            assert row['split']=='novel/train'
            assert sha(row['annotation'])==row['annotation_sha256']
            sample=sample_directory(output,row['sample_id'])
            gt,kind=load_annotation(row['annotation'],tuple(row['size_wh'][::-1]))
            targets,regions={},[]
            for r in read(sample/'candidates.json')['candidates']:
                mask=load_mask(sample/r['mask'])
                values=gt[mask & (gt>=0)]
                if not len(values): continue
                hist=np.bincount(values,minlength=8)
                category=int(hist.argmax())
                targets[f"region_{r['id']:02}"]={'probabilities':{
                    c['name']:float(c['id']==category) for c in definitions['classes']}}
                counts[names[category]]+=1
                regions.append({'region_id':r['id'],'gt_majority':names[category],
                                'purity':float(hist.max()/hist.sum()),'gt_pixel_counts':hist.tolist()})
            if not targets: raise ValueError(f'No training targets: {row["sample_id"]}')
            record={'sample_id':row['sample_id'],'group_id':row['object_instance'],'media_root':str(sample),
                    'request':read(sample/'valen_request.json'),'targets':targets,
                    'annotation_sha256':row['annotation_sha256'],'gt_type':kind,'regions':regions}
            records.append(record)
        write_jsonl(output/'records.jsonl',records)
        write(output/'supervision_audit.json',{'training_images':len(records),'region_classes':dict(counts),
              'target_rule':'majority valid GT pixels including background; ties use smallest native class ID',
              'ignore_gt_id':-1,'target_content_in_request':False,'extra_training_frames':0})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['train-candidates','variant'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--source')
    parser.add_argument('--manifest')
    parser.add_argument('--test-manifest', help='Train/test leakage audit for train-candidates')
    parser.add_argument('--config', help='PartGLEE/pipeline JSON; defaults to configs/umd_pipeline.json')
    parser.add_argument('--definitions', help='Affordance definitions JSON; defaults to configs/umd_affordances.json')
    parser.add_argument('--style',choices=['fill','outline'],default='fill')
    parser.add_argument('--protocol',choices=['original','dominant'],default='original')
    parser.add_argument('--training',action='store_true')
    args=parser.parse_args()
    if args.mode=='train-candidates':
        train_candidates(args.output,args.manifest,args.test_manifest,args.config,args.definitions)
    else:
        variant(args.source,args.output,args.manifest,args.style,args.protocol,args.training,args.config,args.definitions)


if __name__=='__main__': main()
