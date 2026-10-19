"""Freeze query identities before inference; keep GT out of model jobs."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ...common import read, write, read_jsonl, write_jsonl, sha
from ...datasets.umd import CLASS_NAMES, load_annotation
from ...paths import PROJECT, PARTGLEE_ROOT, PARTGLEE_WEIGHTS, VALEN_MODEL
from ...pipeline.run import sample_directory


def initialize(root):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    train_path = PROJECT/'data/UMD/manifests/ooal_novel_train_76.jsonl'
    test_path = PROJECT/'data/UMD/manifests/umd_novel_test.jsonl'
    train, test = read_jsonl(train_path), read_jsonl(test_path)
    assert len(train) == 76 and len(test) == 2661
    for field in ['sample_id', 'object_instance', 'rgb_sha256']:
        assert not {r[field] for r in train} & {r[field] for r in test}
    root.mkdir(parents=True)
    config = read(PROJECT/'configs/umd_pipeline_generic.json')
    config['valen'].update(save_compiled_inputs=True, save_compiled_tensors=False)
    assert config['vocabulary'] == {'objects': ['object'], 'parts': ['part']}
    assert config['partition']['whole_object_fraction'] is None
    definitions = read(PROJECT/'configs/umd_affordances.json')
    write(root/'pipeline.json', config)
    write(root/'definitions.json', definitions)
    counts = {}
    for split, rows in [('train', train), ('test', test)]:
        write_jsonl(root/f'{split}_manifest.jsonl', rows)
        queries, jobs = [], []
        for row in rows:
            assert sha(row['rgb']) == row['rgb_sha256']
            assert sha(row['annotation']) == row['annotation_sha256']
            gt, kind = load_annotation(row['annotation'], tuple(row['size_wh'][::-1]))
            if split == 'test':
                assert kind == 'manual'
            category = row['object_instance'].rsplit('_', 1)[0]
            # This is the task scheduler. GT never enters segmentation or decision jobs.
            for cid in sorted(int(c) for c in np.unique(gt) if c > 0):
                queries.append(dict(sample_id=row['sample_id'], query_id=f"{row['sample_id']}/{CLASS_NAMES[cid]}",
                    rgb=row['rgb'], rgb_sha256=row['rgb_sha256'], size_wh=row['size_wh'],
                    object_category=category, class_id=cid, action=CLASS_NAMES[cid]))
            out = sample_directory(root/f'candidates_{split}', row['sample_id'])
            out.mkdir(parents=True)
            Image.open(row['rgb']).convert('RGB').save(out/'rgb.png')
            write(out/'config_snapshot.json', config)
            write(out/'affordances_snapshot.json', definitions)
            job = dict(sample_id=row['sample_id'], source_rgb=row['rgb'], output=str(out),
                source_sha256=row['rgb_sha256'], rgb_sha256=sha(out/'rgb.png'),
                source_size_wh=row['size_wh'], reference_annotations_used=False,
                partglee_root=str(PARTGLEE_ROOT), partglee_weights=str(PARTGLEE_WEIGHTS),
                valen_model=str(VALEN_MODEL))
            assert not {'annotation', 'object_category', 'action', 'class_id'} & job.keys()
            write(out/'job.json', job)
            write(out/'run.json', {'status': 'pending', 'reference_annotations_used': False})
            jobs.append(job)
        assert len({q['query_id'] for q in queries}) == len(queries)
        if split == 'test':
            assert len(queries) == 5232
        write_jsonl(root/f'{split}_queries.jsonl', queries)
        write_jsonl(root/f'candidates_{split}/model_jobs.jsonl', jobs)
        write(root/f'candidates_{split}/config_snapshot.json', config)
        counts[split] = dict(images=len(rows), queries=len(queries),
            classes=dict(Counter(q['action'] for q in queries)),
            manifest_sha256=sha(root/f'{split}_manifest.jsonl'),
            queries_sha256=sha(root/f'{split}_queries.jsonl'))
    write(root/'protocol.json', dict(task='RGB + known object category + requested operation -> independent binary mask',
        counts=counts, train_test_overlap=False, query_list_fixed_before_inference=True,
        ground_truth_masks_in_model_jobs=False, fresh_predictions=True, shared_fresh_partglee_candidates=True,
        partglee_text={'objects':['object'], 'parts':['part']}, whole_object_prohibition=False,
        ooal='existing final checkpoint; fresh eight-way argmax, then requested class',
        r1='official prompt and original checkpoint; only scheduled actions; independent SAM2 mask',
        valen='binary yes/no per candidate per requested operation; union yes; ties no',
        sft='retrained from original Valen, same 76 images; 300 head + 1000 LoRA steps; final checkpoint',
        metrics=dict(mIoU='mean over seven classes of summed query intersections / summed query unions',
            gIoU='mean per-query IoU', P_at_50='fraction of queries with IoU >= 0.5',
            Dice='mean per-query Dice', errors='empty prediction; never omit failed queries',
            void='exclude -1 pixels', background='false positives inside requested queries count')))
    write(root/'status.json', {'status':'initialized', 'counts':counts})
    print(counts, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    initialize(parser.parse_args().root)


if __name__ == '__main__':
    main()
