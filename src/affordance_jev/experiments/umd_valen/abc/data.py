"""Training-only transformations. Geometry never receives the ground truth."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
import re

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from ....common import read, write, read_jsonl, write_jsonl, load_mask, save_mask, sha
from ....datasets.umd import load_annotation
from ....decision.regions import request_for
from ....visualization import overlay, sheet
from ..data import PROJECT, definitions_for_sft, empty_output, link

NAMES = ['none', 'grasp', 'cut', 'scoop', 'contain', 'pound', 'support', 'wrap-grasp']
PARAPHRASES = {
    'none': 'Mostly background, or a surface without a matching role below. For mixed functional parts, choose the role occupying most of the marked area instead of rejecting the mixture.',
    'grasp': 'A purpose-built handle or finger grip for holding and controlling the object. A broad exterior body and a merely touchable working end are not this role.',
    'cut': 'A sharpened edge or blade working area that divides material. The handle and non-cutting structures are excluded.',
    'scoop': 'A hollowed working surface that gathers and lifts loose or soft material during a scooping movement. This differs from a deep cavity for storage and a flat surface for supporting a load.',
    'contain': 'The visible inner cavity or interior wall for keeping contents inside. A separate handle and the outside body used for a wrapping grip are excluded.',
    'pound': 'A solid striking head or impact surface for applying force to material. The shaft and handle do not have this role.',
    'support': 'A comparatively flat working face that bears or carries material from below. It differs from a hollow scoop, a retaining cavity, and a handle.',
    'wrap-grasp': 'The outside body surface held with the palm and fingers wrapped around it. A dedicated handle and an inner retaining cavity are different regions.',
}


def transformed_geometry(masks, variant):
    """Disjoint median split and nearest-boundary merge; no labels are accepted."""
    split_index = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))[variant % len(masks)]
    mask = masks[split_index]
    ys, xs = np.nonzero(mask)
    axis = (int(np.ptp(xs) >= np.ptp(ys)) + variant // 2) % 2
    coordinates = xs if axis else ys
    order = np.lexsort((ys if axis else xs, coordinates))
    first = np.zeros_like(mask)
    selected = order[:len(order)//2]
    first[ys[selected], xs[selected]] = True
    second = mask & ~first
    assert first.any() and second.any() and not (first & second).any()
    remaining = [i for i in range(len(masks)) if i != split_index]
    pair = None
    if len(remaining) >= 2:
        distances = []
        for j, i in enumerate(remaining):
            distance = distance_transform_edt(~masks[i])
            for k in remaining[j+1:]:
                distances.append((float(distance[masks[k]].min()), i, k))
        _, i, k = min(distances)
        pair = (i, k)
    output = [first, second]
    relations = [{'kind': 'split', 'original': [split_index], 'transformed': [0, 1]}]
    if pair:
        output.append(masks[pair[0]] | masks[pair[1]])
        relations.append({'kind': 'merge', 'original': list(pair), 'transformed': [2]})
    for i in remaining:
        if pair and i in pair:
            continue
        output.append(masks[i])
    coverage = np.stack(output).sum(0)
    assert coverage.max() == 1
    assert np.array_equal(coverage > 0, np.logical_or.reduce(masks))
    for relation in relations:
        before = np.logical_or.reduce([masks[i] for i in relation['original']])
        after = np.logical_or.reduce([output[i] for i in relation['transformed']])
        assert np.array_equal(before, after)
    return output, relations


def render(source, destination, masks, colors=None, panel_order=None):
    destination.mkdir(parents=True)
    link(source/'rgb.png', destination/'rgb.png')
    link(source/'foreground', destination/'foreground')
    (destination/'regions/masks').mkdir(parents=True)
    rgb = Image.open(source/'rgb.png').convert('RGB')
    crop = read(source/'foreground/metadata.json')['crop_xyxy']
    overlay(rgb, masks, colors=colors).crop(crop).save(destination/'regions/numbered_crop.png')
    panels = [(f'Region {i+1}', overlay(rgb, [mask], colors=None if colors is None else [colors[i]],
               labels=[str(i+1)]).crop(crop)) for i, mask in enumerate(masks)]
    if panel_order is not None:
        panels = [panels[i] for i in panel_order]
    sheet(panels, destination/'regions/region_sheet.jpg', columns=3)
    candidates = []
    for i, mask in enumerate(masks, 1):
        path = f'regions/masks/{i:02}.png'
        save_mask(destination/path, mask)
        candidates.append({'id': i, 'mask': path, 'mask_sha256': sha(destination/path)})
    write(destination/'candidates.json', {'candidates': candidates})
    return candidates


def supervise(record, masks, gt, key_to_name):
    targets, details = {}, []
    for i, mask in enumerate(masks, 1):
        hist = np.bincount(gt[mask & (gt >= 0)], minlength=8)
        if not hist.sum():
            raise ValueError('Transformed region has no valid supervised pixels')
        majority = NAMES[int(hist.argmax())]
        qid = f'region_{i:02}'
        targets[qid] = {'probabilities': {key: float(name == majority) for key, name in key_to_name.items()}}
        details.append({'region_id': i, 'gt_majority': majority, 'purity': float(hist.max()/hist.sum()),
                        'gt_pixel_counts': hist.tolist()})
    record.update(targets=targets, regions=details, key_to_name=key_to_name)
    return record


def prepare(output, records_path=None, manifest_path=None, definitions_path=None):
    output = empty_output(output)
    records_path = Path(records_path) if records_path else PROJECT/'outputs/valen_76/train_fill_dominant/records.jsonl'
    manifest_path = Path(manifest_path) if manifest_path else PROJECT/'data/UMD/manifests/ooal_novel_train_76.jsonl'
    records = read_jsonl(records_path)
    manifest = read_jsonl(manifest_path)
    assert len(records) == len({r['group_id'] for r in records}) == 76
    by_id = {r['sample_id']: r for r in manifest}
    original_definitions = definitions_for_sft(definitions_path)
    rows_a, rows_b = [], []
    counts = Counter()
    for sample_index, record in enumerate(records):
        source = Path(record['media_root'])
        native = by_id[record['sample_id']]
        assert native['split'] == 'novel/train'
        assert sha(native['annotation']) == native['annotation_sha256']
        gt, _ = load_annotation(native['annotation'], tuple(native['size_wh'][::-1]))
        masks = [load_mask(source/c['mask']) for c in read(source/'candidates.json')['candidates']]
        for v in range(4):
            # A: physical masks unchanged; all display/wording choices are label-blind.
            rng = random.Random(321 + sample_index*100 + v)
            permutation = list(range(len(masks)))
            rng.shuffle(permutation)
            new_masks = [masks[i] for i in permutation]
            colors = [tuple(rng.randint(50, 245) for _ in range(3)) for _ in masks]
            order = list(range(len(masks)))
            rng.shuffle(order)
            dest = output/'a'/record['sample_id']/str(v)
            candidates = render(source, dest, new_masks, colors, order)
            definitions = deepcopy(original_definitions)
            if v % 2:
                for c in definitions['classes']:
                    c['definition'] = PARAPHRASES[c['name']]
            request = request_for(candidates, definitions)
            key_to_name = {n:n for n in NAMES}
            if v >= 2:
                name_to_key = {n:f'role_{i+1}' for i,n in enumerate(rng.sample(NAMES, len(NAMES)))}
                key_to_name = {key:name for name,key in name_to_key.items()}
                # Hide category keys and any category words in the generic rules too.
                def hide(text):
                    for n in sorted(NAMES, key=len, reverse=True):
                        text = re.sub(r'(?<![\w-])'+re.escape(n)+r'(?![\w-])', name_to_key[n], text)
                    return text
                for message in request['state']['messages']:
                    for item in message['content']:
                        if item['type'] == 'text': item['text'] = hide(item['text'])
                for question in request['questions'].values():
                    question['instructions'] = hide(question['instructions'])
                    question['criteria'] = {name_to_key[n]:hide(d) for n,d in question['criteria'].items()}
            questions = list(request['questions'].items())
            rng.shuffle(questions)
            request['questions'] = dict(questions)
            for question in request['questions'].values():
                pairs = list(question['criteria'].items())
                rng.shuffle(pairs)
                question['criteria'] = dict(pairs)
            row = dict(sample_id=record['sample_id'], group_id=record['group_id'], variant=v,
                       media_root=str(dest), request=request, original_region_indices=permutation)
            supervise(row, new_masks, gt, key_to_name)
            write(dest/'valen_request.json', request)
            write(dest/'training_metadata.json', {k:x for k,x in row.items() if k!='request'})
            rows_a.append(row)
            # B: geometry constructed before inspecting labels.
            new_masks, relations = transformed_geometry(masks, v)
            dest = output/'b'/record['sample_id']/str(v)
            candidates = render(source, dest, new_masks)
            request = request_for(candidates, original_definitions)
            row = dict(sample_id=record['sample_id'], group_id=record['group_id'], variant=v,
                       media_root=str(dest), request=request, relations=relations)
            supervise(row, new_masks, gt, {n:n for n in NAMES})
            original_counts = np.array([r['gt_pixel_counts'] for r in record['regions']])
            transformed_counts = np.array([r['gt_pixel_counts'] for r in row['regions']])
            for relation in relations:
                assert np.array_equal(original_counts[relation['original']].sum(0),
                                      transformed_counts[relation['transformed']].sum(0))
            write(dest/'valen_request.json', request)
            write(dest/'training_metadata.json', {k:x for k,x in row.items() if k!='request'})
            rows_b.append(row)
            counts.update(r['kind'] for r in relations)
        print(f'A/B TRAIN VIEWS {sample_index+1}/76', flush=True)
    write_jsonl(output/'a_records.jsonl', rows_a)
    write_jsonl(output/'b_records.jsonl', rows_b)
    write(output/'audit.json', {'source_records_sha256':sha(records_path),
          'images':76, 'variants_per_image':4, 'a_views':len(rows_a), 'b_views':len(rows_b),
          'b_relations':dict(counts), 'geometry_uses_gt':False,
          'a_mask_permutation_only':True, 'b_union_and_disjointness_verified':True,
          'b_valid_pixel_composition_additivity_verified':True,
          'labels_only_in_targets_and_training_metadata':True, 'extra_source_images':0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--records', help='Original 76-image records.jsonl')
    parser.add_argument('--manifest', help='Original 76-image training manifest')
    parser.add_argument('--definitions', help='Original functional definitions before dominant-protocol adjustment')
    args = parser.parse_args()
    prepare(args.output,args.records,args.manifest,args.definitions)
