"""RGB-only object decisions; GT is never consulted here."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from ...common import read, write, sha, load_mask
from ...visualization import overlay, sheet
from ..conditional_umd.requests import decode
from ..umd_valen.data import link


def eligible_candidates(records, masks, config):
    """Keep every eligible proposal, deduplicating only identical pixel sets."""
    if len(records) != len(masks):
        raise ValueError('Proposal metadata/masks disagree')
    groups, trace = {}, []
    for r, mask in zip(records, masks):
        if int(mask.sum()) != r['area']:
            raise ValueError('Incorrect proposal area')
        reason = ('small' if r['area'] < config['minimum_object_pixels'] else
                  'low_quality' if r['quality'] < config['minimum_object_score'] else None)
        item = {**r, 'id': r['raw_id'] + 1, 'level': 'object'}
        if reason is None:
            digest = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
            groups.setdefault(digest, []).append(item)
        else:
            trace.append({'raw_id': r['raw_id'], 'reason': reason})
    kept = []
    for duplicates in groups.values():
        # Same mask: preserve its highest quality record; stable tie by raw ID.
        duplicates.sort(key=lambda r: (-r['quality'], r['raw_id']))
        kept.append(duplicates[0])
        trace.extend({'raw_id': r['raw_id'], 'reason': 'identical_mask',
                      'retained_raw_id': duplicates[0]['raw_id']} for r in duplicates[1:])
    return sorted(kept, key=lambda r: r['raw_id']), trace


def choose_foreground(candidates, decisions, threshold=0.5):
    ids = {r['id'] for r in candidates}
    if len(decisions) != len(ids) or {d['region_id'] for d in decisions} != ids:
        raise ValueError('Every retained candidate needs exactly one decision')
    scores = {d['region_id']: d for d in decisions}
    accepted = [r for r in candidates if scores[r['id']]['probabilities']['yes'] > threshold]
    if not accepted:
        return None
    return min(accepted, key=lambda r: (-scores[r['id']]['probabilities']['yes'],
                                       -r['quality'], r['raw_id']))


def foreground_request(candidates, category, action):
    text = (
        f'The target object category is {category}. The intended operation is {action}. '
        'Identify the complete visible target OBJECT, before any functional-part selection. '
        'The operation only helps identify which object is intended; do not select just the handle, '
        'blade, working surface, or other functional subregion. '
        'A yes candidate should tightly cover the whole visible target object, including its visible '
        'body and attached parts. Naturally occluded pixels need not be invented. '
        'Reject isolated parts, incomplete fragments, and masks that substantially include another '
        'object or background. Reject a tabletop, display board, support platform, or other object '
        'merely underneath or near the target. '
        'The first image is the original unpainted RGB. The second outlines and numbers this group '
        'of object proposals. Each panel of the third image outlines ONE proposal on the same RGB. '
        'Colors and numbers are annotations, not appearance. Multiple overlapping proposals can '
        'describe the same target: judge each independently. No candidate is required to be yes.')
    content = [{'type': 'text', 'text': text}]
    content += [{'type': 'image_url', 'image_url': {'url': name}}
                for name in ['rgb.png', 'numbered.png', 'region_sheet.jpg']]
    questions = {f'region_{r["id"]:02}': {
        'type': 'choice',
        'instructions': f'Does outlined region {r["id"]} tightly represent the complete visible {category} intended for {action}? Select yes or no.',
        'criteria': {
            'yes': f'This mask tightly represents the whole visible target {category}, rather than only a functional part.',
            'no': 'This is a different object, background, a support surface, an incomplete part, or a mask substantially mixing target and surroundings.'}}
        for r in candidates}
    return {'state': {'messages': [{'role': 'user', 'content': content}]}, 'questions': questions}


def render_groups(source, candidates, group_size):
    source = Path(source)
    rgb = Image.open(source/'rgb.png').convert('RGB')
    groups = []
    for start in range(0, len(candidates), group_size):
        group = candidates[start:start+group_size]
        folder = source/'inputs'/f'group_{len(groups):02}'
        folder.mkdir(parents=True, exist_ok=True)
        masks = [load_mask(source/r['mask'], (rgb.height, rgb.width)) for r in group]
        link(source/'rgb.png', folder/'rgb.png')
        overlay(rgb, masks, labels=[str(r['id']) for r in group], fill_alpha=0).save(folder/'numbered.png')
        sheet([(f'Object {r["id"]}', overlay(rgb, [m], labels=[str(r['id'])], fill_alpha=0))
               for r, m in zip(group, masks)], folder/'region_sheet.jpg', columns=3)
        write(folder/'candidates.json', group)
        groups.append({'path': str(folder.relative_to(source)), 'candidates': group})
    write(source/'groups.json', groups)
    return groups


def decide(model, source, out, query, candidates, config):
    source, out = Path(source), Path(out)
    decisions, runtimes = [], []
    for group in read(source/'groups.json'):
        folder = out/Path(group['path']).name
        folder.mkdir(parents=True, exist_ok=True)
        for name in ['rgb.png', 'numbered.png', 'region_sheet.jpg']:
            target = folder/name
            if not target.exists():
                link(source/group['path']/name, target)
        request = foreground_request(group['candidates'], query['object_category'], query['action'])
        write(folder/'valen_request.json', request)
        write(folder/'prepared_assets.json', {name: sha(folder/name) for name in
              ['rgb.png', 'numbered.png', 'region_sheet.jpg', 'valen_request.json']})
        model.predict(folder)
        response = read(folder/'valen_response.json')
        if response.get('usage', {}).get('output_tokens', 0) != 0:
            raise ValueError('Expected Valen discriminative scoring, without generated text')
        decisions.extend(decode(response, group['candidates']))
        runtimes.append(read(folder/'valen_runtime.json'))
    chosen = choose_foreground(candidates, decisions, config['minimum_yes_probability'])
    return chosen, decisions, runtimes
