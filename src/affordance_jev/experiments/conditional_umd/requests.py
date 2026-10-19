"""Conditional yes/no requests and inspectable RGB assets (no GT access)."""
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...common import read, write, sha, load_mask
from ...visualization import overlay, sheet
from ..umd_valen.data import link
from ..umd_valen.centroid import centroid, point_image


METHODS = {'fill': 'Valen／原始填色', 'outline': 'Valen／仅描边',
           'centroid': 'Valen／质心点，逐候选', 'definition': 'Valen／定义调整',
           'sft': 'Valen／条件判断 SFT（本轮重训）'}


def request_for(candidates, definitions, category, action, *, dominant=False, point=False):
    target = next(c for c in definitions['classes'] if c['name'] == action and c['id'])
    rules = (
        'Use the supplied UMD functional definition to judge the visible pixels. '
        'Distinguish a dedicated handle from a body held by wrapping the hand around it, '
        'and distinguish a concave collecting surface, a storage cavity, and a flat supporting surface. '
        'Do not accept a region merely because it can be touched or held. '
        'A matching functional region may cover most or all of the object; there is no whole-object size prohibition. '
        'Background or parts of a different object do not match. ')
    rules += (
        'For a region containing multiple functional parts, judge the role covering the largest visible portion '
        'of the region; a mixture is not automatically rejected. '
        if dominant else 'Reject a region that is an incompatible mixture of functional parts. ')
    content = [{'type': 'text', 'text': (
        f'The target object category is {category}. The requested operation is {action}. '
        f'Target functional definition: {target["definition"]} '
        'For each indicated candidate, decide whether its primary functional role matches this target operation. '
        'The first image is an unpainted RGB crop; the second shows numbered candidates on the same crop. '
        'Colors and numbers are annotations, not physical appearance. Inspect the unpainted RGB. ' + rules)},
        {'type':'image_url', 'image_url':{'url':'rgb_crop.png'}},
        {'type':'image_url', 'image_url':{'url':'numbered_crop.png'}},
        {'type':'text', 'text': (
            'The third image is the unpainted RGB crop plus one red point at the arithmetic centroid of '
            f'candidate {candidates[0]["id"]}. Judge the visible functional part at that point; '
            'ignore the marker color. The point has not been projected into the mask.'
            if point else 'Each panel in the third image highlights one candidate on the same RGB crop.')},
        {'type':'image_url', 'image_url':{'url':'centroid.png' if point else 'region_sheet.jpg'}}]
    questions = {}
    for r in candidates:
        indicated = 'the visible functional part at the red point' if point else f'the pixels of region {r["id"]}'
        questions[f'region_{r["id"]:02}'] = {'type':'choice',
            'instructions': f'Does {indicated} belong to the {category} and match the requested operation {action}? Select yes or no.',
            'criteria': {
                'yes': f'This region of the {category} matches {action} according to the target definition and rules.',
                'no': f'This region does not match the requested {action} role, or is background or a different object. '
                      'It can have another valid function; no does not mean it has no function.'}}
    return {'state': {'messages': [{'role':'user', 'content':content}]}, 'questions': questions}


def decode(response, candidates):
    answers = response.get('answers', {})
    if set(answers) != {f'region_{r["id"]:02}' for r in candidates}:
        raise ValueError('Missing or unexpected candidate decisions')
    results = []
    for r in candidates:
        answer = answers[f'region_{r["id"]:02}']
        p = answer.get('probabilities', {})
        if answer.get('type') != 'choice' or set(p) != {'yes', 'no'}:
            raise ValueError('Expected binary yes/no probabilities')
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
            raise ValueError('Invalid binary probability')
        if not math.isclose(sum(p.values()), 1., abs_tol=1e-5):
            raise ValueError('Probabilities do not sum to one')
        results.append({'region_id': r['id'], 'probabilities': p,
                        'selected': p['yes'] > p['no'], 'tie_rule': 'no'})
    return results


def prepare_assets(source, destination, method):
    """Render once per RGB/method, shared by its operation requests."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True)
    write(destination/'source.json', {'source':str(source), 'method':method,
                                    'candidate_file_sha256':sha(source/'candidates.json')})
    candidates = read(source/'candidates.json')['candidates']
    rgb = Image.open(source/'rgb.png').convert('RGB')
    crop = read(source/'foreground/metadata.json')['crop_xyxy']
    masks = [load_mask(source/r['mask'], (rgb.height,rgb.width)) for r in candidates]
    fg = load_mask(source/'foreground/object_mask.png', (rgb.height,rgb.width))
    coverage = np.zeros(fg.shape, np.uint8)
    for r, mask in zip(candidates,masks):
        assert sha(source/r['mask']) == r['mask_sha256']
        assert not (mask & ~fg).any()
        coverage += mask
    assert coverage.max() <= 1
    link(source/'foreground/rgb_crop.png', destination/'rgb_crop.png')
    if method == 'outline':
        overlay(rgb,masks,fill_alpha=0).crop(crop).save(destination/'numbered_crop.png')
        if candidates:
            sheet([(f'Region {r["id"]}',overlay(rgb,[mask],labels=[str(r['id'])],fill_alpha=0).crop(crop))
                   for r,mask in zip(candidates,masks)], destination/'region_sheet.jpg',columns=3)
        else:
            rgb.crop(crop).save(destination/'region_sheet.jpg')
    else:
        link(source/'regions/numbered_crop.png', destination/'numbered_crop.png')
        link(source/'regions/region_sheet.jpg', destination/'region_sheet.jpg')
    if method == 'centroid':
        points=[]
        crop_rgb=rgb.crop(crop)
        for r,mask in zip(candidates,masks):
            geometry=centroid(mask,crop)
            name=f'centroid_{r["id"]:02}.png'
            point_image(crop_rgb,geometry).save(destination/name)
            points.append({'region_id':r['id'],'image':name,**geometry})
        write(destination/'centroids.json',points)
    return candidates


def prepare_query(assets, out, query, candidates, definitions, method):
    assets, out = Path(assets), Path(out)
    out.mkdir(parents=True)
    write(out/'query.json', query)
    requests = []
    groups = [[r] for r in candidates] if method=='centroid' else [candidates]
    for group in groups:
        folder = out/f'region_{group[0]["id"]:02}' if method=='centroid' else out
        folder.mkdir(parents=True,exist_ok=True)
        files=['rgb_crop.png','numbered_crop.png']
        for name in files:
            link(assets/name,folder/name)
        if method=='centroid':
            link(assets/f'centroid_{group[0]["id"]:02}.png',folder/'centroid.png')
            files.append('centroid.png')
        else:
            link(assets/'region_sheet.jpg',folder/'region_sheet.jpg')
            files.append('region_sheet.jpg')
        request=request_for(group,definitions,query['object_category'],query['action'],
                            dominant=method in ['definition','sft'],point=method=='centroid')
        write(folder/'valen_request.json',request)
        write(folder/'prepared_assets.json',{name:sha(folder/name) for name in files+['valen_request.json']})
        requests.append({'folder':str(folder),'candidates':group})
    write(out/'requests.json',requests)
    write(out/'run.json',{'status':'prepared','reference_annotations_used':False})
    return requests
