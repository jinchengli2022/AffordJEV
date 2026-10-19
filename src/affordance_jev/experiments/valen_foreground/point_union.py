"""Object foreground decisions from one interior point per disjoint region."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...common import read, write, sha, load_mask, save_mask
from ...segmentation.exclusive import partition_connected
from ...visualization import overlay
from ..umd_valen.data import link
from ..conditional_umd.requests import decode


METHOD = 'interior_point_union'


def enabled(config):
    return config['foreground_selector']['method'] == METHOD


def link_input(source, target):
    if target.exists() or target.is_symlink():
        if target.resolve(strict=True) != source.resolve(strict=True):
            raise ValueError(f'Input symlink refers to a different source: {target}')
    else:
        link(source, target)


def prepare(source, rgb, candidates, config):
    source = Path(source)
    shape = (rgb.height, rgb.width)
    masks = np.stack([load_mask(source/c['mask'], shape) for c in candidates]) if candidates else np.zeros((0, *shape), bool)
    labels, owner, regions, trace = partition_connected(masks, [c['quality'] for c in candidates],
        [c['raw_id'] for c in candidates], config['foreground_selector']['minimum_fragment_pixels'])
    folder = source/'preprocessed'
    folder.mkdir(exist_ok=True)
    np.save(folder/'label_map.npy', labels)
    np.save(folder/'owner_before_components.npy', owner)
    save_mask(folder/'uncovered.png', labels == 0)
    for r in regions:
        relative = f'preprocessed/masks/{r["id"]:03}.png'
        save_mask(source/relative, labels == r['id'])
        r.update(mask=relative, mask_sha256=sha(source/relative), level='object_partition')
        inp = folder/'inputs'/f'region_{r["id"]:03}'
        inp.mkdir(parents=True, exist_ok=True)
        link_input(source/'rgb.png', inp/'rgb.png')
        im = rgb.copy()
        x, y = r['point']['xy']
        ImageDraw.Draw(im).ellipse((x-3, y-3, x+3, y+3), fill=(255, 0, 0), outline=(255, 255, 255), width=1)
        im.save(inp/'point.png')
        write(inp/'point.json', r['point'])
        r.update(input_directory=str(inp.relative_to(source)), point_image_sha256=sha(inp/'point.png'))
    write(folder/'candidates.json', {'candidates': regions})
    write(folder/'trace.json', trace)
    overlay(rgb, [labels == r['id'] for r in regions], labels=[str(r['id']) for r in regions],
            fill_alpha=.50).save(folder/'overview.jpg', quality=95)
    raw = read(source/'raw/objects.json')['candidates']
    raw_masks = [load_mask(source/r['mask'], shape) for r in raw]
    # This is a display of overlapping raw masks, not a claim of exclusivity.
    overlay(rgb, raw_masks, labels=['']*len(raw), fill_alpha=.22).save(source/'raw/overview.jpg', quality=95)
    write(source/'raw/overview.json', {'render': 'raw_id ascending alpha compositing', 'alpha': .22,
        'overlap_allowed': True, 'individual_masks': 'objects.json'})
    return regions


def request(region_id, category, action):
    text = (
        f'The target object category is {category}. The intended operation is {action}. '
        'The first image is the original unpainted RGB. The second is the same full RGB with exactly one red point. '
        'Decide only whether the visible surface at the CENTER of the point belongs to the target object. '
        'Any visible part of the target object counts as foreground, regardless of whether that part performs the intended operation. '
        'The operation only identifies the intended target; do not restrict foreground to its functional working part. '
        'A tabletop, display board, other objects, and background visible through holes are background. '
        'Ignore the marker color and inspect the corresponding location in the original RGB. '
        'The point is a location marker; do not infer a region boundary or require a complete-object silhouette.')
    return {'state': {'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': text},
        {'type': 'image_url', 'image_url': {'url': 'rgb.png'}},
        {'type': 'image_url', 'image_url': {'url': 'point.png'}}]}]},
        'questions': {f'region_{region_id:02}': {'type': 'choice',
            'instructions': f'Does the visible surface at the center of the red point belong to the target {category}? Select yes or no.',
            'criteria': {'yes': f'The indicated surface belongs to the target {category}, including any of its visible parts.',
                         'no': 'The indicated surface is background, a supporting surface, another object, or background visible through a hole.'}}}}


CATEGORIES = {'cup': '杯子', 'ladle': '长柄汤勺', 'pot': '锅', 'saw': '锯子', 'scoop': '铲勺',
              'shears': '剪刀', 'shovel': '铲子', 'tenderizer': '松肉锤', 'trowel': '泥铲'}
ACTIONS = {'grasp': '抓握', 'cut': '切割', 'scoop': '舀取', 'contain': '盛装', 'pound': '敲击',
           'support': '承托', 'wrap-grasp': '环握'}


def chinese_prompt(category, action):
    obj = f'{CATEGORIES.get(category, category)}（{category}）'
    act = f'{ACTIONS.get(action, action)}（{action}）'
    return (f'目标物体类别是{obj}，计划执行的操作是{act}。\n\n'
        '第一张是未绘制标记的原始 RGB，第二张是同一张完整 RGB，上面只有一个红点。'
        '只判断红点中心对应的可见表面是否属于目标物体。目标物体的任意可见部位均属于前景，'
        '无论该部位是否执行指定操作。操作仅帮助确定目标，不要把前景限制为执行操作的功能部位。'
        '桌面、展示板、其它物体，以及孔洞中露出的背景均属于背景。'
        '忽略标记颜色，对照原始 RGB 检查该位置。红点只是位置标记；不要据此推测区域边界，也不要求完整物体轮廓。\n\n'
        f'判断问题：红点中心对应的可见表面是否属于目标{obj}？选择 yes 或 no。\n\n'
        f'yes／前景：该位置的表面属于目标{obj}，包括其任意可见部位。\n'
        'no／背景：该位置是背景、支撑表面、其它物体，或孔洞中露出的背景。\n')


def choose(candidates, decisions, threshold=.5):
    ids = {r['id'] for r in candidates}
    if len(decisions) != len(ids) or {d['region_id'] for d in decisions} != ids:
        raise ValueError('Every partition region needs exactly one decision')
    decision_map = {d['region_id']: d for d in decisions}
    return [r for r in candidates if decision_map[r['id']]['probabilities']['yes'] > threshold]


def decide(model, source, out, query, candidates, config):
    source, out = Path(source), Path(out)
    decisions, runtimes = [], []
    for r in candidates:
        folder = out/f'region_{r["id"]:03}'
        folder.mkdir(parents=True, exist_ok=True)
        for name in ['rgb.png', 'point.png', 'point.json']:
            link_input(source/r['input_directory']/name, folder/name)
        write(folder/'valen_request.json', request(r['id'], query['object_category'], query['action']))
        (folder/'prompt_zh.md').write_text(chinese_prompt(query['object_category'], query['action']), encoding='utf-8')
        write(folder/'prepared_assets.json', {name: sha(folder/name) for name in
              ['rgb.png', 'point.png', 'point.json', 'valen_request.json', 'prompt_zh.md']})
        model.predict(folder)
        response = read(folder/'valen_response.json')
        if response.get('usage', {}).get('output_tokens', 0) != 0:
            raise ValueError('Expected discriminative scoring without text generation')
        decoded = decode(response, [r])
        for d in decoded:
            d['selected'] = d['probabilities']['yes'] > config['minimum_yes_probability']
            d['threshold'] = config['minimum_yes_probability']
        decisions.extend(decoded)
        runtimes.append(read(folder/'valen_runtime.json'))
    return choose(candidates, decisions, config['minimum_yes_probability']), decisions, runtimes
