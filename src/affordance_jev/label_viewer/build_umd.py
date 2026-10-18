"""Build a portable UMD original/OOAL-rule GT viewer; never rewrite source GT."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..common import read, read_jsonl, sha, write
from ..datasets.umd import CLASS_NAMES, load_annotation, read_split
from ..paths import project_path


CATEGORY_NAMES = {
    'bowl': '碗', 'cup': '杯子', 'hammer': '锤子', 'knife': '刀',
    'ladle': '长柄勺', 'mallet': '木槌', 'mug': '马克杯', 'pot': '锅',
    'saw': '锯', 'scissors': '剪刀', 'scoop': '舀勺', 'shears': '大剪刀',
    'shovel': '铲子', 'spoon': '匙', 'tenderizer': '松肉锤',
    'trowel': '小铲', 'turner': '锅铲',
}
AUTHOR_RULE = 'https://github.com/Reagan1311/OOAL/issues/3#issuecomment-2614175357'


def ooal_label_rule(original, object_instance):
    """Relabel ONLY ladle GT contain pixels; no prediction remapping or geometry edit."""
    remapped = original.copy()
    if object_instance.rsplit('_', 1)[0] == 'ladle':
        remapped[original == 4] = 3
    return remapped


def encode_runs(values):
    flat = np.asarray(values).reshape(-1)
    boundaries = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, flat.size]
    return np.column_stack((flat[boundaries[:-1]], np.diff(boundaries))).reshape(-1).tolist()


def decode_runs(runs, shape):
    values = np.asarray(runs, dtype=np.int64).reshape(-1, 2)
    return np.repeat(values[:, 0], values[:, 1]).reshape(shape)


def foreground_box(labels):
    ys, xs = np.where(labels > 0)
    if not len(xs):
        return [0, 0, labels.shape[1], labels.shape[0]]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_label_png(path, labels, palette):
    # Indexed PNG displays colors while retaining native 0..7 pixel IDs.
    encoded = np.where(labels == -1, 255, labels).astype(np.uint8)
    image = Image.fromarray(encoded).convert('P')
    image.putpalette(palette)
    image.save(path)
    with Image.open(path) as saved:
        if not np.array_equal(np.asarray(saved), encoded):
            raise ValueError(f'Indexed PNG failed lossless round trip: {path}')


def build(config_path):
    config_path = Path(config_path).resolve(strict=True)
    config = read(config_path)
    out = project_path(config['output_dir']).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {out}')
    definitions_path = project_path(config['definitions']).resolve(strict=True)
    split_path = project_path(config['dataset_root']).resolve(strict=True) / 'novel_split.txt'
    assignments = read_split(split_path)
    items, sources, seen = [], [], set()
    for entry in config['manifests']:
        manifest = project_path(entry['path']).resolve(strict=True)
        rows = read_jsonl(manifest)
        if len(rows) != entry['expected_samples']:
            raise ValueError(f'Unexpected sample count for {manifest}: {len(rows)}')
        sources.append({**entry, 'path': str(manifest), 'sha256': sha(manifest)})
        for row in rows:
            if row['sample_id'] in seen:
                raise ValueError(f'Duplicate sample ID: {row["sample_id"]}')
            if assignments[row['object_instance']] != entry['partition']:
                raise ValueError(f'Incorrect novel split: {row["sample_id"]}')
            seen.add(row['sample_id'])
            items.append((entry, row))
    classes = sorted(read(definitions_path)['classes'], key=lambda x: x['id'])
    for cls in classes:
        if cls['id'] == 0:
            cls.update(name='background', name_zh='背景', color=[232, 235, 240])
    palette = [0] * 768
    for cls in classes:
        i = cls['id'] * 3
        palette[i:i+3] = cls['color']
    palette[765:768] = [120, 120, 120]
    out.mkdir(parents=True, exist_ok=True)
    status = {'status': 'building', 'samples': len(items), 'completed': 0}
    write(out / 'status.json', status)
    started = time.perf_counter()
    records, audit_rows = [], []
    counts, instance_counts = Counter(), Counter()
    changed_pixels = 0
    original_totals, remapped_totals = np.zeros(8, np.int64), np.zeros(8, np.int64)
    try:
        for entry, row in tqdm(items, desc='UMD RGB / original GT / OOAL GT', mininterval=2):
            rgb_path, label_path = Path(row['rgb']), Path(row['annotation'])
            if sha(rgb_path) != row['rgb_sha256'] or sha(label_path) != row['annotation_sha256']:
                raise ValueError(f'Input changed since indexing: {row["sample_id"]}')
            with Image.open(rgb_path) as source:
                rgb = source.convert('RGB')
            original, kind = load_annotation(label_path, (rgb.height, rgb.width))
            if kind != row['gt_type'] or entry['partition'] == 'test' and kind != 'manual':
                raise ValueError(f'Unexpected annotation type: {row["sample_id"]}')
            category = row['object_instance'].rsplit('_', 1)[0]
            remapped = ooal_label_rule(original, row['object_instance'])
            changed = original != remapped
            expected = original == 4 if category == 'ladle' else np.zeros_like(changed)
            if not np.array_equal(changed, expected) or not np.array_equal(original > 0, remapped > 0):
                raise ValueError(f'Remapping changed geometry: {row["sample_id"]}')
            runs = encode_runs(original)
            if not np.array_equal(decode_runs(runs, original.shape), original):
                raise ValueError('Browser label encoding is not lossless')
            relative = Path('samples') / row['sample_id']
            target = out / relative
            target.mkdir(parents=True)
            shutil.copyfile(rgb_path, target / 'rgb.jpg')
            if sha(target / 'rgb.jpg') != row['rgb_sha256']:
                raise ValueError('Copied RGB differs from source')
            save_label_png(target / 'original.png', original, palette)
            save_label_png(target / 'ooal_rule.png', remapped, palette)
            Image.fromarray(changed.astype(np.uint8) * 255).save(target / 'changed.png')
            (target / 'labels.js').write_text(
                'window.UMDLabelData(' + json.dumps(row['sample_id']) + ','
                + json.dumps(runs, separators=(',', ':')) + ');\n', encoding='utf-8')
            box = foreground_box(original)
            x0, y0, x1, y1 = box
            padding = max(12, round(max(x1-x0, y1-y0)*.2))
            thumb = rgb.crop((max(0, x0-padding), max(0, y0-padding),
                              min(rgb.width, x1+padding), min(rgb.height, y1+padding)))
            thumb.thumbnail((128, 96), Image.Resampling.LANCZOS)
            thumb.save(target / 'thumb.jpg', quality=85)
            old_count = np.bincount(original[original >= 0], minlength=8)
            new_count = np.bincount(remapped[remapped >= 0], minlength=8)
            record = {
                'id': row['sample_id'], 'instance': row['object_instance'],
                'category': category, 'split': entry['partition'], 'frame': row['frame_index'],
                'annotationType': kind, 'size': [rgb.width, rgb.height], 'box': box,
                'changed': int(changed.sum()), 'foreground': int((original > 0).sum()),
                'void': int((original == -1).sum()), 'countsBefore': old_count.tolist(),
                'countsAfter': new_count.tolist(), 'path': relative.as_posix(),
            }
            records.append(record)
            audit_rows.append({
                **record, 'source_rgb': str(rgb_path), 'source_label': str(label_path),
                'source_rgb_sha256': row['rgb_sha256'], 'source_label_sha256': row['annotation_sha256'],
                'original_png_sha256': sha(target/'original.png'),
                'ooal_rule_png_sha256': sha(target/'ooal_rule.png'),
                'changed_png_sha256': sha(target/'changed.png'),
            })
            changed_pixels += record['changed']
            original_totals += old_count
            remapped_totals += new_count
            counts[entry['partition']] += 1
            counts['changed_images'] += int(record['changed'] > 0)
            instance_counts[row['object_instance']] += 1
            status['completed'] = len(records)
            if len(records) % 100 == 0:
                write(out / 'status.json', status)
        dataset = {
            'version': 1, 'createdAt': datetime.now(timezone.utc).isoformat(),
            'classes': classes, 'categoryNames': CATEGORY_NAMES, 'samples': records,
            'summary': {'total': len(records), **dict(counts), 'changed_pixels': changed_pixels,
                        'categories': len({r['category'] for r in records}),
                        'instances': len(instance_counts), 'countsBefore': original_totals.tolist(),
                        'countsAfter': remapped_totals.tolist()},
            'rule': {'object': 'ladle', 'from': 4, 'to': 3, 'source': AUTHOR_RULE},
        }
        (out / 'data.js').write_text('window.UMDDataset=' + json.dumps(dataset, ensure_ascii=False,
                                      separators=(',', ':')) + ';\n', encoding='utf-8')
        assets = Path(__file__).parent / 'web'
        for name in ['index.html', 'style.css', 'app.js']:
            shutil.copyfile(assets / name, out / name)
        shutil.copytree(assets / 'fonts', out / 'fonts')
        (out / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                                for r in audit_rows), encoding='utf-8')
        write(out / 'audit.json', {
            'summary': dataset['summary'], 'sources': sources,
            'split_sha256': sha(split_path), 'definitions_sha256': sha(definitions_path),
            'rule': dataset['rule'], 'annotation': '*_label.mat / gt_label; dominant single-label GT',
            'scope': 'All current 2661 manual novel/test frames plus 76 author-selected novel/train examples; not all UMD frames.',
            'source_files_modified': False, 'geometry_changed': False,
            'rank_labels_modified': False, 'evaluation_results_modified': False,
            'export': 'original.png and ooal_rule.png are lossless indexed PNG, native IDs 0..7; void -1 encoded 255. changed.png is 0/255.',
            'provenance': 'OOAL-rule test labels are generated locally from the author-confirmed rule; not an official released test-label archive.',
            'checks': ['Every source hash matches its manifest', 'Every RGB copy is byte-identical',
                       'Every indexed label PNG and browser RLE round-trips losslessly',
                       'Only ladle contain pixels change, from 4 to 3', 'Foreground geometry is identical'],
        })
        (out / 'README.md').write_text(
            '# UMD GT 标签核对\n\n直接用浏览器打开 `index.html`；也可复制整个目录到另一台电脑后打开。无需模型或服务器。\n\n'
            '- 覆盖本轮 2,661 张人工测试图与 76 张 OOAL 训练示例，默认只看发生修改的图像。\n'
            '- 唯一修改：物体类别为 ladle 时，将 contain（ID 4）改为 scoop（ID 3）；其余标签和边界保持不变。\n'
            '- “物体放大”只是查看方式，导出的 RGB 和标签均为原始像素尺寸。\n'
            '- `original.png`／`ooal_rule.png`：带调色板的类别图，保留 0–7 原始像素 ID；不是 RGB 标签图。\n'
            '- `changed.png`：0/255 变化区域；`samples.jsonl`、`audit.json` 记录来源与逐图校验。\n'
            '- 修改后的测试标签是依据作者公开规则生成的副本，原始 MAT、rank 标签及实验指标均未改动。\n',
            encoding='utf-8')
        status.update(status='complete', elapsed_seconds=time.perf_counter()-started,
                      summary=dataset['summary'], index=str(out/'index.html'))
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(out / 'status.json', status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    build(parser.parse_args().config)


if __name__ == '__main__':
    main()
