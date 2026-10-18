"""Compare already evaluated predictions; no model selection or threshold tuning."""
import argparse
from collections import defaultdict
import csv
import html
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ...common import read, read_jsonl, sha, write
from ...datasets.umd import load_annotation
from ...evaluation.metrics import scores_from_confusion
from ...visualization import overlay, sheet


STYLE = '<style>body{font-family:system-ui,sans-serif;max-width:1600px;margin:28px auto;padding:0 18px;color:#222;background:#fff}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:9px;text-align:left}img{width:100%;height:auto}section{margin:30px 0}a{color:#145ba0}p{line-height:1.6}nav a{display:inline-block;margin:5px 18px 5px 0}</style>'


def document(title, body):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title>{STYLE}<body><h1>{html.escape(title)}</h1>{body}</body></html>')


def percent(value):
    return f'{value:.2%}' if value is not None else '—'


def compare(ooal_root, ours_root, output):
    ooal_root, ours_root, out = [Path(p).resolve(strict=True) if i < 2 else Path(p).resolve()
                               for i, p in enumerate([ooal_root, ours_root, output])]
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Comparison output must be empty: {out}')
    left, right = read(ooal_root / 'evaluation/metrics.json'), read(ours_root / 'evaluation/metrics.json')
    manifests = [read_jsonl(root / 'evaluation_manifest.jsonl') for root in [ooal_root, ours_root]]
    if manifests[0] != manifests[1] or left['manifest_sha256'] != right['manifest_sha256']:
        raise ValueError('Different evaluation samples; comparison is not valid')
    for key in ['annotation', 'ignore_gt_ids', 'foreground_mIoU', 'gIoU_present_queries', 'P_at_50_present_queries']:
        if left['protocol'][key] != right['protocol'][key]:
            raise ValueError(f'Evaluation protocol differs: {key}')
    if left['failures'] or right['failures']:
        raise ValueError('Inspect evaluation failures before constructing this comparison')
    rows = manifests[0]
    if len(rows) != 2661:
        raise ValueError('Expected the complete 2,661-image manual novel/test benchmark')
    out.mkdir(parents=True)
    (out / 'categories').mkdir()
    classes = [c for c in read(ours_root / 'affordances_snapshot.json')['classes'] if c['id']]
    keys = ['foreground_mIoU', 'gIoU_present_queries', 'P_at_50_present_queries', 'mean_Dice_present_queries']
    headline = {
        'manifest_sha256': left['manifest_sha256'], 'images': len(rows),
        'gt_present_queries': left['metrics']['gt_present_queries'],
        'OOAL': {key: left['metrics'][key] for key in keys},
        'PartGLEE_Valen': {key: right['metrics'][key] for key in keys},
        'OOAL_minus_PartGLEE_Valen_percentage_points': {
            key: 100*(left['metrics'][key]-right['metrics'][key]) for key in keys},
        'training_images': {'OOAL': 76, 'PartGLEE_Valen': 0},
        'protocol': 'novel/test; manual frames; original dominant labels; no ladle relabeling; same evaluator',
        'scope': 'OOAL local UMD adaptation versus frozen PartGLEE + Valen; not the original OOAL paper protocol',
        'source_metrics': {
            'OOAL': {'path': str(ooal_root/'evaluation/metrics.json'), 'sha256': sha(ooal_root/'evaluation/metrics.json')},
            'PartGLEE_Valen': {'path': str(ours_root/'evaluation/metrics.json'), 'sha256': sha(ours_root/'evaluation/metrics.json')},
        },
    }
    write(out / 'comparison.json', headline)
    per_class = []
    for cls in classes:
        name = cls['name']
        a, b = left['metrics']['per_class'][name]['iou'], right['metrics']['per_class'][name]['iou']
        per_class.append({'affordance': name, 'OOAL_IoU': a, 'PartGLEE_Valen_IoU': b,
                          'OOAL_minus_ours_pp': 100*(a-b)})
    with (out / 'per_class.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    per_category = {}
    for result, label in [(left, 'OOAL'), (right, 'PartGLEE_Valen')]:
        groups = defaultdict(lambda: np.zeros((8, 8), dtype=np.int64))
        for instance, metrics in result['per_object'].items():
            groups[instance.rsplit('_', 1)[0]] += np.asarray(metrics['confusion_matrix_gt_rows_pred_columns'])
        for category, matrix in groups.items():
            per_category.setdefault(category, {})[label] = scores_from_confusion(matrix)
    write(out / 'per_object_category.json', per_category)
    sample_metrics = [{r['sample_id']: r for r in read_jsonl(root/'evaluation/per_image.jsonl')}
                      for root in [ooal_root, ours_root]]
    cards = defaultdict(list)
    chosen = []
    category_rows = defaultdict(list)
    for row in rows:
        category_rows[row['object_instance'].rsplit('_', 1)[0]].append(row)
    representative_ids = set()
    for group in category_rows.values():
        representative_ids.update(group[i]['sample_id'] for i in [0, len(group)//2, len(group)-1])
    for row in tqdm(rows, desc='RGB / GT / OOAL / PartGLEE+Valen', mininterval=5):
        sample_id = row['sample_id']
        category = row['object_instance'].rsplit('_', 1)[0]
        rgb = Image.open(row['rgb']).convert('RGB')
        gt, kind = load_annotation(row['annotation'])
        if kind != 'manual':
            raise ValueError('Unexpected automatic test label')
        predictions = [np.asarray(Image.open(root/'samples'/sample_id/'result/class_map.png'))
                       for root in [ooal_root, ours_root]]
        images = [overlay(rgb, [pred == c['id'] for c in classes],
                          colors=[c['color'] for c in classes], labels=[c['name'] for c in classes])
                  for pred in [gt, *predictions]]
        scores = [m[sample_id]['mean_IoU_present_classes'] for m in sample_metrics]
        relative = f'visualizations/{sample_id}.jpg'
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        sheet([('RGB', rgb), ('Original UMD GT', images[0]),
               (f'OOAL / image IoU {percent(scores[0])}', images[1]),
               (f'PartGLEE + Valen / image IoU {percent(scores[1])}', images[2])],
              target, columns=4, width=640, height=520)
        card = (f'<section><p>{html.escape(sample_id)} · OOAL {percent(scores[0])} · '
                f'PartGLEE＋Valen {percent(scores[1])}</p><a href="../{relative}">'
                f'<img loading="lazy" src="../{relative}"></a></section>')
        cards[category].append(card)
        if sample_id in representative_ids:
            chosen.append(card.replace('../visualizations/', 'visualizations/'))
    metric_table = '<table><tr><th>方法</th><th>UMD 训练图</th><th>mIoU</th><th>gIoU</th><th>P@50</th><th>Dice</th></tr>'
    for name, key, count in [('OOAL 本地适配', 'OOAL', 76), ('PartGLEE＋Valen-4B', 'PartGLEE_Valen', 0)]:
        metric_table += f'<tr><td>{name}</td><td>{count}</td>'+''.join(f'<td>{percent(headline[key][k])}</td>' for k in keys)+'</tr>'
    metric_table += '</table>'
    navigation = '<nav>'+''.join(f'<a href="categories/{c}.html">{c}（{len(v)}）</a>' for c, v in sorted(cards.items()))+'</nav>'
    for category, category_cards in cards.items():
        category_scores = per_category[category]
        text = ('<p><a href="../index.html">返回总览</a></p>'
                f'<p>{len(category_cards)} 张 · 此物体类别内前景 mIoU：OOAL '
                f'{percent(category_scores["OOAL"]["foreground_mIoU"])}，PartGLEE＋Valen '
                f'{percent(category_scores["PartGLEE_Valen"]["foreground_mIoU"])}</p>'
                '<p>从左到右：RGB → 原始 GT → OOAL → PartGLEE＋Valen。</p>')
        (out/'categories'/f'{category}.html').write_text(document(category, text+'\n'.join(category_cards)), encoding='utf-8')
    body = ('<p>同一批 UMD novel/test 人工标注图：2,661 张、5,232 个有效图像＋功能对。'
            '原始 dominant 标签，同一评测代码。OOAL 训练 76 张；PartGLEE＋Valen 未在 UMD 上微调。</p>'
            + metric_table + navigation
            + '<p>图片顺序：RGB → 原始 GT → OOAL → PartGLEE＋Valen。图上分数为该图 GT 存在功能的平均 IoU。</p>'
            + '<p>下方每类固定展示按帧序的首、中、末三张，完整结果见上方分类链接。</p>'
            + '\n'.join(chosen))
    (out/'index.html').write_text(document('OOAL 与 PartGLEE＋Valen-4B：UMD 对比', body), encoding='utf-8')
    return headline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ooal', required=True)
    parser.add_argument('--ours', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.ooal, args.ours, args.output)
    print(result)


if __name__ == '__main__':
    main()
