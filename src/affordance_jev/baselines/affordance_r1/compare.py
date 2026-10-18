"""Read-only comparison under full-label and known-operation protocols."""
import argparse
import base64
from collections import Counter
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ...common import read, read_jsonl, write, write_jsonl, sha
from ...datasets.umd import CLASS_NAMES, load_annotation
from ...experiments.umd_valen.report import RUNS
from ...paths import PROJECT
from ...visualization import overlay, sheet


def summarize_queries(queries):
    keys = [(r['sample_id'], r['class_id']) for r in queries]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate image/operation queries')
    per_class = {}
    for cid in range(1, 8):
        rows = [r for r in queries if r['class_id'] == cid]
        summed = {k: sum(r[k] for r in rows) for k in ['intersection', 'union', 'gt_pixels', 'pred_pixels']}
        assert rows and summed['union'] > 0
        per_class[CLASS_NAMES[cid]] = {'class_id': cid, 'queries': len(rows), **summed,
            'iou': summed['intersection']/summed['union'],
            'dice': 2*summed['intersection']/(summed['gt_pixels']+summed['pred_pixels'])}
    ious = np.array([r['intersection']/r['union'] for r in queries])
    dice = np.array([2*r['intersection']/(r['gt_pixels']+r['pred_pixels']) for r in queries])
    return {'images': len({r['sample_id'] for r in queries}), 'queries': len(queries),
            'mIoU_known_operations': float(np.mean([r['iou'] for r in per_class.values()])),
            'gIoU': float(ious.mean()), 'P_at_50': float((ious >= .5).mean()), 'Dice': float(dice.mean()),
            'per_class': per_class}


def query_score(sid, cid, gt, prediction):
    valid = gt != -1
    target = (gt == cid) & valid
    prediction = np.asarray(prediction, dtype=bool) & valid
    intersection = int((target & prediction).sum())
    union = int((target | prediction).sum())
    assert target.any()
    return {'sample_id': sid, 'class': CLASS_NAMES[cid], 'class_id': cid,
            'gt_pixels': int(target.sum()), 'pred_pixels': int(prediction.sum()),
            'intersection': intersection, 'union': union,
            'iou': intersection/union, 'dice': 2*intersection/(target.sum()+prediction.sum())}


def known_operations(r1_root, output, include_r1=True):
    manifest = PROJECT/'data/UMD/manifests/umd_novel_test.jsonl'
    rows = read_jsonl(manifest)
    digest = sha(manifest)
    runs = {**RUNS, 'centroid': ('Valen／质心点，逐候选', PROJECT/'outputs/umd_centroid_20261017/test')}
    results, per_query, audit = {}, {}, {}
    expected_keys = None
    for key, (name, root) in runs.items():
        original = read(root/'evaluation/metrics.json')
        if original['manifest_sha256'] != digest:
            raise ValueError(f'Different evaluation manifest: {key}')
        queries = read_jsonl(root/'evaluation/per_query.jsonl')
        keys = {(r['sample_id'], r['class_id']) for r in queries}
        if expected_keys is None:
            expected_keys = keys
        assert keys == expected_keys and len(keys) == 5232
        result = summarize_queries(queries)
        for new, old in [('gIoU','gIoU_present_queries'), ('P_at_50','P_at_50_present_queries'),
                         ('Dice','mean_Dice_present_queries')]:
            assert abs(result[new] - original['metrics'][old]) < 1e-12
        assert result['mIoU_known_operations'] + 1e-12 >= original['metrics']['foreground_mIoU']
        results[key] = {'name': name, 'UMD_training_images': 76 if key in ['ooal','s0'] else 0,
                        'source': str(root), 'metrics': result}
        per_query[key] = queries
        audit[key] = {'source_query_counts_sha256': sha(root/'evaluation/per_query.jsonl'),
                      'original_mIoU': original['metrics']['foreground_mIoU'],
                      'other_three_metrics_match_original': True}
    if include_r1:
        if read(r1_root/'run.json')['status'] != 'complete':
            raise ValueError('Affordance-R1 inference is not complete')
        if sha(r1_root/'evaluation_manifest.jsonl') != digest:
            raise ValueError('R1 evaluated a different manifest')
        queries, errors = [], 0
        for i, row in enumerate(rows, 1):
            gt, kind = load_annotation(row['annotation'], row['size_wh'][::-1])
            assert kind == 'manual' and sha(row['annotation']) == row['annotation_sha256']
            sample = r1_root/'samples'/row['sample_id']
            state = read(sample/'run.json')
            assert state['status'] == 'complete'
            records = {r['class_id']: r for r in read(sample/'responses.json')}
            for cid in sorted(int(c) for c in np.unique(gt) if c > 0):
                path = sample/'raw_masks'/f'{CLASS_NAMES[cid]}.png'
                mask = np.asarray(Image.open(path)) > 0
                assert mask.shape == gt.shape
                q = query_score(row['sample_id'], cid, gt, mask)
                q['mask_sha256'] = sha(path)
                q['error'] = records[cid]['error']
                errors += records[cid]['error'] is not None
                queries.append(q)
            if i % 500 == 0:
                print(f'KNOWN-OP R1 {i}/{len(rows)}', flush=True)
        assert {(r['sample_id'], r['class_id']) for r in queries} == expected_keys
        results['r1'] = {'name': 'Affordance-R1／zero-shot（官方 prompt，类别已知）',
                         'UMD_training_images': 0, 'source': str(r1_root),
                         'metrics': summarize_queries(queries), 'failed_queries': errors}
        per_query['r1'] = queries
    output.mkdir(parents=True, exist_ok=True)
    document = {'protocol': {
        'task': 'Given RGB and operation, predict its binary region',
        'images': 2661, 'queries': 5232, 'manifest_sha256': digest,
        'query_selection': 'GT-present image/operation pairs, identical for every method',
        'mIoU': 'For each of 7 classes, sum intersections/unions over its selected queries; then macro-average',
        'gIoU': 'Mean IoU over the same 5232 queries',
        'P_at_50': 'Fraction of queries with IoU >= 0.5', 'Dice': 'Mean binary Dice over queries',
        'background': 'Background false positives counted inside every queried image; void excluded',
        'absent_class_fp': 'Queries for classes absent from GT are outside this given-operation task',
        'r1_prediction': 'Independent raw SAM2 mask for each requested operation, before cross-operation merging',
        'other_predictions': 'The requested class mask extracted unchanged from the saved semantic prediction',
        'extra_r1_input': 'Known object category, explicitly requested by user; PartGLEE uses a shared object/part vocabulary',
        'model_inference_repeated': False, 'original_results_unchanged': True,
        'paper_UMD_protocol_reproduction': False}, 'results': results, 'audit': audit}
    write(output/'metrics.json', document)
    for key, queries in per_query.items():
        write_jsonl(output/key/'per_query.jsonl', queries)
    return document


def gallery(r1_root, output, known):
    rows = read_jsonl(r1_root/'evaluation_manifest.jsonl')
    classes = [c for c in read(r1_root/'affordances_snapshot.json')['classes'] if c['id']]
    query_sets = {k: {(r['sample_id'], r['class_id']): r
                     for r in read_jsonl(output/'known_operations'/k/'per_query.jsonl')}
                  for k in ['r1', 'centroid', 's0']}
    comparison_roots = {
        'centroid': PROJECT/'outputs/umd_centroid_20261017/test',
        's0': PROJECT/'outputs/valen_76/s0_test'}
    records = []
    overlaps, failures, empty, tokens, lengths = 0, 0, 0, [], 0
    for i, row in enumerate(rows, 1):
        sample = r1_root/'samples'/row['sample_id']
        state = read(sample/'run.json')
        overlaps += state['overlap_pixels_before_merge']
        responses = read(sample/'responses.json')
        failures += sum(r['error'] is not None for r in responses)
        empty += sum(not r['regions'] and r['error'] is None for r in responses)
        tokens.extend(r['generated_tokens'] for r in responses)
        lengths += sum(r['finish_reason'] == 'length' for r in responses)
        rgb = Image.open(row['rgb']).convert('RGB')
        gt, _ = load_annotation(row['annotation'])
        pred = np.asarray(Image.open(sample/'result/class_map.png'))
        def color(label):
            return overlay(rgb, [label == c['id'] for c in classes],
                           colors=[c['color'] for c in classes], labels=[c['name'] for c in classes])
        image_path = output/'images'/f"{row['sample_id']}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        sheet([('RGB', rgb), ('Original dominant GT', color(gt)), ('R1: all 7 merged', color(pred))],
              image_path, columns=3, width=480, height=395)
        queries = []
        for cid in sorted(int(c) for c in np.unique(gt) if c > 0):
            cls = next(c for c in classes if c['id'] == cid)
            mask = np.asarray(Image.open(sample/'raw_masks'/f'{cls["name"]}.png')) > 0
            panels = [('RGB (display crop)', rgb)]
            visible = (gt > 0) | mask
            for title, m in [('GT: '+cls['name'], gt == cid), ('R1: raw query mask', mask)]:
                panels.append((title, overlay(rgb, [m], colors=[cls['color']], labels=[''])))
            for key, title in [('centroid','Valen: centroid'),('s0','Valen: SFT')]:
                p = np.asarray(Image.open(comparison_roots[key]/'samples'/row['sample_id']/'result/class_map.png'))
                visible |= p == cid
                panels.append((title, overlay(rgb, [p == cid], colors=[cls['color']], labels=[''])))
            # Zoom only the saved presentation. Include all GT/predicted pixels,
            # including distant false positives; scoring and model RGB stay full-frame.
            ys, xs = np.where(visible)
            padding = max(20, int(max(xs.max()-xs.min(), ys.max()-ys.min())*.15))
            crop = [max(0,int(xs.min())-padding), max(0,int(ys.min())-padding),
                    min(rgb.width,int(xs.max())+padding+1), min(rgb.height,int(ys.max())+padding+1)]
            panels = [(title, image.crop(crop)) for title, image in panels]
            relative = f"queries/{row['sample_id']}/{cls['name']}.jpg"
            target = output/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            sheet(panels, target, columns=5, width=320, height=270)
            queries.append({'name': cls['name'], 'image': relative, 'display_only_crop_xyxy': crop,
                            'iou': {k: query_sets[k][(row['sample_id'], cid)]['iou'] for k in query_sets}})
        records.append({'id': row['sample_id'], 'category': row['object_instance'].rsplit('_', 1)[0],
                        'image': f"images/{row['sample_id']}.jpg", 'queries': queries,
                        'responses': '../test/samples/'+row['sample_id']+'/responses.json',
                        'failures': state['query_errors']})
        if i % 200 == 0:
            print(f'GALLERY {i}/{len(rows)}', flush=True)
    audit = {'images': len(rows), 'all_operation_queries': len(tokens), 'known_operation_queries': 5232,
             'parse_or_segmentation_errors': failures, 'valid_empty_answers': empty,
             'generation_length_limit': lengths, 'mean_generated_tokens': float(np.mean(tokens)),
             'sum_overlap_pixels_before_merge': overlaps,
             'official_prompt_audit': str(r1_root.parent/'official_prompt_audit.json')}
    write(output/'inference_audit.json', audit)
    data = {'samples': records, 'known_metrics': known['results'],
            'full_metrics': read(r1_root/'evaluation/metrics.json')['metrics'], 'audit': audit}
    (output/'data.js').write_text('window.DATA='+json.dumps(data, ensure_ascii=False)+';\n', encoding='utf-8')
    write_viewer(output)
    return audit


def write_viewer(output):
    assets = Path(__file__).parent/'assets'
    encoded = base64.b64encode((assets/'experiment-ui.woff').read_bytes()).decode('ascii')
    (output/'index.html').write_text(VIEWER.replace('__FONT_BASE64__', encoded), encoding='utf-8')
    shutil.copy2(assets/'OFL.txt', output/'FONT_LICENSE.txt')


VIEWER = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Affordance-R1 · UMD</title>
<style>@font-face{font-family:ExperimentCJK;src:url(data:font/woff;base64,__FONT_BASE64__) format('woff');font-display:block}
body{font:16px ExperimentCJK,system-ui;margin:28px auto;max-width:1500px;padding:0 20px;color:#222;background:#fff}
h1{font-size:25px}p{line-height:1.6}select,input,button{font:inherit;padding:7px;margin:5px}img{max-width:100%;display:block}
section{margin:25px 0;border-top:1px solid #ddd;padding-top:14px}summary{cursor:pointer;padding:10px 0}a{color:#145c93}
.controls{position:sticky;top:0;background:#fff;padding:6px 0;border-bottom:1px solid #ddd}.muted{color:#666}</style>
<h1>Affordance-R1／zero-shot · UMD</h1>
<p>官方 prompt＋已知物体类别＋操作；原始权重，无 UMD 微调。展开样本可看各指定操作的独立 mask。
上图：RGB → 原始 GT → 七类合并预测。下图：RGB → GT → R1 → Valen 质心点 → Valen SFT。
下图仅为展示放大，包含全部真实／预测区域；模型推理与评分均使用完整图像。</p>
<p id="metric"></p><p><a href="../test/evaluation/metrics.json">全部功能指标</a> · <a href="known_operations/metrics.json">给定操作对照</a> · <a href="inference_audit.json">运行核对</a></p>
<div class="controls"><select id="category"><option value="">全部物体类别</option></select>
<input id="search" placeholder="样本 ID"><select id="action"><option value="">全部操作</option></select>
<button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div><main id="cards"></main>
<script src="data.js"></script><script>
const $=id=>document.getElementById(id),pct=x=>(100*x).toFixed(2)+'%';let page=0;
const d=window.DATA,m=d.full_metrics,k=d.known_metrics.r1.metrics;
$('metric').textContent='全部功能 mIoU '+pct(m.foreground_mIoU)+'；给定操作 mIoU '+pct(k.mIoU_known_operations)+'，gIoU '+pct(k.gIoU)+'，P@50 '+pct(k.P_at_50)+'。';
for(const c of [...new Set(d.samples.map(r=>r.category))])$('category').add(new Option(c,c));
for(const c of [...new Set(d.samples.flatMap(r=>r.queries.map(q=>q.name)))])$('action').add(new Option(c,c));
function render(){let rows=d.samples.filter(r=>(!$('category').value||r.category==$('category').value)&&r.id.includes($('search').value)&&(!$('action').value||r.queries.some(q=>q.name==$('action').value)));
const pages=Math.max(1,Math.ceil(rows.length/12));page=Math.max(0,Math.min(page,pages-1));$('page').textContent=`${page+1}/${pages} · ${rows.length} 张`;
$('cards').innerHTML=rows.slice(page*12,page*12+12).map(r=>`<section><b>${r.id}</b> · <a href="${r.responses}">完整 prompt／输出／框点</a><a href="${r.image}"><img loading="lazy" src="${r.image}"></a><details><summary>查看指定操作的独立 mask</summary>${r.queries.filter(q=>!$('action').value||q.name==$('action').value).map(q=>`<p>${q.name} · IoU：R1 ${pct(q.iou.r1)} / 质心点 ${pct(q.iou.centroid)} / SFT ${pct(q.iou.s0)}</p><a href="${q.image}"><img loading="lazy" src="${q.image}"></a>`).join('')}</details></section>`).join('');}
for(const id of ['category','action','search'])$(id).addEventListener('input',()=>{page=0;render()});
$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--existing-only', action='store_true')
    parser.add_argument('--gallery', action='store_true')
    args = parser.parse_args()
    (args.output/'provenance').mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, args.output/'provenance/compare.py')
    write(args.output/'provenance/source_sha256.json', {'compare.py': sha(__file__)})
    result = known_operations(args.root, args.output/'known_operations', include_r1=not args.existing_only)
    if args.gallery:
        gallery(args.root, args.output, result)
    for key, row in result['results'].items():
        print(key, row['metrics']['mIoU_known_operations'], row['metrics']['gIoU'], flush=True)


if __name__ == '__main__':
    main()
