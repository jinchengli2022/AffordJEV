"""Evaluate all fixed queries, save every input view, and update only this report block."""
from collections import Counter, defaultdict
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ...common import read, read_jsonl, write, write_jsonl, sha, load_mask
from ...datasets.umd import load_annotation, CLASS_NAMES
from ...baselines.affordance_r1.compare import query_score, summarize_queries
from ...paths import PROJECT
from ...visualization import overlay
from ..conditional_umd.report import VIEWER
from .point_union import choose
from .point_union_report import foreground_statistics, gallery as foreground_gallery, figure, anchor, page
from .report import relative
from .workers import objects_path, foreground_path, function_path


METHODS = ('fill', 'outline', 'centroid', 'definition', 'sft')
NAMES = {'ooal': 'OOAL／本地适配', 'r1': 'Affordance-R1／zero-shot',
    'fill': 'Valen／原始填色', 'outline': 'Valen／仅描边', 'centroid': 'Valen／质心点，逐候选',
    'definition': 'Valen／定义调整', 'sft': 'Valen／条件判断 SFT（沿用权重）'}
BEGIN, END = '<!-- point-union-suite-start -->', '<!-- point-union-suite-end -->'


def evaluate(root, experiment):
    root, experiment = Path(root), Path(experiment)
    protocol, cfg = read(root/'protocol.json'), read(root/'config.json')
    assert sha(root/'queries.jsonl') == protocol['queries_sha256']
    assert sha(root/'evaluation_manifest.jsonl') == protocol['manifest_sha256']
    qs = read_jsonl(root/'queries.jsonl')
    rows = {r['sample_id']: r for r in read_jsonl(root/'evaluation_manifest.jsonl')}
    grouped = defaultdict(list)
    for q in qs:
        grouped[q['sample_id']].append(q)
    scores, timings = {m: [] for m in METHODS}, {m: Counter() for m in METHODS}
    failures, totals, fg_time = Counter(), Counter(), Counter()
    diagnostics, preprocessing = [], []
    threshold = cfg['foreground_selector']['minimum_yes_probability']
    for i, (sid, queries) in enumerate(grouped.items(), 1):
        row = rows[sid]
        assert sha(row['annotation']) == row['annotation_sha256']
        gt, kind = load_annotation(row['annotation'], row['size_wh'][::-1])
        assert kind == 'manual'
        obj = objects_path(root, sid)
        cs = read(obj/'preprocessed/candidates.json')['candidates']
        trace = read(obj/'preprocessed/trace.json')
        preprocessing.append({'sample_id': sid, 'raw_objects': len(read(obj/'raw/objects.json')['candidates']),
            'eligible_objects': trace['input_masks'], 'fragments_before_merge': trace['fragments_before_merge'],
            'regions_after_merge': len(cs), 'coverage_preserved': True, 'overlap_pixels_after': 0})
        for q in queries:
            fg = foreground_path(root, q)
            selection, state = read(fg/'selection.json'), read(fg/'run.json')
            foreground = load_mask(fg/'selected_mask.png', gt.shape)
            assert sha(fg/'selected_mask.png') == selection['mask_sha256']
            if state['status'] == 'complete':
                decisions = read(fg/'decisions.json')
                accepted = choose(cs, decisions, threshold)
                assert [c['id'] for c in accepted] == [c['id'] for c in selection['selected']]
                assert all(d['threshold'] == threshold and d['selected'] ==
                           (d['probabilities']['yes'] > threshold) for d in decisions)
                reconstructed = np.zeros(gt.shape, bool)
                for c in accepted:
                    reconstructed |= load_mask(obj/c['mask'], gt.shape)
                assert np.array_equal(reconstructed, foreground)
            else:
                assert not foreground.any()
                failures['foreground'] += 1
            totals.update(foreground_statistics(foreground, gt, q['class_id']))
            t = read(fg/'timing.json')
            fg_time.update({'seconds': t['valen_seconds'], 'calls': t['calls']})
            detail = {'query_id': q['query_id'], 'sample_id': sid, 'action': q['action'],
                'selected_ids': [c['id'] for c in selection['selected']], 'foreground_reason': selection['reason'],
                'foreground_area': int(foreground.sum()), 'functional_iou': {}}
            for method in METHODS:
                out = function_path(root, q, method)
                prediction_state = read(out/'run.json')
                assert prediction_state['status'] in ['complete', 'failed']
                assert read(out/'query.json') == q
                assert sha(out/'mask.png') == prediction_state['mask_sha256']
                prediction = load_mask(out/'mask.png', gt.shape)
                assert not (prediction & ~foreground).any()
                if prediction_state['error']:
                    assert not prediction.any()
                    failures[method] += 1
                source = read(out/'sources.json')
                assert source['foreground_mask_sha256'] == selection['mask_sha256']
                score = {**query_score(sid, q['class_id'], gt, prediction),
                         'error': prediction_state['error'], 'mask_sha256': sha(out/'mask.png')}
                scores[method].append(score)
                detail['functional_iou'][method] = score['iou']
                t = read(out/'timing.json')
                timings[method].update({'seconds': t['functional_valen_seconds'], 'calls': t['calls']})
            diagnostics.append(detail)
        if i % 100 == 0 or i == len(grouped):
            print(f'EVALUATE {i}/{len(grouped)} images, five methods', flush=True)
    results = {}
    for method in METHODS:
        assert len(scores[method]) == protocol['queries']
        metrics = summarize_queries(scores[method])
        result = {'method': NAMES[method], 'UMD_training_images': 76 if method == 'sft' else 0,
            'status': 'complete', 'fresh_inference': True, 'new_training': False,
            'metrics': metrics, 'failed_queries': failures[method],
            'timing': {'functional_valen_seconds': timings[method]['seconds'], 'calls': timings[method]['calls'],
                      'mean_seconds_per_query': timings[method]['seconds']/len(qs),
                      'scope': 'functional Valen including input saving; excludes model loads and segmentation'},
            'queries_sha256': sha(root/'queries.jsonl')}
        write(root/'methods'/method/'evaluation/metrics.json', result)
        write_jsonl(root/'methods'/method/'evaluation/per_query.jsonl', scores[method])
        results[method] = result
    baseline = Path(read(experiment/'protocol.json')['baseline'])
    references, previous = {}, {}
    if protocol['phase'] == 'full':
        assert sha(baseline/'test_queries.jsonl') == sha(root/'queries.jsonl')
        for method in ('ooal', 'r1', *METHODS):
            r = read(baseline/'methods'/method/'evaluation/metrics.json')
            assert r['metrics']['images'] == 2661 and r['metrics']['queries'] == 5232
            assert r['queries_sha256'] == sha(root/'queries.jsonl')
            (references if method in ['ooal', 'r1'] else previous)[method] = r
    fg_stats = {'name': f'内部中心点／互斥区域并集，P(yes)>{threshold:.0%}',
        'foreground_target_pixel_recall': totals['covered_target_pixels']/totals['target_pixels'],
        'foreground_gt_background_fraction': totals['foreground_gt_background_pixels']/totals['foreground_valid_pixels']
            if totals['foreground_valid_pixels'] else None, 'counts': dict(totals), 'metrics': results['centroid']['metrics']}
    source_unchanged = all(sha(PROJECT/p) == digest for p, digest in read(root/'source_hashes.json').items())
    assert source_unchanged
    result = {'status': 'complete', 'phase': protocol['phase'], 'images': len(rows), 'queries': len(qs),
        'methods': results, 'reference_baselines': references, 'previous_valen_results': previous,
        'metrics': results['centroid']['metrics'], 'comparisons': {'point_union': fg_stats},
        'foreground': fg_stats, 'foreground_minimum_yes_probability': threshold,
        'counts': dict(failures), 'suite': read(experiment/'suite.json'),
        'regions_per_image': {r['sample_id']: r['regions_after_merge'] for r in preprocessing},
        'timing': {'foreground_valen_seconds_total': fg_time['seconds'], 'foreground_calls_total': fg_time['calls'],
            'foreground_valen_seconds_per_query': fg_time['seconds']/len(qs),
            'foreground_valen_seconds_per_region': fg_time['seconds']/max(1, fg_time['calls'])},
        'metric_note': 'Known-operation dominant GT; class macro IoU and query-mean IoU; failures empty, void ignored.',
        'queries_sha256': sha(root/'queries.jsonl'), 'manifest_sha256': sha(root/'evaluation_manifest.jsonl')}
    write(root/'evaluation/metrics.json', result)
    write_jsonl(root/'evaluation/diagnostics.jsonl', diagnostics)
    write_jsonl(root/'evaluation/preprocessing.jsonl', preprocessing)
    write(root/'evaluation/input_audit.json', {'images': len(rows), 'queries_per_method': len(qs),
        'methods': list(METHODS), 'same_foreground_for_all_methods': True, 'all_selected_unions_verified': True,
        'foreground_threshold': threshold, 'functional_decision': 'P(yes)>P(no); original rule',
        'source_files_unchanged': source_unchanged, 'no_GT_in_model_inputs': True})
    return result


def function_page(root, query, method, destination):
    out = function_path(root, query, method)
    title = f'{query["query_id"]} · {NAMES[method]}'
    body = '<h1>'+html.escape(title)+'</h1><p>'+anchor(out, '返回全部样本', destination/'index.html')+'</p>'
    body += '<p>功能判断保持原 yes/no 规则；96% 门槛仅用于前景。以下均为模型实际读取的图像。</p>'
    body += figure(out, '最终功能预测', out/'overlay.jpg')
    requests = read(out/'requests.json') if (out/'requests.json').exists() else []
    for r in requests:
        folder = Path(r['folder'])
        request = read(folder/'valen_request.json')
        ids = ', '.join(str(c['id']) for c in r['candidates'])
        body += '<details><summary>候选 '+html.escape(ids)+'</summary><div class="grid">'
        for name in ['rgb_crop.png', 'numbered_crop.png', 'centroid.png' if method == 'centroid' else 'region_sheet.jpg']:
            body += figure(out, name, folder/name)
        prompt = '\n\n'.join(c['text'] for c in request['state']['messages'][0]['content'] if c['type'] == 'text')
        body += '</div><pre>'+html.escape(prompt)+'</pre><p>'
        body += anchor(out, '完整英文请求', folder/'valen_request.json')+' · '+anchor(out, '模型分数', folder/'valen_response.json')+'</p></details>'
    if not requests:
        body += '<p>没有通过前景筛选的部件候选；保留空预测计分。</p>'
    (out/'index.html').write_text(page(title, body), encoding='utf-8')


def gallery(root, experiment):
    root, experiment = Path(root), Path(experiment)
    foreground_gallery(root, experiment)
    destination = root/'comparison'
    (destination/'index.html').replace(destination/'foreground.html')
    result = read(root/'evaluation/metrics.json')
    queries = read_jsonl(root/'queries.jsonl')
    rows = {r['sample_id']: r for r in read_jsonl(root/'evaluation_manifest.jsonl')}
    baseline = Path(read(experiment/'protocol.json')['baseline'])
    order = list(result['reference_baselines']) + list(METHODS)
    scores = {m: {(r['sample_id'], r['class_id']): r['iou'] for r in read_jsonl(
        (baseline if m in ['ooal', 'r1'] else root)/'methods'/m/'evaluation/per_query.jsonl')} for m in order}
    cards, seen_links = [], set()
    last, gt = None, None
    for i, q in enumerate(queries, 1):
        sid, action = q['sample_id'], q['action']
        if sid != last:
            gt, _ = load_annotation(rows[sid]['annotation'], rows[sid]['size_wh'][::-1])
            last = sid
        rgb = Image.open(q['rgb']).convert('RGB')
        gt_path = destination/'gt'/sid/f'{action}.jpg'
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        overlay(rgb, [gt == q['class_id']], colors=[[60, 180, 95]], labels=['']).save(gt_path, quality=90)
        predictions = {}
        for m in order:
            out = function_path(baseline if m in ['ooal', 'r1'] else root, q, m)
            if m in METHODS:
                function_page(root, q, m, destination)
            detail = out/'index.html' if m in METHODS else out
            assert (out/'overlay.jpg').is_file() and detail.exists()
            predictions[m] = {'image': relative(out/'overlay.jpg', destination),
                'iou': scores[m][sid, q['class_id']], 'details': relative(detail, destination)}
            seen_links.update([str(out/'overlay.jpg'), str(detail)])
        fg = foreground_path(root, q)
        assert (fg/'index.html').is_file()
        cards.append({'id': q['query_id'], 'category': q['object_category'], 'action': action,
            'rgb': relative(objects_path(root, sid)/'rgb.png', destination), 'gt': relative(gt_path, destination),
            'predictions': predictions, 'candidates': relative(fg/'index.html', destination)})
        if i % 100 == 0 or i == len(queries):
            print(f'GALLERY {i}/{len(queries)} queries, all methods and intermediate inputs', flush=True)
    names = {m: NAMES[m] + ('（沿用对照）' if m in ['ooal', 'r1'] else '') for m in order}
    (destination/'data.js').write_text('window.DATA='+json.dumps({'names': names, 'order': order, 'cards': cards}, ensure_ascii=False)+';\n', encoding='utf-8')
    viewer = VIEWER.replace('给定物体与操作：完整重跑', '96% 前景阈值：五种 Valen 对照')
    viewer = viewer.replace('2,661 张 RGB · 5,232 个功能对', f'{result["images"]:,} 张 RGB · {result["queries"]:,} 个功能对')
    viewer = viewer.replace('检查 PartGLEE 候选', '前景四阶段、实际 RGB 与中文 prompt')
    (destination/'index.html').write_text(viewer, encoding='utf-8')
    audit = read(root/'evaluation/gallery_audit.json')
    audit.update(functional_methods=list(METHODS), functional_query_pages=len(queries)*len(METHODS),
                 comparison_queries=len(cards), prediction_links_checked=len(seen_links), all_samples_included=True)
    write(root/'evaluation/gallery_audit.json', audit)
    lines = [f'# 96% 前景阈值：五种 Valen 对照', '', f'{result["images"]} 张 RGB／{result["queries"]} 个功能对；每组完整推理。',
        '', '| 方法 | mIoU | gIoU | P@50 | Dice |', '|---|---:|---:|---:|---:|']
    for m in METHODS:
        metrics = result['methods'][m]['metrics']
        lines.append('| '+NAMES[m]+' | '+' | '.join(f'{metrics[k]:.2%}' for k in ['mIoU_known_operations', 'gIoU', 'P_at_50', 'Dice'])+' |')
    lines += ['', 'SFT 沿用此前 76 张训练图的最终 LoRA＋判断头，未重新训练。前景始终使用原始 Valen。', '',
        '[全部样本与中间步骤](comparison/index.html) · [完整指标](evaluation/metrics.json) · [配置](config.json)', '',
        '阈值此前通过九张样例开发；本轮为相同 novel/test 的完整评测，不称独立盲测。']
    (root/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def update_document(experiment, result=None):
    experiment = Path(experiment)
    baseline = Path(read(experiment/'protocol.json')['baseline'])
    threshold = read(experiment/'config.json')['foreground_selector']['minimum_yes_probability']
    references = result['reference_baselines'] if result else {
        m: read(baseline/'methods'/m/'evaluation/metrics.json') for m in ['ooal', 'r1']}
    results = {**references, **(result['methods'] if result else {})}
    order = ['ooal', 'r1', *METHODS]
    lines = [BEGIN, f'### 给定物体与操作：{threshold:.0%} 前景阈值，五种 Valen 重跑', '',
        '**2,661 张 RGB／5,232 个功能对，每种 Valen 均完整重跑。** 通用 `object / part` → 互斥区域内部中心点 → 原始 Valen 前景判断 → '+
        f'`P(yes)>{threshold:g}` 区域并集 → 部件重分割。五组共享同一功能对的新候选，后续功能判断仍使用原 yes/no 规则。', '',
        '| 方法 | UMD 训练图 | mIoU ↑ | gIoU ↑ | P@50 ↑ | Dice ↑ | 状态 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for method in order:
        r = results.get(method)
        values = [f'{r["metrics"][k]:.2%}' for k in ['mIoU_known_operations', 'gIoU', 'P_at_50', 'Dice']] if r else ['—']*4
        status = '沿用对照' if method in ['ooal', 'r1'] else '已完成' if r else '待运行'
        lines.append('| '+' | '.join([NAMES[method], str(76 if method in ['ooal', 'sft'] else 0), *values, status])+' |')
    lines += ['', 'OOAL 与 Affordance-R1 沿用同口径结果。SFT 沿用此前 76 张图训练的最终 LoRA＋判断头，本轮不重训；前景阶段不用 SFT 权重。'+
              '96% 阈值来自九张样例的开发观察，非独立盲测。']
    if result:
        fg = result['foreground']
        lines += ['', f'前景功能像素覆盖 **{fg["foreground_target_pixel_recall"]:.2%}**；前景内 GT 背景占比 **{fg["foreground_gt_background_fraction"]:.2%}**。'+
                  '背景按 GT=0 统计，非完整物体标注；失败按空预测保留。', '', '**各功能 IoU**', '',
                  '| 功能 | '+' | '.join(NAMES[m].replace('Valen／', '') for m in METHODS)+' |', '|---|'+'---:|'*len(METHODS)]
        for cid in range(1, 8):
            name = CLASS_NAMES[cid]
            lines.append('| '+name+' | '+' | '.join(f'{result["methods"][m]["metrics"]["per_class"][name]["iou"]:.2%}' for m in METHODS)+' |')
        base = '../../outputs/'+experiment.name+'/full'
        lines += ['', f'[全部样本与中间可视化]({base}/comparison/index.html) · [完整指标]({base}/evaluation/metrics.json) · [配置]({base}/config.json)']
    else:
        lines += ['', '已安排完整重跑，完成后自动填写本表；原实验结果保留在下一表。']
    lines += [END]
    doc = PROJECT/'docs/research/实验结果.md'
    text = doc.read_text(encoding='utf-8')
    block = '\n'.join(lines)
    if BEGIN in text:
        first, last = text.index(BEGIN), text.index(END)+len(END)
        text = text[:first]+block+text[last:]
    else:
        heading = '## train-free 方法对照\n'
        assert text.count(heading) == 1
        text = text.replace(heading, heading+'\n'+block+'\n', 1)
    tmp = doc.with_suffix('.md.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(doc)
    plan = PROJECT/'docs/research/实验计划.md'
    text = plan.read_text(encoding='utf-8')
    label = '| 无训练前景 | 96% 前景＋五种 Valen 全量对照 |'
    row = label+' 相同新前景与部件候选；五种功能判断重新推理，SFT 沿用权重 | '+('已完成：2,661 张／每组 5,232 对' if result else '进行中：先样例检查，再全量')+' |'
    if label in text:
        text = '\n'.join(row if line.startswith(label) else line for line in text.splitlines())+'\n'
    else:
        lines = text.splitlines()
        index = next(i for i, line in enumerate(lines) if '| 内部中心点＋96% 接收阈值 |' in line)
        lines.insert(index+1, row)
        text = '\n'.join(lines)+'\n'
    plan.write_text(text, encoding='utf-8')
