"""GT-only reporting after inference. Never used by model workers."""
from collections import Counter, defaultdict
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ...common import read, write, read_jsonl, write_jsonl, sha, load_mask
from ...datasets.umd import load_annotation
from ...baselines.affordance_r1.compare import query_score, summarize_queries
from ...visualization import overlay
from ...paths import PROJECT
from .selection import choose_foreground
from .workers import objects_path, foreground_path, function_path


def evaluate(root, experiment):
    root, experiment = Path(root), Path(experiment)
    from .point_union import enabled
    if enabled(read(root/'config.json')):
        from .point_union_report import evaluate as evaluate_points
        return evaluate_points(root, experiment)
    qs = read_jsonl(root/'queries.jsonl')
    protocol = read(root/'protocol.json')
    assert sha(root/'queries.jsonl') == protocol['queries_sha256']
    rows = {r['sample_id']: r for r in read_jsonl(root/'evaluation_manifest.jsonl')}
    by_image = defaultdict(list)
    for q in qs:
        by_image[q['sample_id']].append(q)
    baseline = Path(read(experiment/'protocol.json')['baseline'])
    old_rows = {(r['sample_id'], r['class_id']): r for r in
                read_jsonl(baseline/'methods/centroid/evaluation/per_query.jsonl')}
    scores, old_scores, details = [], [], []
    counts = Counter()
    timing_sum = Counter()
    part_uses = Counter(read(foreground_path(root, q)/'parts.json')['path'] for q in qs)
    gt_pixels = covered = candidate_covered = old_covered = 0
    for sid, queries in by_image.items():
        row = rows[sid]
        assert sha(row['annotation']) == row['annotation_sha256']
        gt, kind = load_annotation(row['annotation'], tuple(row['size_wh'][::-1]))
        assert kind == 'manual'
        object_out = objects_path(root, sid)
        old_fg = load_mask(baseline/'candidates_test/samples'/sid/'foreground/object_mask.png', gt.shape)
        object_time = read(object_out/'timing.json')['object_forward_seconds'] if (object_out/'timing.json').exists() else 0.
        timing_sum['object_forward_seconds_total'] += object_time
        for q in queries:
            fg, out = foreground_path(root, q), function_path(root, q)
            state, selection = read(out/'run.json'), read(fg/'selection.json')
            assert state['status'] in ['complete', 'failed']
            assert read(out/'query.json') == q
            assert sha(out/'mask.png') == state['mask_sha256']
            prediction = load_mask(out/'mask.png', gt.shape)
            selected_mask = load_mask(fg/'selected_mask.png', gt.shape)
            assert sha(fg/'selected_mask.png') == selection['mask_sha256']
            if selection['selected']:
                candidates = read(object_out/'eligible.json')['candidates']
                decisions = read(fg/'decisions.json')
                expected = choose_foreground(candidates, decisions)
                assert expected['raw_id'] == selection['selected']['raw_id']
                assert np.array_equal(selected_mask, load_mask(object_out/expected['mask'], gt.shape))
            elif selection['reason'] == 'no_target':
                assert choose_foreground(read(object_out/'eligible.json')['candidates'], read(fg/'decisions.json')) is None
            if selection['reason'] != 'selected':
                assert not prediction.any() and not selected_mask.any()
            assert not (prediction & ~selected_mask).any()
            if state['error']:
                assert not prediction.any()
                counts['failed_queries'] += 1
            counts['foreground_'+selection['reason']] += 1
            source = read(out/'sources.json')
            counts['functional_'+source['reason']] += 1
            part = read(fg/'parts.json')
            target = gt == q['class_id']
            target_pixels = int(target.sum())
            gt_pixels += target_pixels
            covered += int((target & selected_mask).sum())
            old_covered += int((target & old_fg).sum())
            part_seconds = 0.
            if part['status'] == 'ready':
                part_out = root/part['path']
                labels = np.asarray(Image.open(part_out/'regions/label_map.png'))
                assert labels.shape == gt.shape
                candidate_covered += int(((labels > 0) & target).sum())
                part_seconds = read(part_out/'partglee_runtime.json')['timings']['part_forward_seconds']/part_uses[part['path']]
            fg_t, fn_t = read(fg/'timing.json'), read(out/'timing.json')
            timing = {'object_forward_seconds_amortized': object_time/len(queries),
                'foreground_valen_seconds': fg_t['valen_seconds'], 'part_forward_seconds_amortized': part_seconds,
                'functional_valen_seconds': fn_t['functional_valen_seconds']}
            timing['pipeline_components_seconds_amortized'] = sum(timing.values())
            timing_sum.update(timing)
            timing_sum['valen_compiled_save_seconds_total'] += fg_t['compiled_save_seconds'] + fn_t['compiled_save_seconds']
            score = query_score(sid, q['class_id'], gt, prediction)
            scores.append({**score, 'error': state['error']})
            old_scores.append(old_rows[(sid, q['class_id'])])
            details.append({'query_id': q['query_id'], 'selected_raw_id': selection['selected']['raw_id'] if selection['selected'] else None,
                'foreground_reason': selection['reason'], 'part_path': part['path'],
                'foreground_target_recall': int((target & selected_mask).sum())/target_pixels,
                'new_iou': score['iou'], 'old_iou': old_rows[(sid, q['class_id'])]['iou'], 'timing': timing})
    assert len(scores) == protocol['queries'] and len(by_image) == protocol['images']
    result = {'status': 'complete', 'phase': protocol['phase'], 'UMD_training_images': 0,
        'metrics': summarize_queries(scores), 'baseline_metrics': summarize_queries(old_scores),
        'counts': dict(counts), 'foreground_target_pixel_recall': covered/gt_pixels,
        'baseline_foreground_target_pixel_recall': old_covered/gt_pixels,
        'candidate_target_pixel_recall': candidate_covered/gt_pixels,
        'rejection_rate': counts['foreground_no_target']/len(qs),
        'unique_part_segmentations': sum(k is not None for k in part_uses),
        'timing': {'mean_per_query': {k: v/len(qs) for k, v in timing_sum.items() if not k.endswith('_total')},
            'compiled_input_save_seconds_total': timing_sum['valen_compiled_save_seconds_total'],
            'scope': 'Amortized stage components in parallel offline runs; RGB object and identical-mask part computations shared. '
                     'Valen includes preprocessing and input audit saving; excludes model loading and other artifact IO. Not live serial latency.'},
        'queries_sha256': sha(root/'queries.jsonl'), 'gt_in_model_inputs': False}
    write_jsonl(root/'evaluation/per_query.jsonl', scores)
    write_jsonl(root/'evaluation/diagnostics.jsonl', details)
    write(root/'evaluation/metrics.json', result)
    return result


STYLE = 'body{font:16px system-ui;margin:24px;background:white;color:#222}h1{font-size:24px}h2{font-size:18px}.grid{display:grid;grid-template-columns:repeat(3,minmax(200px,1fr));gap:12px}img{width:100%;object-fit:contain}article{border:1px solid #ddd;padding:10px}a{color:#1764a2}pre{white-space:pre-wrap}button,select{font:inherit;padding:6px;margin:4px} @media(max-width:800px){.grid{grid-template-columns:1fr}}'


def relative(path, folder):
    return os.path.relpath(path, folder)


def detail_page(root, q):
    out, fg = function_path(root, q), foreground_path(root, q)
    source = read(out/'sources.json')
    body = [f'<h1>{html.escape(q["query_id"])}</h1><p>物体：{html.escape(q["object_category"])}；操作：{html.escape(q["action"])}；{html.escape(source["reason"])}</p>']
    def panels(title, files):
        body.append('<h2>'+html.escape(title)+'</h2><div class="grid">')
        for label, f in files:
            if not f.is_file():
                continue
            url = html.escape(relative(f, out), quote=True)
            body.append(f'<article>{html.escape(label)}<a href="{url}"><img loading="lazy" src="{url}"></a></article>')
        body.append('</div>')
    def links(folder, names):
        for name in names:
            if (folder/name).exists():
                body.append(f'<a href="{html.escape(relative(folder/name,out),quote=True)}">{html.escape(name)}</a> · ')
    obj = objects_path(root, q['sample_id'])
    panels('1. 原始物体候选与选定前景', [('RGB', obj/'rgb.png'), ('所有原始物体候选', obj/'raw/object_candidates.jpg'), ('Valen 选择', fg/'overlay.jpg')])
    links(fg, ['selection.json', 'decisions.json', 'timing.json'])
    for group in sorted(fg.glob('group_*')):
        panels('2. 前景判断实际输入 '+group.name, [(n, group/n) for n in ['rgb.png', 'numbered.png', 'region_sheet.jpg']])
        links(group, ['valen_request.json', 'valen_response.json', 'valen_compiled/prompt.txt', 'valen_runtime.json'])
    if source['parts']:
        part = root/source['parts']
        panels('3. 前景内重新分割部件', [(n, part/n) for n in ['foreground/input_crop.png', 'raw/part_candidates.jpg', 'regions/numbered_crop.png']])
        links(part, ['candidates.json', 'partglee_runtime.json', 'geometry_audit.json'])
    for region in sorted(out.glob('region_*')):
        panels('4. 功能判断实际输入 '+region.name, [(n, region/n) for n in ['rgb_crop.png', 'numbered_crop.png', 'centroid.png']])
        links(region, ['valen_request.json', 'valen_response.json', 'valen_compiled/prompt.txt', 'valen_runtime.json'])
    panels('5. 最终功能区域', [('预测', out/'overlay.jpg')])
    links(out, ['decisions.json', 'sources.json', 'timing.json', 'run.json', 'mask.png'])
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Valen 前景分步检查</title><style>'+STYLE+'</style>'+''.join(body)+'</html>', encoding='utf-8')


def gallery(root, experiment):
    root, experiment = Path(root), Path(experiment)
    from .point_union import enabled
    if enabled(read(root/'config.json')):
        from .point_union_report import gallery as gallery_points
        return gallery_points(root, experiment)
    dest = root/'comparison'
    dest.mkdir(exist_ok=True)
    baseline = Path(read(experiment/'protocol.json')['baseline'])
    diagnostics = {r['query_id']: r for r in read_jsonl(root/'evaluation/diagnostics.jsonl')}
    rows = {r['sample_id']: r for r in read_jsonl(root/'evaluation_manifest.jsonl')}
    cards, last, gt = [], None, None
    for i, q in enumerate(read_jsonl(root/'queries.jsonl')):
        sid, action = q['sample_id'], q['action']
        if sid != last:
            gt, _ = load_annotation(rows[sid]['annotation'])
            last = sid
        detail_page(root, q)
        image = Image.open(q['rgb']).convert('RGB')
        gt_path = dest/'gt'/sid/(action+'.jpg')
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        overlay(image, [gt == q['class_id']], colors=[[60,180,95]], labels=['']).save(gt_path, quality=90)
        fg, out = foreground_path(root, q), function_path(root, q)
        d = diagnostics[q['query_id']]
        panels = [('RGB', objects_path(root,sid)/'rgb.png'),
            ('旧几何前景', baseline/'candidates_test/samples'/sid/'foreground/object_overlay.png'),
            ('新 Valen 前景', fg/'overlay.jpg'), ('目标 GT', gt_path),
            (f'旧功能预测 · IoU {d["old_iou"]:.2%}', baseline/'methods/centroid/samples'/sid/'queries'/action/'overlay.jpg'),
            (f'新功能预测 · IoU {d["new_iou"]:.2%}', out/'overlay.jpg')]
        if d['part_path']:
            parts = root/d['part_path']
            if (parts/'regions/numbered_crop.png').exists():
                panels.append(('新部件候选', parts/'regions/numbered_crop.png'))
        cards.append({'id': q['query_id'], 'category': q['object_category'], 'action': action,
            'reason': d['foreground_reason'], 'details': relative(out/'index.html',dest),
            'panels': [[label, relative(path,dest)] for label,path in panels]})
        if (i+1)%500 == 0:
            print(f'GALLERY {i+1}', flush=True)
    (dest/'data.js').write_text('window.DATA='+json.dumps(cards, ensure_ascii=False)+';', encoding='utf-8')
    (dest/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Valen 前景选择对照</title><style>'+STYLE+'</style>'+VIEWER, encoding='utf-8')


VIEWER = '''<h1>Valen 前景选择 → 部件质心判断</h1><select id="category"></select><select id="action"></select><select id="reason"></select><button id="prev">上一条</button><button id="next">下一条</button><input id="pos" type="number" value="1" min="1" style="width:80px"><span id="count"></span><h2 id="title"></h2><a id="details">逐步输入、prompt、判断与计时</a><div id="panels" class="grid"></div><script src="data.js"></script><script>
const $=id=>document.getElementById(id);let list=window.DATA,index=0;
for(const [key,label] of [['category','全部物体'],['action','全部操作'],['reason','全部前景结果']]){const el=$(key);el.add(new Option(label,''));for(const x of [...new Set(DATA.map(q=>q[key]))].sort())el.add(new Option(x,x));el.onchange=()=>{list=DATA.filter(q=>['category','action','reason'].every(k=>!$(k).value||q[k]===$(k).value));index=0;show()};}
function show(){$('panels').replaceChildren();$('count').textContent=' / '+list.length;$('pos').value=index+1;const q=list[index];if(!q)return;$('title').textContent=q.id+' · '+q.reason;$('details').href=q.details;for(const [label,path] of q.panels){const a=document.createElement('article'),t=document.createElement('div'),im=document.createElement('img');t.textContent=label;im.src=path;im.loading='lazy';a.append(t,im);$('panels').append(a)}}
$('prev').onclick=()=>{index=Math.max(0,index-1);show()};$('next').onclick=()=>{index=Math.min(list.length-1,index+1);show()};$('pos').onchange=()=>{index=Math.max(0,Math.min(list.length-1,Number($('pos').value)-1));show()};show();</script></html>'''


def update_document(root, result):
    from .point_union import enabled
    if enabled(read(Path(root)/'config.json')):
        # Its separate report already records the actual sample scope. Never
        # replace the historical outline experiment's marked document block.
        return
    old, new = result['baseline_metrics'], result['metrics']
    delta = 100*(new['mIoU_known_operations']-old['mIoU_known_operations'])
    relative_root = '../../outputs/'+Path(root).name+'/full'
    lines = ['<!-- valen-foreground-start -->', '### Valen 选择前景：通用词表＋质心判断', '',
        '**2,661 张 RGB／5,232 个功能对；原始权重、无 UMD 训练。** 前景阶段新增物体名＋操作的逐候选 yes/no 判断；选一个完整物体后重新分割部件。功能判断沿用质心点方案。', '',
        '| 前景选择方式 | mIoU ↑ | gIoU ↑ | P@50 ↑ | Dice ↑ |', '|---|---:|---:|---:|---:|']
    for name, m in [('几何排序／原对照', old), ('Valen 判断／新实验', new)]:
        lines.append('| '+name+' | '+' | '.join(f'{m[k]:.2%}' for k in ['mIoU_known_operations','gIoU','P_at_50','Dice'])+' |')
    lines += ['', f'mIoU 变化 **{delta:+.2f} pp**；前景功能像素覆盖 **{result["baseline_foreground_target_pixel_recall"]:.2%} → {result["foreground_target_pixel_recall"]:.2%}**。拒识 **{result["counts"].get("foreground_no_target",0)}/{new["queries"]}**，运行失败 **{result["counts"].get("failed_queries",0)}**；均保留计分。', '',
        '| 功能 IoU | grasp | cut | scoop | contain | pound | support | wrap-grasp |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in [('几何排序',old),('Valen 判断',new)]:
        lines.append('| '+name+' | '+' | '.join(f'{m["per_class"][k]["iou"]:.2%}' for k in ['grasp','cut','scoop','contain','pound','support','wrap-grasp'])+' |')
    t=result['timing']['mean_per_query']
    lines += ['', f'每功能对摊销：前景 Valen **{t["foreground_valen_seconds"]:.3f} s**，功能 Valen **{t["functional_valen_seconds"]:.3f} s**，各阶段合计 **{t["pipeline_components_seconds_amortized"]:.3f} s**。离线并行、共享重复计算；含 Valen 输入保存，不作为实时单次延迟。', '',
        f'[完整可视化]({relative_root}/comparison/index.html) · [指标与诊断]({relative_root}/evaluation/metrics.json)', '<!-- valen-foreground-end -->']
    doc=PROJECT/'docs/research/实验结果.md'
    text=doc.read_text(encoding='utf-8'); block='\n'.join(lines)
    begin,end='<!-- valen-foreground-start -->','<!-- valen-foreground-end -->'
    if begin in text:
        a=text.index(begin); b=text.index(end,a)+len(end); text=text[:a]+block+text[b:]
    else:
        heading='## 前景选择方法对照\n'
        assert text.count(heading)==1
        text=text.replace(heading,heading+'\n'+block+'\n',1)
    doc.write_text(text,encoding='utf-8')
