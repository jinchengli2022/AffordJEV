"""All-sample foreground inspection and GT-only evaluation after inference."""
from collections import Counter
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ...common import read, read_jsonl, write, write_jsonl, sha, load_mask
from ...datasets.umd import load_annotation
from ...paths import PROJECT
from ...visualization import overlay, font
from ...baselines.affordance_r1.compare import query_score, summarize_queries
from .workers import objects_path, foreground_path, function_path
from .point_union import choose, request, chinese_prompt, CATEGORIES, ACTIONS
from .report import relative


METHOD_NAMES = {'geometric': '几何前景／旧方案', 'outline': '描边判断／最高分单 mask',
                'point_union': '内部中心点／互斥区域并集'}


def comparison_roots(experiment):
    cfg = read(experiment/'config.json')
    baseline = Path(read(experiment/'protocol.json')['baseline'])
    outline = Path(cfg['comparisons']['outline_run'])
    if not outline.is_absolute():
        outline = PROJECT/outline
    return baseline, outline


def point_reference(experiment):
    value = read(experiment/'config.json').get('comparisons', {}).get('point_union_run')
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT/path


def foreground_statistics(mask, gt, class_id):
    valid, target = gt != -1, gt == class_id
    return {'target_pixels': int(target.sum()), 'covered_target_pixels': int((mask & target).sum()),
            'foreground_valid_pixels': int((mask & valid).sum()),
            'foreground_gt_background_pixels': int((mask & (gt == 0)).sum())}


def evaluate(root, experiment):
    root, experiment = Path(root), Path(experiment)
    qs = read_jsonl(root/'queries.jsonl')
    rows = {r['sample_id']: r for r in read_jsonl(root/'evaluation_manifest.jsonl')}
    protocol = read(root/'protocol.json')
    assert sha(root/'queries.jsonl') == protocol['queries_sha256']
    assert sha(root/'evaluation_manifest.jsonl') == protocol['manifest_sha256']
    baseline, outline = comparison_roots(experiment)
    threshold = read(root/'config.json')['foreground_selector']['minimum_yes_probability']
    reference = point_reference(experiment)
    names = {k: v for k, v in METHOD_NAMES.items() if k != 'point_union'}
    if reference is not None:
        assert read_jsonl(reference/'queries.jsonl') == qs
        assert sha(reference/'evaluation_manifest.jsonl') == sha(root/'evaluation_manifest.jsonl')
        old_threshold = read(reference/'config.json')['foreground_selector']['minimum_yes_probability']
        names['point_union_reference'] = f'内部中心点／互斥区域并集（P(yes)>{old_threshold:.0%}）'
    names['point_union'] = f'内部中心点／互斥区域并集（P(yes)>{threshold:.0%}）'
    scores = {k: [] for k in names}
    totals = {k: Counter() for k in names}
    details, image_details, timing, counts = [], [], Counter(), Counter()
    part_uses = Counter(read(foreground_path(root, q)/'parts.json')['path'] for q in qs)
    images, candidates_by_image = {}, {}
    audited_calls = 0
    for sid, row in rows.items():
        assert sha(row['annotation']) == row['annotation_sha256']
        gt, kind = load_annotation(row['annotation'], row['size_wh'][::-1])
        assert kind == 'manual'
        images[sid] = gt
        obj = objects_path(root, sid)
        assert np.array_equal(np.asarray(Image.open(row['rgb']).convert('RGB')),
                              np.asarray(Image.open(obj/'rgb.png').convert('RGB')))
        cs = read(obj/'preprocessed/candidates.json')['candidates']
        labels = np.load(obj/'preprocessed/label_map.npy', allow_pickle=False)
        reconstructed = np.zeros_like(labels)
        union = np.zeros(gt.shape, bool)
        for r in read(obj/'eligible.json')['candidates']:
            assert sha(obj/r['mask']) == r['mask_sha256']
            union |= load_mask(obj/r['mask'], gt.shape)
        for c in cs:
            m = load_mask(obj/c['mask'], gt.shape)
            assert sha(obj/c['mask']) == c['mask_sha256']
            assert not (m & (reconstructed > 0)).any()
            reconstructed[m] = c['id']
            x, y = c['point']['xy']
            assert m[y, x]
        assert np.array_equal(labels, reconstructed)
        assert np.array_equal(union, labels > 0)
        trace = read(obj/'preprocessed/trace.json')
        candidates_by_image[sid] = cs
        image_details.append({'sample_id': sid, 'raw_objects': len(read(obj/'raw/objects.json')['candidates']),
            'eligible_objects': trace['input_masks'], 'fragments_before_merge': trace['fragments_before_merge'],
            'regions_after_merge': len(cs), 'coverage_preserved': True, 'overlap_pixels_after': 0,
            'centers_inside': len(cs), 'centers_outside': 0})
        t = read(obj/'timing.json')
        timing['object_forward_seconds_total'] += t['object_forward_seconds']
        timing['foreground_assets_seconds_total'] += t['foreground_assets_seconds']
    for q in qs:
        sid, cid = q['sample_id'], q['class_id']
        gt, cs = images[sid], candidates_by_image[sid]
        obj, fg, out = objects_path(root, sid), foreground_path(root, q), function_path(root, q)
        selection = read(fg/'selection.json')
        state = read(out/'run.json')
        selected = load_mask(fg/'selected_mask.png', gt.shape)
        assert sha(fg/'selected_mask.png') == selection['mask_sha256']
        decisions = read(fg/'decisions.json') if (fg/'decisions.json').exists() else []
        if selection['reason'] != 'error':
            expected = choose(cs, decisions, threshold=threshold)
            assert selection.get('minimum_yes_probability', threshold) == threshold
            assert all(d['threshold'] == threshold and d['selected'] ==
                       (d['probabilities']['yes'] > threshold) for d in decisions)
            assert [c['id'] for c in expected] == [c['id'] for c in selection['selected']]
            reconstructed = np.zeros(gt.shape, bool)
            for c in expected:
                reconstructed |= load_mask(obj/c['mask'], gt.shape)
            assert np.array_equal(reconstructed, selected)
            for c in cs:
                inp = fg/f'region_{c["id"]:03}'
                for name, digest in read(inp/'prepared_assets.json').items():
                    assert sha(inp/name) == digest
                assert read(inp/'valen_request.json') == request(c['id'], q['object_category'], q['action'])
                assert (inp/'prompt_zh.md').read_text(encoding='utf-8') == chinese_prompt(q['object_category'], q['action'])
                assert sha(inp/'point.png') == c['point_image_sha256']
                assert sha(inp/'rgb.png') == sha(obj/'rgb.png')
                response = read(inp/'valen_response.json')
                assert response['request_sha256'] == sha(inp/'valen_request.json')
                assert response.get('usage', {}).get('output_tokens', 0) == 0
                audited_calls += 1
        assert sha(out/'mask.png') == state['mask_sha256']
        masks = {
            'geometric': (baseline/'candidates_test/samples'/sid/'foreground/object_mask.png',
                          baseline/'methods/centroid/samples'/sid/'queries'/q['action']/'mask.png'),
            'outline': (foreground_path(outline, q)/'selected_mask.png', function_path(outline, q)/'mask.png'),
            'point_union': (fg/'selected_mask.png', out/'mask.png')}
        if reference is not None:
            masks['point_union_reference'] = (foreground_path(reference, q)/'selected_mask.png',
                                               function_path(reference, q)/'mask.png')
        row = {'query_id': q['query_id'], 'sample_id': sid, 'action': q['action'],
               'regions': len(cs), 'selected_ids': [c['id'] for c in selection['selected']],
               'foreground_reason': selection['reason'], 'methods': {}}
        for method, (fg_file, mask_file) in masks.items():
            m, p = load_mask(fg_file, gt.shape), load_mask(mask_file, gt.shape)
            s = query_score(sid, cid, gt, p)
            scores[method].append(s)
            stat = foreground_statistics(m, gt, cid)
            totals[method].update(stat)
            row['methods'][method] = {**stat, 'foreground_target_recall': stat['covered_target_pixels']/stat['target_pixels'],
                'foreground_gt_background_fraction': stat['foreground_gt_background_pixels']/stat['foreground_valid_pixels']
                    if stat['foreground_valid_pixels'] else None, 'functional_iou': s['iou']}
        assert not (load_mask(out/'mask.png', gt.shape) & ~selected).any()
        fg_time, fn_time = read(fg/'timing.json'), read(out/'timing.json')
        timing['foreground_valen_seconds_total'] += fg_time['valen_seconds']
        timing['foreground_calls_total'] += fg_time['calls']
        timing['functional_valen_seconds_total'] += fn_time['functional_valen_seconds']
        part = read(fg/'parts.json')
        part_seconds = 0.
        if part['status'] == 'ready':
            part_seconds = read(root/part['path']/'partglee_runtime.json')['timings']['part_forward_seconds']/part_uses[part['path']]
        timing['part_forward_seconds_total'] += part_seconds
        row['timing'] = {'foreground_valen_seconds': fg_time['valen_seconds'],
                        'functional_valen_seconds': fn_time['functional_valen_seconds'],
                        'part_forward_seconds_amortized': part_seconds}
        counts['failed_queries'] += int(state['status'] == 'failed')
        counts['empty_foreground_queries'] += int(not selected.any())
        details.append(row)
    methods = {}
    for method in names:
        t = totals[method]
        methods[method] = {'name': names[method], 'metrics': summarize_queries(scores[method]),
            'foreground_target_pixel_recall': t['covered_target_pixels']/t['target_pixels'],
            'foreground_gt_background_fraction': t['foreground_gt_background_pixels']/t['foreground_valid_pixels']
                if t['foreground_valid_pixels'] else None, 'counts': dict(t)}
    assert len(qs) == protocol['queries'] and len(rows) == protocol['images']
    result = {'status': 'complete', 'phase': protocol['phase'], 'images': len(rows), 'queries': len(qs),
        'UMD_training_images': 0, 'metrics': methods['point_union']['metrics'], 'comparisons': methods,
        'foreground_minimum_yes_probability': threshold, 'foreground_threshold_comparison': 'strictly greater than',
        'functional_decision_rule': 'P(yes) > P(no); unchanged',
        'counts': dict(counts), 'regions_per_image': {r['sample_id']: r['regions_after_merge'] for r in image_details},
        'timing': {**dict(timing), 'foreground_valen_seconds_per_query': timing['foreground_valen_seconds_total']/len(qs),
            'foreground_valen_seconds_per_region': timing['foreground_valen_seconds_total']/max(1, timing['foreground_calls_total']),
            'note': 'Serial GPU stages; includes Valen preprocessing/input audit, excludes model loads and most artifact I/O. Not live end-to-end latency.'},
        'metric_note': 'Known-operation metrics on this exact sample subset. Foreground background fraction uses GT==0, '
                       'not a separately annotated complete-object mask; void ignored. No all-test claims.',
        'gt_in_model_inputs': False, 'queries_sha256': sha(root/'queries.jsonl')}
    write(root/'evaluation/metrics.json', result)
    write_jsonl(root/'evaluation/per_query.jsonl', scores['point_union'])
    write_jsonl(root/'evaluation/diagnostics.jsonl', details)
    write_jsonl(root/'evaluation/preprocessing.jsonl', image_details)
    for method in names:
        write_jsonl(root/'evaluation'/method/'per_query.jsonl', scores[method])
    write(root/'evaluation/input_audit.json', {'images': len(rows), 'queries': len(qs),
        'point_requests_audited': audited_calls, 'all_regions_disjoint': True, 'coverage_preserved': True,
        'all_points_inside': True, 'all_selected_unions_verified': True, 'actual_inputs_and_prompts_verified': True,
        'foreground_threshold_verified': threshold,
        'source_files_unchanged': all(sha(PROJECT/p) == digest for p, digest in read(root/'source_hashes.json').items()),
        'reference_annotations_used_by_models': False})
    return result


STYLE = '''body{font:16px system-ui,sans-serif;color:#222;background:#fff;margin:28px auto;padding:0 20px;max-width:1500px}h1{font-size:26px}h2{font-size:21px;margin-top:32px}h3{font-size:17px}a{color:#165bad}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}figure{margin:0;padding:10px;border:1px solid #ddd}img{width:100%;height:auto;display:block}figcaption{margin-bottom:8px}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{text-align:left;border-bottom:1px solid #ddd;padding:9px}pre{white-space:pre-wrap;line-height:1.65;background:#f7f7f7;padding:16px}summary{cursor:pointer;padding:12px 0}select{font:inherit;padding:7px}details{border-bottom:1px solid #ddd}.muted{color:#666;font-size:14px}code{background:#f3f3f3}'''


def page(title, body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+'</style>'+body+'</html>'


def figure(folder, label, image):
    url = html.escape(relative(image, folder), quote=True)
    return f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{url}"><img loading="lazy" src="{url}"></a></figure>'


def anchor(folder, text, path):
    return f'<a href="{html.escape(relative(path,folder),quote=True)}">{html.escape(text)}</a>'


def render_partition_overview(obj):
    """Display IDs at the audited interior points; never used as model input."""
    rgb = Image.open(obj/'rgb.png').convert('RGB')
    cs = read(obj/'preprocessed/candidates.json')['candidates']
    im = overlay(rgb, [load_mask(obj/c['mask']) for c in cs], labels=['']*len(cs), fill_alpha=.50)
    draw = ImageDraw.Draw(im)
    for c in cs:
        x, y = c['point']['xy']
        x, y = max(14, min(rgb.width-14, x)), max(12, min(rgb.height-12, y))
        draw.text((x, y), str(c['id']), anchor='mm', font=font(18), fill='white', stroke_width=2, stroke_fill='black')
    im.save(obj/'preprocessed/overview.jpg', quality=95)


def gallery(root, experiment):
    root, experiment = Path(root), Path(experiment)
    qs, result = read_jsonl(root/'queries.jsonl'), read(root/'evaluation/metrics.json')
    diagnostics = {r['query_id']: r for r in read_jsonl(root/'evaluation/diagnostics.jsonl')}
    prep = {r['sample_id']: r for r in read_jsonl(root/'evaluation/preprocessing.jsonl')}
    baseline, outline = comparison_roots(experiment)
    reference = point_reference(experiment)
    threshold = result['foreground_minimum_yes_probability']
    dest = root/'comparison'
    dest.mkdir(exist_ok=True)
    for sid in dict.fromkeys(q['sample_id'] for q in qs):
        render_partition_overview(objects_path(root, sid))
    cards, pages, point_count = [], [], 0
    for q in qs:
        sid, action = q['sample_id'], q['action']
        obj, fg, out = objects_path(root, sid), foreground_path(root, q), function_path(root, q)
        title = f'{sid} · {ACTIONS.get(action,action)}（{action}）'
        diag = diagnostics[q['query_id']]
        body = f'<h1>{html.escape(title)}</h1><p>'+anchor(fg, '返回全部样例', dest/'index.html')+'</p>'
        body += '<h2>1. PartGLEE 原始 object-level 分割</h2><div class="grid">'
        body += figure(fg, '完整 RGB', obj/'rgb.png')+figure(fg, '50 个原始候选色块总览（允许重叠）', obj/'raw/overview.jpg')+'</div>'
        body += '<p class="muted">原始色块按 raw ID 顺序半透明叠加；遮挡不代表候选互斥。下方可查看每一个原始 mask。</p>'
        raw = read(obj/'raw/objects.json')['candidates']
        selector = []
        for c in raw:
            preview = obj/'raw/individual'/f'{c["raw_id"]:03}.jpg'
            if not preview.exists():
                preview.parent.mkdir(exist_ok=True)
                overlay(Image.open(obj/'rgb.png').convert('RGB'), [load_mask(obj/c['mask'])],
                        labels=[str(c['raw_id'])]).save(preview, quality=90)
            selector.append({'id': c['raw_id'], 'area': c['area'], 'quality': c['quality'],
                             'image': relative(preview, fg), 'mask': relative(obj/c['mask'], fg)})
        body += '<details><summary>逐个检查全部 50 个原始 mask</summary><select id="raw-select"></select><p id="raw-info"></p><div class="grid"><figure><img id="raw-image"></figure><figure><img id="raw-mask"></figure></div></details>'
        p = prep[sid]
        body += '<h2>2. 互斥预处理结果</h2><p>'+f'原始 50 → 合格 {p["eligible_objects"]} → 连通碎片 {p["fragments_before_merge"]} → 合并后 {p["regions_after_merge"]}。覆盖像素保留，重叠像素为 0。'+'</p><div class="grid">'
        body += figure(fg, '互斥区域色块与编号（512 像素合并阈值）', obj/'preprocessed/overview.jpg')+'</div><p>'
        body += anchor(fg, '合并记录', obj/'preprocessed/trace.json')+' · '+anchor(fg, '区域与中心点', obj/'preprocessed/candidates.json')+'</p>'
        body += '<h2>3. Valen 实际输入与中文 prompt</h2><p>每个区域独立输入原始 RGB＋单个内部中心点；以下图片即模型实际读取的文件。点击区域展开查看。</p>'
        body += f'<p>前景接收条件：P(yes) &gt; {threshold:.0%}，等于阈值也不接收。未通过阈值不代表模型高置信判为背景；后续功能判断规则不变。</p>'
        decisions = read(fg/'decisions.json')
        body += '<table><tr><th>区域</th><th>面积／像素</th><th>中心点</th><th>P(yes)</th><th>阈值接收</th></tr>'
        for c in decisions:
            yes = c['id'] in diag['selected_ids']
            body += f'<tr><td><a href="#region-{c["id"]}">{c["id"]}</a></td><td>{c["area"]}</td><td>{c["point"]["xy"]}</td><td>{c["probabilities"]["yes"]:.4f}</td><td>{"接收为前景" if yes else "不接收"}</td></tr>'
        body += '</table>'
        for c in decisions:
            folder = fg/f'region_{c["id"]:03}'
            yes = c['id'] in diag['selected_ids']
            body += f'<details id="region-{c["id"]}"><summary>区域 {c["id"]} · {"接收为前景" if yes else "不接收"} · P(yes)={c["probabilities"]["yes"]:.4f}</summary><div class="grid">'
            body += figure(fg, '实际输入 1：原始 RGB', folder/'rgb.png')
            body += figure(fg, '实际输入 2：RGB＋一个内部中心点', folder/'point.png')+'</div>'
            body += '<pre>'+html.escape((folder/'prompt_zh.md').read_text(encoding='utf-8'))+'</pre><p>'
            body += ' · '.join(anchor(fg, label, folder/name) for label,name in [
                ('原始英文请求','valen_request.json'), ('中文翻译','prompt_zh.md'), ('模型回答','valen_response.json'),
                ('实际编译 prompt','valen_compiled/prompt.txt'), ('计时','valen_runtime.json')])+'</p></details>'
            point_count += 1
        body += '<h2>4. Valen 选择的最终前景</h2><p>接受的区域编号：'+html.escape(str(diag['selected_ids']))+'</p><div class="grid">'
        body += figure(fg, '被接受的互斥区域', fg/'accepted_regions.jpg')
        body += figure(fg, '前景并集叠加 RGB', fg/'overlay.jpg')
        body += figure(fg, '前景二值 mask', fg/'selected_mask.png')+'</div><p>'
        body += anchor(fg, '完整判断记录', fg/'decisions.json')+' · '+anchor(fg, '并集来源', fg/'selection.json')+'</p>'
        body += '<details><summary>同样例旧前景对照与后续功能预测</summary><div class="grid">'
        body += figure(fg, '旧几何前景', baseline/'candidates_test/samples'/sid/'foreground/object_overlay.png')
        body += figure(fg, '旧描边判断前景', foreground_path(outline,q)/'overlay.jpg')
        if reference is not None:
            old_threshold = read(reference/'config.json')['foreground_selector']['minimum_yes_probability']
            body += figure(fg, f'原内部中心点前景（阈值 {old_threshold:.0%}）', foreground_path(reference,q)/'overlay.jpg')
            body += figure(fg, '原内部中心点功能预测', function_path(reference,q)/'overlay.jpg')
        body += figure(fg, '新前景下的功能预测（原后续参数）', out/'overlay.jpg')+'</div></details>'
        body += '<script>const RAW='+json.dumps(selector)+';const s=document.getElementById("raw-select");RAW.forEach((r,i)=>s.add(new Option("raw "+r.id,i)));function show(){const r=RAW[s.value||0];document.getElementById("raw-image").src=r.image;document.getElementById("raw-mask").src=r.mask;document.getElementById("raw-info").textContent="面积 "+r.area+"；质量分 "+r.quality.toFixed(4);}s.onchange=show;show();document.querySelectorAll("a").forEach(a=>{const href=a.getAttribute("href")||"";if(href.startsWith("#region-"))a.onclick=()=>{document.querySelector(href).open=true;};});</script>'
        (fg/'index.html').write_text(page(title,body), encoding='utf-8')
        pages.append(fg/'index.html')
        cards.append({'query_id':q['query_id'], 'sample_id':sid, 'action':action, 'title':title,
            'page':relative(fg/'index.html',dest), 'rgb':relative(obj/'rgb.png',dest),
            'raw':relative(obj/'raw/overview.jpg',dest), 'preprocessed':relative(obj/'preprocessed/overview.jpg',dest),
            'foreground':relative(fg/'overlay.jpg',dest), 'regions':len(decisions), 'accepted':len(diag['selected_ids'])})
    body = f'<h1>互斥候选＋内部中心点：前景检查</h1><p>全部 {result["images"]} 张 RGB／{result["queries"]} 个功能对。原始权重，无训练；512 像素碎片合并。每条均可查看全部输入和中文 prompt。</p>'
    body += f'<p>本轮仅调整前景接收阈值：P(yes) &gt; {threshold:.0%}；功能判断仍使用原 yes/no 规则。</p>'
    scope = '这是样例验证，不代表完整测试集结果。' if result['phase'] == 'smoke' else '完整 novel/test 人工帧评测；阈值此前在九张样例上开发。'
    body += f'<p class="muted">{scope}图中色块仅供人工检查，前景 Valen 实际只读取 RGB 和单点图。</p>'
    body += '<table><tr><th>同样例方案</th><th>前景功能像素覆盖</th><th>前景内 GT 背景占比</th><th>最终功能 mIoU</th><th>gIoU</th></tr>'
    for key, m in result['comparisons'].items():
        bg = m['foreground_gt_background_fraction']
        body += f'<tr><td>{m["name"]}</td><td>{m["foreground_target_pixel_recall"]:.2%}</td><td>{format(bg,".2%") if bg is not None else "—"}</td><td>{m["metrics"]["mIoU_known_operations"]:.2%}</td><td>{m["metrics"]["gIoU"]:.2%}</td></tr>'
    body += '</table><p class="muted">背景占比依据 UMD 的 GT=0，忽略 void；不是额外标注的完整物体纯度。</p>'
    for sid in dict.fromkeys(q['sample_id'] for q in qs):
        related = [c for c in cards if c['sample_id']==sid]
        first = related[0]
        body += '<h2>'+html.escape(sid)+'</h2><div class="grid">'
        for label,key in [('RGB','rgb'),('1. 原始重叠分割','raw'),('2. 预处理互斥区域','preprocessed')]:
            body += figure(dest,label,dest/first[key])
        body += '</div><div class="grid">'
        for c in related:
            label = f'{c["action"]}：接受 {c["accepted"]}/{c["regions"]} 个区域'
            body += '<figure><figcaption>'+anchor(dest,label,dest/c['page'])+'</figcaption><a href="'+html.escape(c['page'],quote=True)+'"><img loading="lazy" src="'+html.escape(c['foreground'],quote=True)+'"></a></figure>'
        body += '</div>'
    body += '<p>'+anchor(dest,'完整指标',root/'evaluation/metrics.json')+' · '+anchor(dest,'输入与并集核验',root/'evaluation/input_audit.json')+'</p>'
    (dest/'index.html').write_text(page('前景检查：全部样例',body),encoding='utf-8')
    write(dest/'samples.json',cards)
    write(root/'evaluation/gallery_audit.json', {'images':len({c['sample_id'] for c in cards}),
        'query_pages':len(pages), 'point_inputs_shown':point_count, 'all_queries_included':len(cards)==len(qs),
        'all_raw_candidates_inspectable':True, 'stages': ['raw objects','exclusive regions','actual inputs and Chinese prompt','foreground union']})
    write_readme(root, result)


def write_readme(root, result):
    lines = [f'# 内部中心点前景判断：{result["images"]} 张图', '',
        f'**{result["images"]} 张 RGB／{result["queries"]} 个功能对。** 通用 object / part；仅 object-level 候选，互斥分配后合并不足 512 像素的连通碎片。', '',
        f'RGB＋单个区域内部中心点 → Valen 判断目标归属 → P(yes)>{result["foreground_minimum_yes_probability"]:.0%} 的区域取并集。后续部件与功能判断参数沿用原方案。', '',
        '| 同样例方案 | 前景功能像素覆盖 ↑ | 前景内 GT 背景占比 ↓ | 最终功能 mIoU ↑ | gIoU ↑ |',
        '|---|---:|---:|---:|---:|']
    for m in result['comparisons'].values():
        bg=m['foreground_gt_background_fraction']
        lines.append(f'| {m["name"]} | {m["foreground_target_pixel_recall"]:.2%} | {format(bg,".2%") if bg is not None else "—"} | {m["metrics"]["mIoU_known_operations"]:.2%} | {m["metrics"]["gIoU"]:.2%} |')
    lines += ['', '背景占比按 GT=0 统计、void 忽略，不是独立完整物体标注。两项前景指标均按同一组功能对汇总。', '',
        '[全部样例与中间步骤](comparison/index.html) · [指标](evaluation/metrics.json) · [输入核验](evaluation/input_audit.json)', '',
        '| RGB | 互斥候选数 |', '|---|---:|']
    lines += [f'| {sid} | {n} |' for sid,n in result['regions_per_image'].items()]
    t=result['timing']
    lines += ['', f'前景 Valen：{t["foreground_valen_seconds_per_region"]:.3f} 秒／区域，{t["foreground_valen_seconds_per_query"]:.3f} 秒／功能对；包含输入保存，不含模型加载，不能当作实时端到端延迟。', '',
        '原始候选色块总览允许重叠；互斥结果只用于本轮单点判断。历史输出未改写。']
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
