"""Add the historical full-vocabulary geometry baseline to a fixed sample comparison.

PartGLEE candidates are reused unchanged. Run conditional_umd.infer separately to
obtain fresh centroid yes/no decisions; annotations are read only by report().
"""
import argparse
from collections import Counter
import html
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ...common import read, read_jsonl, write, write_jsonl, sha, load_mask
from ...paths import PROJECT
from ...segmentation.foreground import select_object
from ...visualization import overlay
from ..conditional_umd.requests import request_for, decode
from .point_union_report import foreground_statistics, page, figure, anchor


NAMES = {
    'specific_geometric': '完整物体／部件词表＋几何筛选',
    'geometric': '通用 object / part＋几何筛选',
    'outline': '通用 object / part＋描边判断／最高分单 mask',
    'point_union': '通用 object / part＋内部中心点／互斥区域并集',
}
DEFINITIONS_ZH = {
    'grasp': '用于握持和控制物体的专用手柄、握把或手指抓持区域。优先选择预期握持位置，而非任意可触摸的工作表面或宽大的容器外壁。',
    'cut': '用于切开材料的锋利刀片、刃口或切割工作面，不包括手柄及非切割支撑结构。',
    'scoop': '通过舀取动作收集并提起松散或柔软材料的凹形工作面。区别于深的储存腔体和较平的承托表面。',
    'contain': '主要用于保存内容物的可见内表面或腔体。不包括独立手柄及用于环抱抓握的外表面。',
    'pound': '向其他材料施加冲击力的实心锤头或敲击工作面。不包括手柄和柄杆。',
    'support': '从下方支撑或搬运材料的较平工作面。区别于凹形舀取面、储存腔体和手柄。',
    'wrap-grasp': '用手指和手掌环绕握持的物体外表面。区别于独立的专用手柄和可见的容纳内腔。',
}


def prompt_zh(query, region_id):
    from .point_union import CATEGORIES, ACTIONS
    category, action = query['object_category'], query['action']
    return (
        f'目标物体：{CATEGORIES.get(category, category)}（{category}）；目标操作：{ACTIONS.get(action, action)}（{action}）。\n'
        f'功能定义：{DEFINITIONS_ZH[action]}\n'
        '判断指定候选的主要功能是否符合目标操作。第一张图为未填色的 RGB 裁剪图；第二张图为同一裁剪图上的编号候选。'
        '颜色和编号是标注，不是物体外观，请检查未填色的 RGB。\n'
        '按给定 UMD 功能定义判断可见像素。区分专用手柄和环抱握持的外壁；区分凹形收集面、储存腔体和平面承托面。'
        '不要仅因为一个区域能被触碰或握住就接受。符合定义的区域可以覆盖大部分或整个物体，不限制整物体面积。'
        '背景或其他物体不匹配；功能不相容的混合区域应拒绝。\n'
        f'第三张图为未填色的 RGB 裁剪图加一个红点，位置是候选 {region_id} 的算术质心。判断该点所在的可见功能部位，'
        '忽略标记颜色；该点没有投影回 mask 内部。\n'
        f'问题：红点所在的可见功能部位是否属于目标物体并符合 {action}？选择 yes 或 no。\n'
        'yes：符合目标功能定义和规则。no：不符合目标功能，或属于背景／其他物体；它可以具有其他功能，no 不代表没有任何功能。\n'
    )


def prepare(root, reference, candidates):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists():
        raise FileExistsError('Use a new output directory; preparation never replaces an existing run')
    qs = read_jsonl(reference/'queries.jsonl')
    assert len(qs) == 18 and len({q['sample_id'] for q in qs}) == 9
    generic = PROJECT/'outputs/umd_given_operation_20261018'
    cfg = read(candidates/'samples'/qs[0]['sample_id']/'config_snapshot.json')
    current = read(generic/'pipeline.json')
    assert cfg['foreground'] == current['foreground'] and cfg['partition'] == current['partition']
    assert len(cfg['vocabulary']['objects']) == 17 and len(cfg['vocabulary']['parts']) == 18
    cfg['valen'] = current['valen']
    assert cfg['valen'] == read(reference.parent/'config.json')['valen']
    write(root/'pipeline.json', cfg)
    shutil.copyfile(generic/'definitions.json', root/'definitions.json')
    shutil.copyfile(reference/'queries.jsonl', root/'test_queries.jsonl')
    shutil.copyfile(reference/'evaluation_manifest.jsonl', root/'test_manifest.jsonl')
    audit, hashes = [], {}
    for sid in dict.fromkeys(q['sample_id'] for q in qs):
        src = candidates/'samples'/sid
        q = next(q for q in qs if q['sample_id'] == sid)
        original = read(src/'config_snapshot.json')
        assert original['vocabulary'] == cfg['vocabulary']
        assert original['foreground'] == cfg['foreground'] and original['partition'] == cfg['partition']
        assert read(src/'run.json')['reference_annotations_used'] is False
        assert sha(q['rgb']) == q['rgb_sha256'] == read(src/'job.json')['source_sha256']
        assert np.array_equal(np.asarray(Image.open(q['rgb']).convert('RGB')),
                              np.asarray(Image.open(src/'rgb.png').convert('RGB')))
        raw = read(src/'raw/objects.json')['candidates']
        masks = np.stack([load_mask(src/r['mask']) for r in raw])
        selected, _ = select_object(masks, [r['quality'] for r in raw], cfg['foreground'])
        assert selected == read(src/'foreground/metadata.json')['selected_raw_id']
        assert np.array_equal(masks[selected], load_mask(src/'foreground/object_mask.png'))
        out = root/'candidates_test/samples'/sid
        out.mkdir(parents=True)
        # Expose candidate inputs only, never the historic functional predictions.
        for name in ['rgb.png', 'foreground', 'regions', 'raw', 'candidates.json', 'config_snapshot.json']:
            (out/name).symlink_to(src/name)
        write(out/'run.json', {'status': 'prepared', 'reference_annotations_used': False})
        used = ['rgb.png', 'candidates.json', 'config_snapshot.json', 'raw/objects.json',
                'foreground/object_mask.png', 'foreground/metadata.json', 'foreground/rgb_crop.png',
                'regions/numbered_crop.png', 'regions/region_sheet.jpg']
        cs = read(src/'candidates.json')['candidates']
        used.extend(c['mask'] for c in cs)
        hashes.update({str(src/name): sha(src/name) for name in used})
        audit.append({'sample_id': sid, 'source': str(src), 'raw_objects': len(raw),
                      'selected_raw_id': selected, 'candidates': len(cs), 'geometry_reproduced': True})
    write(root/'candidate_audit.json', audit)
    write(root/'candidate_source_hashes.json', hashes)
    write(root/'source_hashes.json', {str(p.relative_to(PROJECT)): sha(p)
          for p in (PROJECT/'src/affordance_jev').rglob('*.py')})
    write(root/'protocol.json', {
        'reference': str(reference), 'historical_candidates': str(candidates),
        'images': 9, 'queries': 18, 'UMD_training_images': 0,
        'queries_sha256': sha(root/'test_queries.jsonl'), 'manifest_sha256': sha(root/'test_manifest.jsonl'),
        'historical_candidates_reused': True, 'functional_predictions_rerun': True,
        'functional_method': 'centroid; conditional yes/no; original Valen; same as the other three rows',
        'source_difference': 'Full 17-object/18-part vocabulary, geometric foreground and resulting part candidates',
        'gt_used_for_model_inputs': False,
        'reference_metrics_sha256': sha(reference/'evaluation/metrics.json'),
    })
    print('Prepared historical candidates for 9 images / 18 fresh functional queries', flush=True)


def method_files(root, reference, q):
    sid, action = q['sample_id'], q['action']
    generic = PROJECT/'outputs/umd_given_operation_20261018'
    outline = PROJECT/'outputs/umd_valen_foreground_20261018/full'
    return {
        'specific_geometric': (root/'candidates_test/samples'/sid/'foreground/object_mask.png',
                               root/'methods/centroid/samples'/sid/'queries'/action/'mask.png'),
        'geometric': (generic/'candidates_test/samples'/sid/'foreground/object_mask.png',
                      generic/'methods/centroid/samples'/sid/'queries'/action/'mask.png'),
        'outline': (outline/'foreground/samples'/sid/'queries'/action/'selected_mask.png',
                    outline/'methods/centroid/samples'/sid/'queries'/action/'mask.png'),
        'point_union': (reference/'foreground/samples'/sid/'queries'/action/'selected_mask.png',
                        reference/'methods/centroid/samples'/sid/'queries'/action/'mask.png'),
    }


def report(root):
    from ...datasets.umd import load_annotation
    from ...baselines.affordance_r1.compare import query_score, summarize_queries
    protocol = read(root/'protocol.json')
    reference = Path(protocol['reference'])
    assert sha(reference/'evaluation/metrics.json') == protocol['reference_metrics_sha256']
    assert sha(root/'test_queries.jsonl') == sha(reference/'queries.jsonl') == protocol['queries_sha256']
    assert sha(root/'test_manifest.jsonl') == protocol['manifest_sha256']
    assert all(sha(path) == digest for path, digest in read(root/'candidate_source_hashes.json').items())
    definitions, qs = read(root/'definitions.json'), read_jsonl(root/'test_queries.jsonl')
    rows = {r['sample_id']: r for r in read_jsonl(root/'test_manifest.jsonl')}
    metrics, details, totals = {k: [] for k in NAMES}, [], {k: Counter() for k in NAMES}
    calls, failures = 0, 0
    pages, cards = [], []
    dest = root/'comparison'
    for q in qs:
        sid, action = q['sample_id'], q['action']
        src = root/'candidates_test/samples'/sid
        out = root/'methods/centroid/samples'/sid/'queries'/action
        cs = read(src/'candidates.json')['candidates']
        state = read(out/'run.json')
        assert state['status'] in ['complete', 'failed'] and read(out/'query.json') == q
        assert sha(out/'mask.png') == state['mask_sha256']
        failures += state['status'] == 'failed'
        requests = read(out/'requests.json')
        decisions, reconstructed = [], np.zeros(tuple(q['size_wh'][::-1]), bool)
        if state['status'] == 'complete':
            for r in requests:
                folder = Path(r['folder'])
                assert read(folder/'valen_request.json') == request_for(r['candidates'], definitions,
                    q['object_category'], action, point=True)
                assert all(sha(folder/name) == digest for name, digest in read(folder/'prepared_assets.json').items())
                response = read(folder/'valen_response.json')
                assert response['request_sha256'] == sha(folder/'valen_request.json')
                assert response['usage']['output_tokens'] == 0
                decisions.extend(decode(response, r['candidates']))
                (folder/'prompt_zh.md').write_text(prompt_zh(q, r['candidates'][0]['id']), encoding='utf-8')
                calls += 1
            assert decisions == read(out/'decisions.json')
            for c, decision in zip(cs, decisions):
                assert c['id'] == decision['region_id']
                if decision['selected']:
                    reconstructed |= load_mask(src/c['mask'])
            assert np.array_equal(reconstructed, load_mask(out/'mask.png'))
        row = rows[sid]
        assert sha(row['annotation']) == row['annotation_sha256']
        gt, kind = load_annotation(row['annotation'], q['size_wh'][::-1])
        assert kind == 'manual'
        files = method_files(root, reference, q)
        diagnostic = {'query_id': q['query_id'], 'methods': {}}
        folder = dest/'samples'/sid/action
        folder.mkdir(parents=True, exist_ok=True)
        rgb = Image.open(q['rgb']).convert('RGB')
        overlay(rgb, [gt == q['class_id']], colors=[[60,180,95]], labels=['']).save(folder/'gt.jpg', quality=92)
        title = f'{sid} · {action}'
        body = f'<h1>{html.escape(title)}</h1><p>'+anchor(folder, '全部样例', dest/'index.html')+'</p>'
        body += '<h2>RGB 与目标功能 GT</h2><div class="grid">'
        body += figure(folder, 'RGB', q['rgb'])+figure(folder, 'GT（仅用于评分及展示）', folder/'gt.jpg')+'</div>'
        for key, (fg_file, mask_file) in files.items():
            fg, prediction = load_mask(fg_file), load_mask(mask_file)
            score = query_score(sid, q['class_id'], gt, prediction)
            metrics[key].append(score)
            stat = foreground_statistics(fg, gt, q['class_id'])
            totals[key].update(stat)
            diagnostic['methods'][key] = {**stat, 'functional_iou': score['iou']}
            overlay(rgb, [fg], labels=['']).save(folder/f'{key}_foreground.jpg', quality=92)
            overlay(rgb, [prediction], colors=[[60,180,95]], labels=['']).save(folder/f'{key}_prediction.jpg', quality=92)
            body += f'<h2>{NAMES[key]} · 功能 IoU {score["iou"]:.2%}</h2><div class="grid">'
            body += figure(folder, '前景', folder/f'{key}_foreground.jpg')
            body += figure(folder, '最终功能预测', folder/f'{key}_prediction.jpg')+'</div>'
        body += '<h2>完整词表 baseline：分割中间结果</h2><div class="grid">'
        body += figure(folder, '原始 object-level 候选', src/'raw/object_candidates.jpg')
        body += figure(folder, '几何排序选出的前景', src/'foreground/object_overlay.png')
        body += figure(folder, '最终互斥部件候选', src/'regions/numbered_crop.png')+'</div><p>'
        body += anchor(folder, '完整词表与参数', root/'pipeline.json')+' · '
        body += anchor(folder, '几何排序得分', src/'foreground/selection_trace.json')+'</p>'
        body += '<h2>本次 Valen 功能判断：实际输入与中文 prompt</h2>'
        for r, decision in zip(requests, decisions):
            inp = Path(r['folder'])
            body += f'<details><summary>候选 {decision["region_id"]} · P(yes)={decision["probabilities"]["yes"]:.4f} · {"接受" if decision["selected"] else "拒绝"}</summary><div class="grid">'
            for label, name in [('输入 1：RGB 裁剪','rgb_crop.png'), ('输入 2：编号候选','numbered_crop.png'),
                                ('输入 3：RGB＋算术质心','centroid.png')]:
                body += figure(folder, label, inp/name)
            body += '</div><pre>'+html.escape((inp/'prompt_zh.md').read_text())+'</pre><p>'
            body += anchor(folder, '实际英文请求', inp/'valen_request.json')+' · '
            body += anchor(folder, 'Valen 回答', inp/'valen_response.json')+'</p></details>'
        body += '<p>'+anchor(folder, '内部中心点方案：完整四阶段及全部输入',
            reference/'foreground/samples'/sid/'queries'/action/'index.html')+'</p>'
        (folder/'index.html').write_text(page(title, body), encoding='utf-8')
        pages.append(str(folder/'index.html'))
        cards.append((q, folder))
        details.append(diagnostic)
    comparisons = {}
    old = read(reference/'evaluation/metrics.json')['comparisons']
    for key in NAMES:
        t = totals[key]
        result = {'name': NAMES[key], 'metrics': summarize_queries(metrics[key]), 'counts': dict(t),
            'foreground_target_pixel_recall': t['covered_target_pixels']/t['target_pixels'],
            'foreground_gt_background_fraction': t['foreground_gt_background_pixels']/t['foreground_valid_pixels']}
        if key in old:
            for name in ['metrics', 'counts', 'foreground_target_pixel_recall', 'foreground_gt_background_fraction']:
                assert result[name] == old[key][name], (key, name)
        comparisons[key] = result
        write_jsonl(root/'evaluation'/key/'per_query.jsonl', metrics[key])
    summary = {'status': 'complete', 'images': len(rows), 'queries': len(qs), 'comparisons': comparisons,
        'protocol': protocol, 'baseline_functional_calls': calls, 'baseline_failed_queries': failures,
        'original_three_results_verified_unchanged': True,
        'metric_note': 'mIoU: mean of 7 class-wise pooled IoUs; gIoU: mean of 18 query IoUs. '
                       'Foreground ratios pooled over the same queries; GT=0 background, void=-1 ignored.'}
    write(root/'evaluation/metrics.json', summary)
    write_jsonl(root/'evaluation/diagnostics.jsonl', details)
    body = '<h1>前景选择对照：补充完整词表＋几何筛选</h1><p>相同 9 张 RGB／18 个功能对。'
    body += '新增 baseline 复用历史 PartGLEE 前景与部件候选，重新运行当前质心点 yes/no 功能判断；其余三组保持原预测。</p>'
    body += '<p>完整词表同时作用于物体和部件分割，因此本行是历史分割链路 baseline，不是只替换前景的单因素消融。</p>'
    body += '<table><tr><th>方法</th><th>功能像素覆盖 ↑</th><th>前景内 GT 背景占比 ↓</th><th>最终功能 mIoU ↑</th><th>gIoU ↑</th></tr>'
    for r in comparisons.values():
        body += f'<tr><td>{r["name"]}</td><td>{r["foreground_target_pixel_recall"]:.2%}</td><td>{r["foreground_gt_background_fraction"]:.2%}</td><td>{r["metrics"]["mIoU_known_operations"]:.2%}</td><td>{r["metrics"]["gIoU"]:.2%}</td></tr>'
    body += '</table><p>'+anchor(dest, '完整指标', root/'evaluation/metrics.json')+' · '
    body += anchor(dest, '词表与分割参数', root/'pipeline.json')+'</p><div class="grid">'
    for sid in rows:
        q = next(q for q in qs if q['sample_id'] == sid)
        body += '<figure><figcaption>'+html.escape(sid)+'</figcaption>'
        from .report import relative
        body += f'<img loading="lazy" src="{html.escape(relative(Path(q["rgb"]),dest),quote=True)}"><p>'
        body += ' · '.join(anchor(dest, query['action'], folder/'index.html') for query, folder in cards if query['sample_id'] == sid)
        body += '</p></figure>'
    body += '</div>'
    (dest/'index.html').write_text(page('前景选择方法对照', body), encoding='utf-8')
    write(root/'evaluation/gallery_audit.json', {'images': len(rows), 'queries': len(qs), 'pages': pages,
        'actual_valen_calls_audited': calls, 'all_original_results_unchanged': True})
    print({k: {n: v for n, v in r.items() if n not in ['counts', 'metrics']} |
        {'mIoU': r['metrics']['mIoU_known_operations'], 'gIoU': r['metrics']['gIoU']} for k, r in comparisons.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'report'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--reference', default=str(PROJECT/'outputs/umd_valen_point_union_20261019/smoke'))
    parser.add_argument('--candidates', default=str(PROJECT/'outputs/umd_novel_test'))
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(Path(args.root).resolve(), Path(args.reference).resolve(), Path(args.candidates).resolve())
    else:
        report(Path(args.root).resolve())


if __name__ == '__main__':
    main()
