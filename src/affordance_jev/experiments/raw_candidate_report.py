"""Human-readable report for the separate full-RGB raw proposal diagnostic."""
import argparse
import html
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read, read_jsonl, save_mask, write
from ..datasets.umd import load_annotation
from ..visualization import overlay, sheet
from .raw_candidates import VARIANTS, load_proposals


def build(out):
    out = Path(out).resolve()
    summary = read(out/'evaluation/metrics.json')
    assert summary['status'] == 'complete'
    rows = read_jsonl(out/'evaluation_manifest.jsonl')
    queries = read_jsonl(out/'evaluation/per_query.jsonl')
    expected = read_jsonl(out/'queries.jsonl')
    lookup = {(q['query_id'],q['variant']):q for q in queries}
    assert len(rows) == summary['images']
    assert len(lookup) == 2*len(expected) == 2*summary['queries']
    for q in expected:
        assert all((q['query_id'],v) in lookup for v in VARIANTS)
    metrics = {v:summary['metrics'][v]['overall'] for v in VARIANTS}
    table = ['| 指标 | 具体词表 | `object / part` | 变化（百分点） |',
             '|---|---:|---:|---:|']
    for label,key in [('Object 最佳单个 mask：平均 IoU','object_best_single'),
                      ('Part 最佳单个 mask：平均 IoU','part_best_single'),
                      ('Object＋Part 最佳单个 mask：平均 IoU','joint_best_single'),
                      ('Object＋Part 最佳并集：gIoU 下界','union_oracle_lower'),
                      ('Object＋Part 最佳并集：gIoU 上界','union_oracle_upper'),
                      ('全部候选覆盖真实功能像素','raw_covered_gt_pixels')]:
        a,b = (metrics[v][key] for v in VARIANTS)
        table.append(f'| {label} | {a:.4%} | {b:.4%} | {(b-a)*100:+.4f} |')
    category_table = ['| 物体类别 | 功能对数 | 具体词表 oracle gIoU | 通用词表 oracle gIoU |',
                      '|---|---:|---:|---:|']
    for cat in summary['metrics']['specific']['by_category']:
        a,b = (summary['metrics'][v]['by_category'][cat] for v in VARIANTS)
        def interval(r):
            return f"{r['union_oracle_lower']:.2%}–{r['union_oracle_upper']:.2%}"
        category_table.append(f"| {cat} | {a['queries']} | {interval(a)} | {interval(b)} |")
    note = (f"具体词表／通用词表分别有 {metrics['specific']['union_oracle_certified_queries']}／"
            f"{metrics['generic']['union_oracle_certified_queries']} 个查询获得最优性证书（IoU 误差 ≤ 10⁻⁶）。"
            '未收敛查询保留可行下界与严格上界，不将启发式分数称为精确 oracle。')
    content = [
        '# 原始 object＋part 候选 oracle',
        f"{summary['images']:,} 张完整 RGB，{summary['queries']:,} 个固定已知操作查询；GT 为原始 dominant，void 忽略。",
        '两组仅更换文本：固定具体物体／部件词表，或 `object / part`。每组直接对同一完整 RGB 前向一次，保留原生返回的 50 个 object＋50 个 part mask。没有本地前景选择、置灰、裁剪、质量／面积筛选、NMS、互斥化或 6 区域上限。模型内部 top-k、正常缩放与 mask logit > 0 二值化保持相同。',
        '\n'.join(table),
        'Oracle 用 GT 选择整个候选 mask 的最佳并集，每个操作独立优化；允许候选重叠，重叠像素只计一次，不切割、不重画边界。它是候选诊断，不能直接部署，也不等同于七类 mIoU。',
        note,
        '\n'.join(category_table),
        '注意：历史流程的 part 来自选定前景的裁剪图；本轮全部来自完整 RGB，因此原始候选不是历史最终候选的严格超集，不能把两轮数值之差解释为纯后处理损失。全部候选接近 100% 的覆盖率也不意味着高 IoU，大范围背景 mask 同样能覆盖目标。',
        '[代表样本可视化](comparison/index.html) · [完整指标](evaluation/metrics.json) · [逐查询指标与选择的候选编号](evaluation/per_query.jsonl) · [固定协议与词表](protocol.json)',
    ]
    (out/'README.md').write_text('\n\n'.join(content)+'\n',encoding='utf-8')

    # Fixed first frame per category, chosen before scoring (same as smoke check).
    picked = {}
    for row in rows:
        picked.setdefault(row['object_instance'].rsplit('_',1)[0],row)
    preview = out/'comparison'
    preview.mkdir(exist_ok=True)
    body = ['<h1>完整 RGB → 原始 object＋part 候选 oracle</h1>',
            '<p>每类首张人工帧的全部已知操作。每行：RGB → GT → 具体词表最佳并集 → object/part 最佳并集。仅用于分割诊断，GT 选择候选不属于实际模型能力。</p>',
            '<p><a href="../README.md">方法与指标</a> · <a href="../evaluation/metrics.json">完整指标</a> · <a href="../evaluation/per_query.jsonl">所有查询与 mask 编号</a></p>']
    exports = []
    for row in picked.values():
        sid = row['sample_id']
        rgb = Image.open(row['rgb']).convert('RGB')
        gt,_ = load_annotation(row['annotation'], row['size_wh'][::-1])
        masks = {v:load_proposals(out/'proposals'/v/sid) for v in VARIANTS}
        for q in [x for x in expected if x['sample_id']==sid]:
            qid = q['query_id']
            stem = qid.replace('/','__')
            target = gt == q['class_id']
            panels = [('RGB',rgb), (f"GT: {q['action']}",overlay(rgb,[target],colors=[(30,180,60)],labels=['GT']))]
            export = {'query_id':qid}
            for variant in VARIANTS:
                result = lookup[(qid,variant)]
                selected = result['union_oracle']['chosen']
                prediction = masks[variant][selected].any(0) if selected else np.zeros_like(target)
                actual = float((prediction&target).sum()/((prediction|target)&(gt>=0)).sum())
                assert abs(actual-result['union_oracle']['lower']) < 1e-7
                save_mask(preview/f'{stem}__{variant}.png', prediction)
                title = f'{variant}: {actual:.2%} / {len(selected)} masks'
                panels.append((title,overlay(rgb,[prediction],colors=[(45,135,230)],labels=['oracle'])))
                export[variant] = {'iou':actual,'selected_raw_ids':selected}
            image_name = f'{stem}.jpg'
            sheet(panels,preview/image_name,columns=4,width=400,height=340)
            body.append(f'<h2>{html.escape(qid)}</h2><a href="{image_name}"><img loading="lazy" src="{image_name}"></a>')
            body.append('<p>'+html.escape('；'.join(f"{v} 候选编号：{export[v]['selected_raw_ids']}" for v in VARIANTS))+'</p>')
            exports.append(export)
    write(preview/'selected_examples.json',exports)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原始候选 oracle</title><style>body{font-family:system-ui;margin:24px;color:#222;background:white}h1{font-size:24px}h2{font-size:17px;margin-top:30px}p{max-width:1200px}img{max-width:100%;height:auto}a{color:#1260a0}</style>'+''.join(body)+'</html>'
    (preview/'index.html').write_text(page,encoding='utf-8')
    return {'images':len(rows),'queries':len(expected),'example_queries':len(exports),'metrics':metrics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    print(build(parser.parse_args().output))
