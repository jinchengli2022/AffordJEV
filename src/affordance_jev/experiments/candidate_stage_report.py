"""Concise report and auditable visual examples for stage attribution."""
import argparse
import html
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import read,read_jsonl,write,sha
from ..datasets.umd import load_annotation
from ..visualization import overlay,sheet
from .candidate_stage_diagnosis import STAGES,RAW,ROOT,candidate_stages
from .raw_candidates import load_proposals


LABELS={
    'full_joint':'整图原始 object＋part',
    'full_parts':'整图原始 part（仅保留 part 层）',
    'full_parts_fg':'原始 part 限制在已选前景内',
    'grey_full_parts_fg':'前景外置灰，整图重新分 part',
    'crop_parts_fg':'再裁剪，重新分 part',
    'quality_filter':'再加质量阈值 0.02',
    'component_filter':'再删除小连通区域',
    'partition_no_cap':'再互斥分配像素，不限 6 个',
    'final_cap6':'再限制最多 6 个（历史最终结果）',
}


def interval(value):
    lo,hi=value['lower']*100,value['upper']*100
    return f'{lo:.2f}%' if round(lo,2)==round(hi,2) else f'{lo:.2f}%–{hi:.2f}%'


def difference(current,previous):
    lo=(current['lower']-previous['upper'])*100
    hi=(current['upper']-previous['lower'])*100
    if max(abs(lo),abs(hi))<.005:
        return '≈0'
    return f'{lo:+.2f}' if round(lo,2)==round(hi,2) else f'{lo:+.2f}～{hi:+.2f}'


def make_table(metrics,variant='generic'):
    rows=['| 阶段 | Oracle gIoU | 相对上一步（百分点） |',
          '|---|---:|---:|']
    stages=metrics[variant]['stages']
    for i,name in enumerate(STAGES):
        change='—' if i==0 else difference(stages[name],stages[STAGES[i-1]])
        rows.append(f'| {LABELS[name]} | {interval(stages[name])} | {change} |')
    return '\n'.join(rows)


def render(out):
    out=Path(out).resolve()
    summary=read(out/'evaluation/metrics.json');assert summary['status']=='complete'
    metrics=summary['metrics']
    queries=read_jsonl(out/'evaluation/per_query.jsonl')
    manifest=read_jsonl(out/'manifest.jsonl')
    records={r['sample_id']:r for r in manifest}
    lookup={(q['query_id'],q['variant']):q for q in queries}
    assert len(lookup)==len(queries)==2*summary['queries']
    reference={(r['sample_id'],r['source'],r['class_id']):r['best_subset_iou']
        for r in read_jsonl(ROOT/'outputs/umd_given_operation_20261018/diagnosis/geometry_per_query.jsonl')}
    reproduction_errors=[abs(q['stages']['final_cap6']['lower']-reference[(q['sample_id'],q['variant'],q['class_id'])]) for q in queries]
    assert max(reproduction_errors)<1e-7
    assert all(q['stages']['quality_filter']['lower']<=q['stages']['crop_parts_fg']['upper']+1e-7 for q in queries)
    assert all(q['stages']['full_parts']['lower']<=q['stages']['full_joint']['upper']+1e-7 for q in queries)
    protocol=read(out/'protocol.json')
    # Smoke results may precede a validated evaluation optimization. Full-run
    # source hashes must match the files used throughout the production test.
    source_unchanged=all(sha(ROOT/k)==v for k,v in protocol['source_sha256'].items())
    if summary['images']==2661: assert source_unchanged
    assert sha(protocol['manifest'])==protocol['manifest_sha256']
    g=metrics['generic'];s=metrics['specific']
    repair=g['stages']['old_foreground_final']; baseline=g['stages']['final_cap6']
    text=[
        '# 前景、裁剪与候选过滤：阶段诊断',
        f"{summary['images']:,} 张图／{summary['queries']:,} 个已知操作查询。原始 UMD dominant GT；所有数值为候选最佳并集的 oracle gIoU，不调用 Valen。",
        '## 通用词表：逐项加入处理的对照',
        make_table(metrics),
        '前景 mask 固定为历史几何排序结果。首先保持整图原始 part 不变，仅与前景取交集，测量前景硬限制的影响；这一步是受控诊断。随后各组都限制在同一前景内，依次更换为置灰整图、置灰裁剪图的 part 输出；裁剪沿用 10% 边框。最后加入质量阈值、小连通域清理、互斥分配与 6 候选限制。',
        '小区域阈值为 max(64, 前景面积×3%)；互斥分配后仍使用同一阈值清理碎片。限制数量会重新分配重叠像素，阶段间可能存在交互；表中是按此固定顺序的增减，不是互相独立的因果贡献。',
        '## 只替换前景的完整对照',
        '| 前景来源（其余均为通用词表及原参数） | 最终候选 oracle gIoU |\n|---|---:|\n'
        f"| 通用词表＋几何选择 | {interval(baseline)} |\n"
        f"| 替换为旧具体词表方案预测的前景 | {interval(repair)} |",
        f"仅替换前景，完整后续流程变化 **{difference(repair,baseline)} 个百分点**，包含前景引起的裁剪范围、过滤阈值等连带变化，不能再与上表的阶段增减相加。替换 mask 来自旧模型预测，未用 GT 生成；这是诊断对照，不是无词表的部署方案。",
        '## 前景覆盖与过滤阈值',
        '| 诊断 | 具体词表 | 通用词表 |\n|---|---:|---:|\n'
        f"| 所选前景覆盖真实功能像素 | {s['foreground_gt_coverage']:.2%} | {g['foreground_gt_coverage']:.2%} |\n"
        f"| 裁剪框覆盖真实功能像素 | {s['crop_bbox_gt_coverage']:.3%} | {g['crop_bbox_gt_coverage']:.3%} |\n"
        f"| 平均小区域阈值（像素） | {s['mean_small_component_threshold']:.1f} | {g['mean_small_component_threshold']:.1f} |",
        '## 具体词表：相同阶段对照',
        make_table(metrics,'specific'),
        '## 计算与来源',
        '已完成所有图像、所有阶段。重叠 mask 按像素并集计算；求解器未完全收敛的查询保留可行下界和严格上界，表格在两位小数不同的情况下显示区间。历史最终 label map 逐像素复核一致。GT 仅用于 oracle 和评分；模型推理只读取 RGB、固定词表及已有预测前景。',
        '[可视化](comparison/index.html) · [完整指标](evaluation/metrics.json) · [逐查询、逐阶段结果](evaluation/per_query.jsonl) · [固定协议](protocol.json)',
    ]
    (out/'README.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')

    # Fixed representative frames plus explicitly labeled maximum-loss cases.
    first={}
    for row in manifest:first.setdefault(row['object_instance'].rsplit('_',1)[0],row['sample_id'])
    selected={q['query_id']:'每类首张人工帧' for q in queries if q['variant']=='generic' and q['sample_id'] in first.values()}
    generic=[q for q in queries if q['variant']=='generic']
    for before,after,label in [('full_parts','full_parts_fg','前景限制损失最大'),
                               ('grey_full_parts_fg','crop_parts_fg','裁剪损失最大'),
                               ('crop_parts_fg','component_filter','阈值过滤损失最大'),
                               ('component_filter','final_cap6','互斥与数量限制损失最大')]:
        q=max(generic,key=lambda q:q['stages'][before]['lower']-q['stages'][after]['lower'])
        selected[q['query_id']]=label
    preview=out/'comparison';preview.mkdir(exist_ok=True)
    body=['<h1>前景／裁剪／候选过滤：逐阶段 oracle</h1>',
          '<p>通用 object / part；绿色为 GT，蓝色为所选前景或最佳候选并集。每类首帧＋明确标出的最大损失案例；展示用于诊断，不是实际模型成绩。</p>',
          '<p><a href="../README.md">方法与汇总</a> · <a href="../evaluation/metrics.json">完整指标</a></p>']
    exports=[]
    panel_names=['full_parts','full_parts_fg','grey_full_parts_fg','crop_parts_fg',
                 'component_filter','partition_no_cap','final_cap6','old_foreground_final']
    panel_labels=['Full parts','Foreground gate','Grey full RGB','Cropped RGB',
                  'Small regions filtered','Disjoint regions','Final <= 6','Old foreground only']
    stage_cache={}
    for qid,selection_reason in selected.items():
        q=lookup[(qid,'generic')];row=records[q['sample_id']]
        rgb=Image.open(row['rgb']).convert('RGB')
        gt,_=load_annotation(row['annotation'],row['size_wh'][::-1]);target=gt==q['class_id']
        if q['sample_id'] not in stage_cache:
            stage_cache[q['sample_id']]=candidate_stages(row,out,'generic')
        stages,fg,meta=stage_cache[q['sample_id']]
        cropbox=meta['crop_xyxy']
        grey=np.asarray(rgb).copy();grey[~fg]=127
        panels=[('RGB',rgb),('GT: '+q['action'],overlay(rgb,[target],colors=[(30,180,60)],labels=['GT'])),
                ('Selected foreground',overlay(rgb,[fg],colors=[(45,135,230)],labels=['FG'])),
                ('Actual cropped input',Image.fromarray(grey).crop(cropbox))]
        for stage,label in zip(panel_names,panel_labels):
            choice=q['stages'][stage]['chosen']
            pred=stages[stage][choice].any(0) if choice else np.zeros_like(target)
            iou=float((pred&target).sum()/((pred|target)&(gt>=0)).sum())
            assert abs(iou-q['stages'][stage]['lower'])<1e-7,(qid,stage)
            panels.append((f'{label}: {iou:.1%}',overlay(rgb,[pred],colors=[(45,135,230)],labels=['oracle'])))
        image_name=qid.replace('/','__')+'.jpg'
        sheet(panels,preview/image_name,columns=4,width=400,height=340)
        body.append(f'<h2>{html.escape(qid)} — {selection_reason}</h2><a href="{image_name}"><img loading="lazy" src="{image_name}"></a>')
        exports.append({'query_id':qid,'selection_reason':selection_reason,'image':image_name})
    write(preview/'examples.json',exports)
    (preview/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>候选阶段诊断</title><style>body{font-family:system-ui;margin:24px;color:#222;background:white}h1{font-size:24px}h2{font-size:17px;margin-top:30px}img{max-width:100%;height:auto}a{color:#1260a0}</style>'+''.join(body)+'</html>',encoding='utf-8')
    write(out/'completion_audit.json',{'images':summary['images'],'queries_per_variant':summary['queries'],
        'unique_query_variant_records':len(lookup),'historical_oracle_max_error':max(reproduction_errors),
        'quality_subset_bound_checked':True,'parts_subset_bound_checked':True,
        'manifest_unchanged':True,'source_unchanged':source_unchanged,'examples_validated':len(exports),
        'oracle_intervals_preserved':True,'gt_in_model_input':False})
    return {'images':summary['images'],'queries':summary['queries'],'examples':len(exports)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(render(parser.parse_args().output))
