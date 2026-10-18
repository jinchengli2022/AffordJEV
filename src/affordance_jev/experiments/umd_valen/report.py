"""Concise benchmark tables and a local, complete per-image inspection page."""
import argparse
import base64
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import os
import shutil

import numpy as np
from PIL import Image,ImageDraw

from ...visualization import font as render_font

from ...common import read,write,read_jsonl,write_jsonl,sha
from ...datasets.umd import load_annotation
from ...evaluation.metrics import Metrics
from .data import PROJECT

RUNS={
    'ooal':('OOAL／本地适配',PROJECT/'outputs/ooal_umd_novel_20261016/test'),
    'z0':('Valen／原始填色',PROJECT/'outputs/umd_visual_prompt_ablation/fill'),
    'outline':('Valen／仅描边',PROJECT/'outputs/umd_visual_prompt_ablation/outline'),
    'z1':('Valen／定义调整',PROJECT/'outputs/valen_76/z1_recheck'),
    's0':('Valen／普通 SFT',PROJECT/'outputs/valen_76/s0_test'),
}


def percent(value):
    return '—' if value is None else f'{value*100:.2f}%'


def relative(path,folder):
    return Path(os.path.relpath(path,folder)).as_posix()


def protocol_rescore(roots,output):
    manifest=read_jsonl(PROJECT/'data/UMD/manifests/umd_novel_test.jsonl')
    accum={key:Metrics() for key in roots}
    for i,row in enumerate(manifest,1):
        gt,_=load_annotation(row['annotation'],tuple(row['size_wh'][::-1]))
        if row['object_instance'].startswith('ladle_'):
            gt=gt.copy()
            gt[gt==4]=3
        for key,root in roots.items():
            path=root/'samples'/row['sample_id']/'result/class_map.png'
            state=read(root/'samples'/row['sample_id']/'run.json')
            pred=np.asarray(Image.open(path)) if state['status']=='complete' and path.exists() else np.zeros(gt.shape,np.uint8)
            accum[key].add(gt,pred)
        if i%500==0: print(f'RESCORE {i}/{len(manifest)}',flush=True)
    results={key:metric.summary() for key,metric in accum.items()}
    write(output/'ladle_remap_metrics.json',{'rule':'GT ladle contain(4) -> scoop(3); predictions unchanged',
          'results':results,'original_gt_files_modified':False})
    return results


def label_data(array):
    stream=io.BytesIO()
    Image.fromarray(np.asarray(array,dtype=np.uint8)).save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def region_comparison(roots,metrics,output):
    """Reuse diagnostic GT counts only after prediction; never change decisions."""
    source=PROJECT/'outputs/umd_error_analysis_20261016/per_region.jsonl'
    reference=read_jsonl(source)
    methods=[key for key in roots if key!='ooal']
    names=['none','grasp','cut','scoop','contain','pound','support','wrap-grasp']
    summaries={key:{'high_purity_confusion':np.zeros((8,8),dtype=int),'changed_from_z0':0,
                    'top2_correct':0} for key in methods}
    cache={}
    rows=[]
    for original in reference:
        sid=original['sample_id']
        if cache.get('sample_id')!=sid:
            cache={'sample_id':sid,**{key:read(root/'samples'/sid/'decisions.json')['regions']
                                    for key,root in roots.items() if key in methods}}
        count=np.asarray(original['gt_pixel_counts'])
        target=int(count.argmax())
        pure=target>0 and count.max()/max(1,count.sum())>=.9
        region=original['region_id']
        row={'sample_id':sid,'region_id':region,'gt_majority':names[target],
             'purity':float(count.max()/max(1,count.sum())),'choices':{}}
        for key in methods:
            decision=cache[key][region-1]
            assert decision['region_id']==region
            pred=names.index(decision['final_choice'])
            row['choices'][key]=decision['final_choice']
            summaries[key]['changed_from_z0']+=int(decision['final_choice']!=cache['z0'][region-1]['final_choice'])
            if pure:
                summaries[key]['high_purity_confusion'][target,pred]+=1
                top2=sorted(decision['probabilities'],key=decision['probabilities'].get,reverse=True)[:2]
                summaries[key]['top2_correct']+=int(names[target] in top2)
        rows.append(row)
    result={}
    for key,s in summaries.items():
        cm=s.pop('high_purity_confusion')
        result[key]={**s,'all_regions':len(reference),'pure_foreground_regions':int(cm.sum()),
                     'pure_region_accuracy':float(cm.trace()/cm.sum()),
                     'pure_region_macro_accuracy':float(np.mean([cm[c,c]/cm[c].sum() for c in range(1,8) if cm[c].sum()])),
                     'high_purity_confusion_gt_rows_pred_columns':cm.tolist(),
                     'labels':names}
    write(output/'region_comparison.json',{'methods':result,'gt_counts_source':str(source),
          'gt_only_used_after_prediction':True,'majority_tie_rule':'lowest native ID'})
    write_jsonl(output/'region_changes.jsonl',rows)
    return result


def gallery(roots,metrics,output):
    shutil.copytree(Path(__file__).parent/'assets',output/'fonts',dirs_exist_ok=True)
    rows=read_jsonl(PROJECT/'data/UMD/manifests/umd_novel_test.jsonl')
    per_image={key:{r['sample_id']:r for r in read_jsonl(root/'evaluation/per_image.jsonl')} for key,root in roots.items()}
    items=[]
    for i,row in enumerate(rows,1):
        gt,_=load_annotation(row['annotation'],tuple(row['size_wh'][::-1]))
        entry={'id':row['sample_id'],'category':row['object_instance'].rsplit('_',1)[0],
               'rgb':relative(PROJECT/'outputs/umd_novel_test/samples'/row['sample_id']/'rgb.png',output),
               'size':row['size_wh'],
               'crop':read(PROJECT/'outputs/umd_novel_test/samples'/row['sample_id']/'foreground/metadata.json')['crop_xyxy'],
               'gt':label_data(np.where(gt<0,255,gt)),'predictions':{},'ious':{},'inputs':{}}
        for key,root in roots.items():
            sample=root/'samples'/row['sample_id']
            path=sample/'result/class_map.png'
            entry['predictions'][key]='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode() if path.exists() else label_data(np.zeros(gt.shape,np.uint8))
            entry['ious'][key]=per_image[key][row['sample_id']]['mean_IoU_present_classes']
            if key!='ooal':
                entry['inputs'][key]=[relative(sample/p,output) for p in ['foreground/rgb_crop.png','regions/numbered_crop.png','regions/region_sheet.jpg']]
        items.append(entry)
        if i%500==0: print(f'GALLERY {i}/{len(rows)}',flush=True)
    payload={'methods':{k:RUNS[k][0] for k in roots},'items':items}
    (output/'data.js').write_text('window.EXPERIMENT='+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    (output/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>UMD：输入呈现与普通 SFT 对比</title>
<style>@font-face{font-family:UMDExperiment;src:url('fonts/experiment-ui.woff') format('woff');font-display:swap}body{font:16px UMDExperiment,system-ui;margin:24px;color:#222;background:white}button,select{font:inherit;padding:6px;margin:4px}h1{font-size:24px}h2{font-size:18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:14px}.panel{position:relative}.panel img{display:block;width:100%;height:auto}.panel canvas{position:absolute;left:0;top:0;width:100%;height:100%}figure{margin:0}figcaption{margin:8px 0}.legend span{display:inline-block;padding:4px 9px;margin:4px;color:#111}.inputs{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}.inputs img{max-height:260px;max-width:100%}.note{color:#555;font-size:14px}</style>
<h1>UMD novel/test：完整逐图对比</h1>
<p class="note">固定同一批候选。原始 dominant GT，仅用于评估。图上分数为该图已有功能的平均 IoU；总体 mIoU 见结果文档。</p>
<select id="category"><option value="">全部物体类别</option></select><select id="sample"></select>
<button id="prev">上一张</button><button id="next">下一张</button>
<label><input id="crop" type="checkbox" checked>放大物体区域</label>
<div class="legend" id="legend"></div><div class="grid" id="grid"></div>
<h2>实际输入 Valen 的三张图片</h2><select id="inputMethod"></select><div class="inputs" id="inputs"></div>
<p class="note">输入图中的颜色／编号不代表功能类别。取消放大可查看完整画面及背景误报；放大范围来自固定的物体定位。</p>
<script src="data.js"></script><script>
const data=window.EXPERIMENT, names=['none','grasp','cut','scoop','contain','pound','support','wrap-grasp'];
const colors=[[135,135,135],[230,85,65],[65,120,230],[245,190,45],[60,180,95],[195,85,180],[45,190,200],[235,130,45]];
const $=id=>document.getElementById(id);let filtered=data.items;
for(const cat of [...new Set(data.items.map(x=>x.category))])$('category').add(new Option(cat,cat));
for(const [k,v] of Object.entries(data.methods))if(k!='ooal')$('inputMethod').add(new Option(v,k));
for(let k=1;k<8;k++){let s=document.createElement('span');s.style.background=`rgba(${colors[k]},0.5)`;s.textContent=names[k];$('legend').append(s)}
function panel(label,rgb,mask){let f=document.createElement('figure'),title=document.createElement('figcaption');title.textContent=label;f.append(title);let p=document.createElement('div');p.className='panel';let im=new Image();im.src=rgb;p.append(im);if(mask){let canvas=document.createElement('canvas');p.append(canvas);let m=new Image();m.onload=()=>{canvas.width=m.width;canvas.height=m.height;let ctx=canvas.getContext('2d');ctx.drawImage(m,0,0);let pix=ctx.getImageData(0,0,m.width,m.height);for(let i=0;i<pix.data.length;i+=4){let k=pix.data[i];if(k>=1&&k<=7){let c=colors[k];pix.data[i]=c[0];pix.data[i+1]=c[1];pix.data[i+2]=c[2];pix.data[i+3]=140}else pix.data[i+3]=0}ctx.putImageData(pix,0,0)};m.src=mask}f.append(p);$('grid').append(f)}
function inputs(){let item=filtered[+$('sample').value];$('inputs').replaceChildren();for(const [i,src] of item.inputs[$('inputMethod').value].entries()){let f=document.createElement('figure'),cap=document.createElement('figcaption');cap.textContent=['1 · RGB 裁剪','2 · 全部候选编号','3 · 逐区域拼图'][i];let a=document.createElement('a');a.href=src;a.target='_blank';let im=new Image();im.src=src;a.append(im);f.append(cap,a);$('inputs').append(f)}}
function show(){let item=filtered[+$('sample').value];$('grid').replaceChildren();panel('RGB · '+item.id,item.rgb,null);panel('UMD 原始 GT',item.rgb,item.gt);for(const [k,v]of Object.entries(data.methods)){let score=item.ious[k];panel(v+' · '+(score==null?'—':(score*100).toFixed(2)+'%'),item.rgb,item.predictions[k])}let [x0,y0,x1,y1]=$('crop').checked?item.crop:[0,0,...item.size];for(const p of document.querySelectorAll('.panel')){p.style.aspectRatio=(x1-x0)+' / '+(y1-y0);p.style.overflow='hidden';for(const im of p.children){im.style.position='absolute';im.style.width=(100*item.size[0]/(x1-x0))+'%';im.style.height=(100*item.size[1]/(y1-y0))+'%';im.style.left=(-100*x0/(x1-x0))+'%';im.style.top=(-100*y0/(y1-y0))+'%'}}inputs()}
function filter(){filtered=data.items.filter(x=>!$('category').value||x.category==$('category').value);$('sample').replaceChildren();filtered.forEach((x,i)=>$('sample').add(new Option(x.id,String(i))));show()}
$('category').onchange=filter;$('sample').onchange=show;$('inputMethod').onchange=inputs;
$('crop').onchange=show;
$('prev').onclick=()=>{$('sample').selectedIndex=($('sample').selectedIndex+filtered.length-1)%filtered.length;show()};
$('next').onclick=()=>{$('sample').selectedIndex=($('sample').selectedIndex+1)%filtered.length;show()};filter();
</script></html>''',encoding='utf-8')


def representative_examples(roots,output):
    """Show both directions of change; choose median changes rather than extremes."""
    pairs=[('outline','z0','Outline / filled'),('s0','z1','SFT / same-prompt zero-shot')]
    per_image={k:{r['sample_id']:r['mean_IoU_present_classes'] for r in read_jsonl(root/'evaluation/per_image.jsonl')}
               for k,root in roots.items()}
    chosen=[]
    for method,baseline,label in pairs:
        if method not in roots or baseline not in roots: continue
        delta=[(value-per_image[baseline][sid],sid) for sid,value in per_image[method].items()]
        for direction in [1,-1]:
            group=sorted((d,sid) for d,sid in delta if direction*d>1e-8)
            if not group: continue
            d,sid=group[len(group)//2]
            chosen.append({'sample_id':sid,'reference':baseline,'variant':method,'delta':d,'label':label,
                           'selection':'median positive change' if direction>0 else 'median negative change'})
    if not chosen: return
    rows={r['sample_id']:r for r in read_jsonl(PROJECT/'data/UMD/manifests/umd_novel_test.jsonl')}
    palette=np.asarray([[0,0,0],[230,85,65],[65,120,230],[245,190,45],[60,180,95],
                       [195,85,180],[45,190,200],[235,130,45]],dtype=np.uint8)
    f=render_font(18)
    canvas=Image.new('RGB',(1440,365*len(chosen)+65),'white')
    draw=ImageDraw.Draw(canvas)
    for line,item in enumerate(chosen):
        sid=item['sample_id'];row=rows[sid];sample=PROJECT/'outputs/umd_novel_test/samples'/sid
        rgb=np.asarray(Image.open(sample/'rgb.png').convert('RGB'))
        crop=read(sample/'foreground/metadata.json')['crop_xyxy']
        gt,_=load_annotation(row['annotation'],tuple(row['size_wh'][::-1]))
        maps=[None,gt]+[np.asarray(Image.open(roots[k]/'samples'/sid/'result/class_map.png')) for k in [item['reference'],item['variant']]]
        y=365*line
        draw.text((12,y+6),f"{item['label']} | {sid} | image IoU change: {item['delta']*100:+.2f} pp",font=f,fill='black')
        titles={'z0':'Filled','outline':'Outline','z1':'Prompt adjusted','s0':'SFT'}
        for col,(title,labelmap) in enumerate(zip(['RGB','GT',titles[item['reference']],titles[item['variant']]],maps)):
            visual=rgb.copy()
            if labelmap is not None:
                valid=(labelmap>0)&(labelmap<8)
                visual[valid]=(.55*visual[valid]+.45*palette[labelmap[valid]]).astype(np.uint8)
            view=Image.fromarray(visual).crop(crop)
            scale=min(348/view.width,295/view.height)
            view=view.resize((max(1,round(view.width*scale)),max(1,round(view.height*scale))),Image.Resampling.LANCZOS)
            x=col*360
            draw.text((x+12,y+34),title,font=f,fill='black')
            canvas.paste(view,(x+(360-view.width)//2,y+64+(295-view.height)//2))
    y=365*len(chosen)+15
    for i,name in enumerate(['grasp','cut','scoop','contain','pound','support','wrap-grasp'],1):
        x=12+(i-1)*200
        draw.rectangle((x,y,x+20,y+20),fill=tuple(palette[i].tolist()))
        draw.text((x+27,y),name,font=f,fill='black')
    canvas.save(output/'examples.jpg',quality=94)
    write(output/'examples.json',{'cases':chosen,'scores':'per-image mean IoU over GT-present classes; not global mIoU',
                                'crop':'fixed predicted object crop; use full gallery to inspect full frame'})


def update_document(metrics,remap,regions,output):
    document=PROJECT/'docs/research/实验结果汇总.md'
    date=datetime.now(timezone.utc).strftime('%Y-%m-%d')
    lines=['# UMD 实验结果汇总','',
           f'{date}。统一使用 **novel/test 的 2,661 张人工帧、原始 dominant GT**；7 类 mIoU，背景误报计入，忽略 void。所有 Valen 方案均为 **PartGLEE＋Valen-4B**，使用相同候选。','',
           '| 方法 | UMD 训练图 | mIoU ↑ | gIoU ↑ | P@50 ↑ | Dice ↑ |',
           '|---|---:|---:|---:|---:|---:|']
    for key,(label,root) in RUNS.items():
        count=76 if key in ['ooal','s0'] else 0
        if key not in metrics:
            lines.append(f'| {label} | {count} | 进行中 | — | — | — |')
            continue
        m=metrics[key]['metrics']
        values=[percent(m[k]) for k in ['foreground_mIoU','gIoU_present_queries','P_at_50_present_queries','mean_Dice_present_queries']]
        if key=='s0': values[0]='**'+values[0]+'**'
        lines.append(f'| {label} | {count} | '+' | '.join(values)+' |')
    lines+=['']
    lines+=['填色／定义调整采用本轮同环境复算；历史值为 41.64%／41.17%，原输出保留。[复算差异记录](../../outputs/valen_76/provenance/repeat_audit.json)','']
    comparisons=[('outline','z0','仅描边相对原始填色'),('z1','z0','定义调整相对原始提示词'),('s0','z1','SFT 相对相同提示词的未训练模型')]
    effects=[]
    for a,b,label in comparisons:
        if a in metrics and b in metrics:
            delta=100*(metrics[a]['metrics']['foreground_mIoU']-metrics[b]['metrics']['foreground_mIoU'])
            effects.append(f'{label} **{delta:+.2f}** 个百分点')
    if effects: lines+=['；'.join(effects)+'。','']
    lines+=['mIoU 按 7 类功能平均；gIoU、P@50、Dice 按 5,232 个图像＋功能对统计。P@50 不是机器人操作成功率。','']
    completed=[k for k in RUNS if k in metrics]
    lines+=['| 功能 IoU | '+' | '.join(RUNS[k][0].replace('Valen／','').replace('OOAL／本地适配','OOAL') for k in completed)+' |',
            '|---|'+'---:|'*len(completed)]
    for name in ['grasp','cut','scoop','contain','pound','support','wrap-grasp']:
        best=max(metrics[k]['metrics']['per_class'][name]['iou'] for k in completed)
        values=[('**'+percent(metrics[k]['metrics']['per_class'][name]['iou'])+'**')
                if metrics[k]['metrics']['per_class'][name]['iou']==best else percent(metrics[k]['metrics']['per_class'][name]['iou']) for k in completed]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    if 's0' in metrics:
        s=metrics['s0']['metrics']['per_class']
        lines+=['',f"SFT 的主要提升来自 **wrap-grasp（{percent(s['wrap-grasp']['iou'])}）**；**support 仍仅 {percent(s['support']['iou'])}**，功能混淆尚未解决。"]
    lines+=['',
        '- **描边**：仅取消输入图的 45% 区域填色，轮廓、编号、Prompt、mask 和权重固定。',
        '- **定义调整**：仅修改混合区域与 none 的说明，要求选择主要功能；其余七类定义不变，保持填色输入。',
        '- **SFT**：76 张／366 个候选，32 张人工、44 张自动标注；按候选内 GT 多数类别监督。决策头预热 300 步＋LoRA 联合 1,000 步，r=8、有效 batch=4、seed=321；固定最终权重，未用 test 选模型。',
        '- **OOAL**：本地适配、20,000 步；作者原 UMD 配置未完全公开，本结果不等于精确复现论文 59.7%。','']
    timing={}
    for key in ['z0','outline','z1','s0']:
        p=PROJECT/'outputs/valen_76/timing'/('z0_verified' if key=='z0' else key)/'metrics.json'
        if p.exists(): timing[key]=read(p)
    if timing:
        lines+=['| 推理耗时 | Valen 均值 | RGB→mask 均值 | RGB→mask P95 |', '|---|---:|---:|---:|']
        for key,item in timing.items():
            t=item['timing_seconds']
            lines.append(f"| {RUNS[key][0]} | {t['valen_seconds']['mean']:.3f} s | {t['end_to_end_seconds']['mean']:.3f} s | {t['end_to_end_seconds']['p95']:.3f} s |")
        lines+=['','同一 RTX 4090，两个模型常驻，batch=1，预热后测相同 200 张图；每次判断该图全部候选。完整链路含 RGB 读取、两次 PartGLEE、候选处理、三图传递、Valen 和合并；不含加载权重、展示及结果保存。','']
    lines+=['OOAL 本地实测为 **14.05 ms／图**（RGB→类别图，预热后、不含保存）；当前框架仍慢于 OOAL。','']
    if regions:
        n=next(iter(regions.values()))['pure_foreground_regions']
        lines+=[f'纯度 ≥90% 的 {n:,} 个前景候选，对 GT 多数类的一致率：'+ '；'.join(f"{RUNS[k][0].replace('Valen／','')} **{percent(v['pure_region_accuracy'])}**" for k,v in regions.items())+'。','']
    lines+=['**协议敏感性：** 只将 ladle 的 GT `contain→scoop`、保持预测不变时，各方法 mIoU 如下：','',
            '| '+' | '.join(RUNS[k][0] for k in completed)+' |',
            '|'+'---:|'*len(completed),
            '| '+' | '.join(percent(remap[k]['foreground_mIoU']) for k in completed)+' |','']
    failures={k:sum(metrics[k].get('failures',{}).values()) for k in completed}
    lines+=['失败计入评测；本轮已完成方案的失败数：'+('均为 0。' if not any(failures.values()) else '、'.join(f'{k}={v}' for k,v in failures.items())+'。'),'',
            '固定候选、使用 GT 多数类别赋类的 **81.95% mIoU** 仅是诊断值，不是模型成绩或严格上限。[误差分析](PartGLEE与Valen误差分析.md)','',
            '[完整逐图对比与三张实际输入](../../outputs/valen_76/comparison/index.html) · [改善／退化示例](../../outputs/valen_76/comparison/examples.jpg) · [训练配置](../../configs/valen_76_sft.json) · [SFT 权重](../../checkpoints/valen_76/s0/321/final) · [OOAL 复现说明](../中间文档/OOAL复现说明.md)','']
    lines+=['SFT 为 seed=321 单次结果，训练／测试实例及物体类别分离；未执行 A／B／C。测试集此前用于误差分析，不称独立盲测。','']
    links=[]
    for k in completed:
        links.append(f'[{RUNS[k][0]}指标]({relative(RUNS[k][1]/"evaluation/metrics.json",document.parent)})')
    lines+=[' · '.join(links),'']
    if 's0' not in metrics: lines+=['**普通 SFT 仍在进行中，尚无测试成绩。**','']
    if document.exists() and not (output/'summary_before_this_run.md').exists():
        (output/'summary_before_this_run.md').write_text(document.read_text(),encoding='utf-8')
    document.write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-gallery',action='store_true')
    args=parser.parse_args()
    output=PROJECT/'outputs/valen_76/comparison'
    output.mkdir(parents=True,exist_ok=True)
    roots={k:root for k,(_,root) in RUNS.items() if (root/'evaluation/metrics.json').exists()}
    metrics={k:read(root/'evaluation/metrics.json') for k,root in roots.items()}
    for k,m in metrics.items():
        assert m['metrics']['images']==2661,k
        assert m['metrics']['gt_present_queries']==5232,k
    remap=protocol_rescore(roots,output)
    regions=region_comparison(roots,metrics,output)
    if not args.skip_gallery: gallery(roots,metrics,output)
    representative_examples(roots,output)
    update_document(metrics,remap,regions,output)
    write(output/'summary.json',{k:m['metrics'] for k,m in metrics.items()})
    print(PROJECT/'docs/research/实验结果汇总.md',flush=True)


if __name__=='__main__': main()
