"""Evaluate new predictions only; publish a compact table and a complete query viewer."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ...common import read, write, read_jsonl, write_jsonl, sha, load_mask
from ...datasets.umd import load_annotation, CLASS_NAMES
from ...baselines.affordance_r1.compare import query_score, summarize_queries
from ...paths import PROJECT
from ...pipeline.run import sample_directory
from ...visualization import overlay
from .requests import METHODS


NAMES={'ooal':'OOAL／本地适配','r1':'Affordance-R1／zero-shot',**METHODS}
ORDER=['ooal','r1','fill','outline','centroid','definition','sft']


def evaluate(root,method):
    root=Path(root).resolve()
    protocol=read(root/'protocol.json')
    assert sha(root/'test_queries.jsonl')==protocol['counts']['test']['queries_sha256']
    queries=read_jsonl(root/'test_queries.jsonl')
    rows={r['sample_id']:r for r in read_jsonl(root/'test_manifest.jsonl')}
    target=root/'methods'/method
    groups=defaultdict(list)
    for q in queries: groups[q['sample_id']].append(q)
    scores=[]
    failures=0
    timings=[]
    for sid, qs in groups.items():
        sample=sample_directory(target,sid)
        row=rows[sid]
        if method=='ooal':
            if not (sample/'run.json').exists() or read(sample/'run.json')['status']!='complete':
                return None
        elif any(not (sample/'queries'/q['action']/'run.json').exists() or
                 read(sample/'queries'/q['action']/'run.json')['status'] not in ['complete','failed'] for q in qs):
            return None
        assert sha(row['annotation'])==row['annotation_sha256']
        gt,kind=load_annotation(row['annotation'],tuple(row['size_wh'][::-1]))
        assert kind=='manual'
        if method=='ooal':
            class_map=np.asarray(Image.open(sample/'result/class_map.png'))
        for q in qs:
            error=None
            if method=='ooal':
                mask=class_map==q['class_id']
                mask_path=sample/'result/masks'/f'{q["action"]}.png'
                assert np.array_equal(mask,load_mask(mask_path,gt.shape))
            else:
                out=sample/'queries'/q['action']
                assert read(out/'query.json')==q
                state=read(out/'run.json')
                mask_path=out/'mask.png'
                assert state['mask_sha256']==sha(mask_path)
                mask=load_mask(mask_path,gt.shape)
                error=state['error']
                if error:
                    assert not mask.any()
                    failures+=1
                if (out/'timing.json').exists():
                    timings.append(read(out/'timing.json'))
            scores.append({**query_score(sid,q['class_id'],gt,mask),
                           'mask_sha256':sha(mask_path),'error':error})
    assert len(scores)==5232 and len({(r['sample_id'],r['class_id']) for r in scores})==5232
    metrics=summarize_queries(scores)
    result={'method':NAMES[method],'UMD_training_images':76 if method in ['ooal','sft'] else 0,
        'status':'complete','fresh_inference':True,'metrics':metrics,'failed_queries':failures,
        'queries_sha256':sha(root/'test_queries.jsonl'),'manifest_sha256':sha(root/'test_manifest.jsonl')}
    if timings:
        key='seconds_per_query_amortized' if method=='r1' else 'valen_predict_seconds'
        result['timing']={'mean_seconds':float(np.mean([t[key] for t in timings])),
            'scope': 'amortized RGB -> requested masks, sharing encoder and batching' if method=='r1'
                     else 'Valen only per requested operation, including preprocessing and input audit saving',
            'strict_serial_end_to_end_latency':False}
    write_jsonl(target/'evaluation/per_query.jsonl',scores)
    write(target/'evaluation/metrics.json',result)
    return result


def refresh(root):
    root=Path(root).resolve()
    results={}
    for key in ORDER:
        result=evaluate(root,key)
        if result is not None: results[key]=result
        print(f'EVALUATE {key}: {"complete" if result else "pending"}',flush=True)
    write(root/'comparison/metrics.json',{'protocol':read(root/'protocol.json'),'results':results})
    lines=['<!-- conditional-umd-start -->','### 给定物体与操作：完整重跑','',
        '**2,661 张 RGB／5,232 个预先固定的功能对。** 全部重新推理；PartGLEE 仅使用 `object / part`，五种 Valen 方法共享本轮新候选。','',
        'OOAL 保留八类竞争后提取目标类；Affordance-R1 使用官方 prompt＋物体名＋目标操作；Valen 对每个候选作目标操作的 yes/no 判断并合并 yes。后两者的各操作独立，不做跨操作消重。','',
        '| 方法 | UMD 训练图 | mIoU ↑ | gIoU ↑ | P@50 ↑ | Dice ↑ | 状态 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for key in ORDER:
        row=results.get(key)
        values=[f'{row["metrics"][k]*100:.2f}%' for k in ['mIoU_known_operations','gIoU','P_at_50','Dice']] if row else ['—']*4
        lines.append('| '+' | '.join([NAMES[key],str(76 if key in ['ooal','sft'] else 0),*values,'已完成' if row else '进行中'])+' |')
    lines+=['','SFT 从原始 Valen 重训：同一批 76 张训练图，头预热 300 步＋LoRA 1,000 步，固定最终权重；按候选 GT 多数类别生成二选一标签。测试集不参与训练或选权重。',
        '', 'mIoU＝七类累计交并比的平均；gIoU＝功能对 IoU 平均。背景误报计入、void 忽略；失败按空预测计分。历史结果与本轮的候选及判断任务不同。','']
    if results:
        lines+=['**各功能 IoU**','','| 功能 | '+' | '.join(NAMES[k].replace('Valen／','') for k in ORDER if k in results)+' |',
                '|---|'+'---:|'*len(results)]
        for cid in range(1,8):
            action=CLASS_NAMES[cid]
            lines.append('| '+action+' | '+' | '.join(f'{results[k]["metrics"]["per_class"][action]["iou"]*100:.2f}%' for k in ORDER if k in results)+' |')
        lines.append('')
    relative='../../outputs/'+root.name
    lines += [f'[逐图／逐操作可视化]({relative}/comparison/index.html) · [完整指标]({relative}/comparison/metrics.json) · [训练配置]({relative}/training/config.json)',
              '<!-- conditional-umd-end -->']
    doc=PROJECT/'docs/research/实验结果.md'
    text=doc.read_text(encoding='utf-8')
    block='\n'.join(lines)
    start,end='<!-- conditional-umd-start -->','<!-- conditional-umd-end -->'
    if start in text:
        first=text.index(start); last=text.index(end,first)+len(end)
        text=text[:first]+block+text[last:]
    else:
        heading='## train-free 方法对照\n'
        assert text.count(heading)==1
        text=text.replace(heading,heading+'\n'+block+'\n\n### 历史实验\n',1)
    doc.write_text(text,encoding='utf-8')
    return results


def gallery(root):
    root=Path(root).resolve()
    dest=root/'comparison'
    dest.mkdir(exist_ok=True)
    queries=read_jsonl(root/'test_queries.jsonl')
    rows={r['sample_id']:r for r in read_jsonl(root/'test_manifest.jsonl')}
    results=read(dest/'metrics.json')['results']
    scores={k:{(r['sample_id'],r['class_id']):r['iou'] for r in read_jsonl(root/f'methods/{k}/evaluation/per_query.jsonl')} for k in results}
    cards=[]
    gt_cache=None
    last=None
    for i,q in enumerate(queries):
        sid=q['sample_id']; action=q['action']
        if sid!=last:
            gt_cache,_=load_annotation(rows[sid]['annotation'])
            last=sid
        rgb=Image.open(q['rgb']).convert('RGB')
        gt_rel=f'gt/{sid}/{action}.jpg'
        gt_path=dest/gt_rel
        if not gt_path.exists():
            gt_path.parent.mkdir(parents=True,exist_ok=True)
            overlay(rgb,[gt_cache==q['class_id']],colors=[[60,180,95]],labels=['']).save(gt_path,quality=90)
        predictions={}
        for key in ORDER:
            if key not in results: continue
            sample=sample_directory(root/'methods'/key,sid)
            if key=='ooal':
                out=sample/'queries'/action
                out.mkdir(parents=True,exist_ok=True)
                if not (out/'overlay.jpg').exists():
                    mask=load_mask(sample/'result/masks'/f'{action}.png')
                    overlay(rgb,[mask],colors=[[60,180,95]],labels=['']).save(out/'overlay.jpg',quality=90)
            predictions[key]={'image':f'../methods/{key}/samples/{sid}/queries/{action}/overlay.jpg',
                              'iou':scores[key][(sid,q['class_id'])],
                              'details':f'../methods/{key}/samples/{sid}/queries/{action}/'}
        cards.append({'id':q['query_id'],'category':q['object_category'],'action':action,
            'rgb':f'../candidates_test/samples/{sid}/rgb.png','gt':gt_rel,'predictions':predictions,
            'candidates':f'../candidates_test/samples/{sid}/index.html'})
        if (i+1)%500==0: print(f'GALLERY {i+1}/{len(queries)}',flush=True)
    (dest/'data.js').write_text('window.DATA='+json.dumps({'names':NAMES,'order':ORDER,'cards':cards},ensure_ascii=False)+';\n',encoding='utf-8')
    (dest/'index.html').write_text(VIEWER,encoding='utf-8')


VIEWER='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>UMD 给定操作 · 完整重跑</title>
<style>body{font:16px system-ui;margin:24px;background:#fff;color:#222}header{position:sticky;top:0;background:white;padding:10px 0;border-bottom:1px solid #ddd}button,select{font:inherit;padding:6px;margin-right:8px}#grid{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:14px}img{width:100%;object-fit:contain}article{border:1px solid #ddd;padding:10px}h1{font-size:24px}h2{font-size:16px;margin:0 0 5px}a{color:#1764a2}#title{margin:16px 0}@media(max-width:800px){#grid{grid-template-columns:1fr 1fr}}</style>
<h1>给定物体与操作：完整重跑</h1><p>2,661 张 RGB · 5,232 个功能对 · 绿色显示目标操作区域</p>
<header><select id="cat"></select><select id="act"></select><button id="prev">上一条</button><button id="next">下一条</button><input id="pos" type="number" min="1" value="1" style="width:80px"><span id="count"></span></header>
<div id="title"></div><div id="grid"></div><script src="data.js"></script><script>
const D=window.DATA,$=x=>document.getElementById(x);let list=D.cards,index=0;
for(const [id,key,label] of [['cat','category','全部物体'],['act','action','全部操作']]){
 const el=$(id);el.add(new Option(label,''));for(const v of [...new Set(D.cards.map(x=>x[key]))].sort())el.add(new Option(v,v));el.onchange=filter;
}
function filter(){list=D.cards.filter(x=>(!$('cat').value||x.category===$('cat').value)&&(!$('act').value||x.action===$('act').value));index=0;show()}
function show(){const q=list[index];$('grid').replaceChildren();$('count').textContent=' / '+list.length;$('pos').value=index+1;if(!q)return;
 $('title').replaceChildren(document.createTextNode(q.id+'　'));const link=document.createElement('a');link.href=q.candidates;link.textContent='检查 PartGLEE 候选';$('title').append(link);
 const panels=[['RGB',q.rgb,null],['GT',q.gt,null],...D.order.filter(k=>q.predictions[k]).map(k=>[D.names[k]+' · IoU '+(q.predictions[k].iou*100).toFixed(2)+'%',q.predictions[k].image,q.predictions[k].details])];
 for(const [name,image,details] of panels){const a=document.createElement('article'),h=document.createElement('h2'),im=document.createElement('img');h.textContent=name;im.src=image;a.append(h,im);if(details){const l=document.createElement('a');l.href=details;l.textContent='输入与判断记录';a.append(l)}$('grid').append(a)}
} $('prev').onclick=()=>{index=Math.max(0,index-1);show()};$('next').onclick=()=>{index=Math.min(list.length-1,index+1);show()};$('pos').onchange=()=>{index=Math.max(0,Math.min(list.length-1,Number($('pos').value)-1));show()};show();</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--gallery',action='store_true')
    args=parser.parse_args()
    refresh(args.root)
    if args.gallery: gallery(args.root)
