"""Audit and compare every fixed final checkpoint; keep the first-round report."""
import argparse
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image,ImageDraw

from ....common import read,write,read_jsonl,write_jsonl,sha
from ....paths import training_config
from ....visualization import font as render_font
from ....datasets.umd import load_annotation
from ..data import PROJECT
from .. import report as previous
from .data import NAMES

ROOT=PROJECT/'outputs/valen_76_abc_20261017'
LABELS={'s0':'普通 SFT／首轮','s0r':'普通 SFT／本轮对照','a0':'A0／仅增强',
        'a1':'A1／增强＋配对约束','b0':'B0／组成监督',
        'b1':'B1／组成监督＋一致性','c0':'C0／目标区域特征','c1':'C1／目标＋邻区＋整物体特征'}
METHODS=['s0r','a0','a1','b0','b1','c0','c1']
PAIRS=[('a1','a0'),('b1','b0'),('c1','c0')]


def roots():
    return {'s0':PROJECT/'outputs/valen_76/s0_test',**{k:ROOT/'tests'/k for k in METHODS}}


def load_metrics(paths):
    result={k:read(p/'evaluation/metrics.json') for k,p in paths.items()}
    for key,item in result.items():
        assert item['metrics']['images']==2661 and item['metrics']['gt_present_queries']==5232,key
        assert item['manifest_sha256']==sha(PROJECT/'data/UMD/manifests/umd_novel_test.jsonl')
    return result


def audit(paths,output):
    rows=read_jsonl(PROJECT/'data/UMD/manifests/umd_novel_test.jsonl')
    original=PROJECT/'outputs/umd_novel_test'
    findings={k:{'images':0,'candidate_masks':0,'identical_model_input_files':0,'zero_output_token_responses':0,
                 'explicit_region_head_responses':0} for k in paths}
    for row in rows:
        source=original/'samples'/row['sample_id']
        digest=sha(source/'candidates.json')
        candidates=read(source/'candidates.json')['candidates']
        input_hashes={name:sha(paths['s0r']/'samples'/row['sample_id']/name) for name in [
            'foreground/rgb_crop.png','regions/numbered_crop.png','regions/region_sheet.jpg',
            'valen_request.json','affordances_snapshot.json']}
        for key,root in paths.items():
            sample=root/'samples'/row['sample_id']
            assert sha(sample/'candidates.json')==digest
            for name,expected in input_hashes.items():
                assert sha(sample/name)==expected,(key,row['sample_id'],name)
                findings[key]['identical_model_input_files']+=1
            for candidate in candidates:
                assert sha(sample/candidate['mask'])==candidate['mask_sha256']
            response=read(sample/'valen_response.json')
            assert response['usage']['output_tokens']==0
            findings[key]['zero_output_token_responses']+=1
            if key in ['c0','c1']:
                assert response['explicit_region_head']==key
                findings[key]['explicit_region_head_responses']+=1
            findings[key]['images']+=1
            findings[key]['candidate_masks']+=len(candidates)
    checkpoints={}
    for key in METHODS:
        config=training_config(PROJECT/'configs/valen_76_abc'/f'{key}.json')
        final=Path(config['checkpoint_dir'])/'final'
        assert read(final/'status.json')['step']==1000
        used=read(paths[key]/'checkpoint_used.json')
        assert Path(used['checkpoint']).resolve()==final.resolve()
        assert used['head_sha256']==sha(final/'head.safetensors')
        assert used['adapter_sha256']==sha(final/'adapter/adapter_model.safetensors')
        assert used['training_config.json_sha256']==sha(final/'training_config.json')
        if key in ['c0','c1']:
            assert used['extra_heads.safetensors_sha256']==sha(final/'extra_heads.safetensors')
        checkpoints[key]={str(p.relative_to(final)):sha(p) for p in final.rglob('*')
                          if p.is_file() and p.suffix in ['.json','.safetensors']}
    write(output/'provenance_audit.json',{'fixed_test_candidates_verified':findings,
          'checkpoint_sha256':checkpoints,'selection':'same fixed final 1000 updates, no test selection'})


def paired_bootstrap(metrics,output):
    """Resample 29 object instances, never treat their adjacent frames as IID."""
    instances=sorted(metrics['s0r']['per_object'])
    arrays={k:np.stack([m['per_object'][s]['confusion_matrix_gt_rows_pred_columns'] for s in instances])
            for k,m in metrics.items()}
    rng=np.random.default_rng(321)
    draws=rng.integers(0,len(instances),(2000,len(instances)))
    scores={}
    for key,array in arrays.items():
        cm=array[draws].sum(1)
        intersection=np.diagonal(cm,axis1=1,axis2=2)
        union=cm.sum(1)+cm.sum(2)-intersection
        iou=np.divide(intersection[:,1:],union[:,1:],out=np.full_like(intersection[:,1:],np.nan,dtype=float),where=union[:,1:]>0)
        scores[key]=np.nanmean(iou,axis=1)
    pairs=PAIRS+[(k,'s0r') for k in METHODS if k!='s0r']
    values={f'{a}-{b}':{'delta_pp':100*(metrics[a]['metrics']['foreground_mIoU']-metrics[b]['metrics']['foreground_mIoU']),
                      'paired_instance_bootstrap_95_interval_pp':(100*np.percentile(scores[a]-scores[b],[2.5,97.5])).tolist()}
            for a,b in pairs}
    write(output/'instance_bootstrap.json',{'instances':len(instances),'draws':2000,'pairs':values,
          'note':'Exploratory unadjusted intervals over object instances, not multi-seed training stability.'})
    return values


def region_statistics(paths,output):
    reference=read_jsonl(PROJECT/'outputs/umd_error_analysis_20261016/per_region.jsonl')
    matrices={k:np.zeros((8,8),np.int64) for k in paths}
    cache,changes={},[]
    for row in reference:
        sid=row['sample_id']
        if cache.get('id')!=sid:
            cache={'id':sid,**{k:read(p/'samples'/sid/'decisions.json')['regions'] for k,p in paths.items()}}
        counts=np.asarray(row['gt_pixel_counts'])
        label=int(counts.argmax())
        pure=label>0 and counts.max()/max(1,counts.sum())>=.9
        choices={k:cache[k][row['region_id']-1]['final_choice'] for k in paths}
        if pure:
            for key,choice in choices.items(): matrices[key][label,NAMES.index(choice)]+=1
        changes.append({'sample_id':sid,'region_id':row['region_id'],'gt_majority':NAMES[label],
                        'purity':float(counts.max()/max(1,counts.sum())),'choices':choices})
    result={}
    for key,cm in matrices.items():
        result[key]={'pure_foreground_regions':int(cm.sum()),'accuracy':float(cm.trace()/cm.sum()),
                     'macro_accuracy':float(np.mean([cm[c,c]/cm[c].sum() for c in range(1,8) if cm[c].sum()])),
                     'per_class_accuracy':{NAMES[c]:float(cm[c,c]/cm[c].sum()) for c in range(1,8) if cm[c].sum()},
                     'confusion_gt_rows_pred_columns':cm.tolist()}
    write(output/'region_statistics.json',{'methods':result,'gt_used_only_after_inference':True})
    write_jsonl(output/'region_changes.jsonl',changes)
    return result


def examples(paths,output):
    per_image={k:{r['sample_id']:r['mean_IoU_present_classes'] for r in read_jsonl(p/'evaluation/per_image.jsonl')}
               for k,p in paths.items()}
    selected=[]
    for method,control in PAIRS:
        delta=[(v-per_image[control][sid],sid) for sid,v in per_image[method].items()]
        for direction in [1,-1]:
            ordered=sorted((d,s) for d,s in delta if direction*d>1e-6)
            if not ordered: continue
            d,sid=ordered[len(ordered)//2]
            selected.append({'method':method,'control':control,'sample_id':sid,'delta':d,
                             'selection':'median improvement' if direction>0 else 'median regression'})
    palette=np.array([[0,0,0],[230,85,65],[65,120,230],[245,190,45],[60,180,95],
                      [195,85,180],[45,190,200],[235,130,45]],np.uint8)
    font=render_font(18)
    manifest={r['sample_id']:r for r in read_jsonl(PROJECT/'data/UMD/manifests/umd_novel_test.jsonl')}
    canvas=Image.new('RGB',(1440,365*len(selected)+65),'white');draw=ImageDraw.Draw(canvas)
    for row_index,item in enumerate(selected):
        sid=item['sample_id'];row=manifest[sid]
        sample=PROJECT/'outputs/umd_novel_test/samples'/sid
        rgb=np.asarray(Image.open(sample/'rgb.png').convert('RGB'))
        crop=read(sample/'foreground/metadata.json')['crop_xyxy']
        gt,_=load_annotation(row['annotation'],tuple(row['size_wh'][::-1]))
        ids=[item['control'],item['method']]
        maps=[None,gt]+[np.asarray(Image.open(paths[k]/'samples'/sid/'result/class_map.png')) for k in ids]
        y=row_index*365
        draw.text((12,y+6),f"{item['method'].upper()} / {item['control'].upper()} | {sid} | image IoU: {item['delta']*100:+.2f} pp",fill='black',font=font)
        for col,(title,labelmap) in enumerate(zip(['RGB','GT']+[k.upper() for k in ids],maps)):
            visual=rgb.copy()
            if labelmap is not None:
                valid=(labelmap>0)&(labelmap<8)
                visual[valid]=.55*visual[valid]+.45*palette[labelmap[valid]]
            im=Image.fromarray(visual).crop(crop);scale=min(348/im.width,295/im.height)
            im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
            x=360*col
            draw.text((x+12,y+34),title,fill='black',font=font)
            canvas.paste(im,(x+(360-im.width)//2,y+64+(295-im.height)//2))
    y=365*len(selected)+15
    for i,name in enumerate(NAMES[1:],1):
        x=12+(i-1)*200;draw.rectangle((x,y,x+20,y+20),fill=tuple(palette[i].tolist()))
        draw.text((x+27,y),name,font=font,fill='black')
    canvas.save(output/'examples.jpg',quality=94)
    write(output/'examples.json',{'cases':selected,'score':'per-image mean GT-present class IoU, not global mIoU'})


def curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(12,4))
    for key in METHODS:
        rows=read_jsonl(ROOT/'runs'/key/'joint/loss.jsonl')
        axes[0].plot([r['step'] for r in rows],[max(r['ce'],1e-7) for r in rows],label=key.upper(),linewidth=1)
    axes[0].set_yscale('log');axes[0].set_ylabel('Classification CE on current training views')
    for key,field in [('a1','invariance'),('b0','composition'),('b1','composition'),('b1','consistency')]:
        rows=read_jsonl(ROOT/'runs'/key/'joint/loss.jsonl')
        axes[1].plot([r['step'] for r in rows],[max(r[field],1e-7) for r in rows],label=f'{key.upper()} {field}',linewidth=1)
    axes[1].set_yscale('log');axes[1].set_ylabel('Auxiliary training objective (not IoU)')
    for ax in axes:
        ax.set_xlabel('Fixed joint-training update');ax.grid(alpha=.2);ax.legend(fontsize=8)
    figure.tight_layout();figure.savefig(output/'training_curves.png',dpi=160);figure.savefig(output/'training_curves.pdf');plt.close(figure)


def training_cost(output):
    result={}
    for key in METHODS:
        folder=ROOT/'runs'/key/'joint'
        status=read(folder/'status.json')
        params=read(folder/'trainable_parameters.json')
        config=read(folder/'config.json')
        assert status['status']=='complete' and status['step']==1000
        views=2 if key.startswith(('a','b')) else 1
        sources=status['step']*config['effective_images_per_step']
        result[key]={'joint_training_seconds':status['training_seconds'],
                     'peak_allocated_mib':status['peak_allocated_mib'],
                     'source_image_presentations':sources,
                     'backbone_forward_passes':sources*views,
                     'lora_parameters':params['lora'],
                     'all_head_parameters':params['all_heads'],
                     'extra_head_parameters':params['extra'],
                     'world_size':read(folder/'environment.json')['world_size']}
    write(output/'training_cost.json',{'methods':result,
          'scope':'Joint-training loop only; excludes model loading, input compilation, shared 300-step head warmup and evaluation. Includes intermediate checkpoint saving. One GPU per method.',
          'shared_warmup_reused':True})
    return result


def timing_repeat_audit(output):
    result={}
    for key in ['s0r','c0','c1']:
        path=ROOT/'timing'/key/'per_image.jsonl'
        if not path.exists(): continue
        rows=read_jsonl(path)
        changed=[r for r in rows if not r['matches_saved_prediction']]
        result[key]={'images':len(rows),'changed_predictions':len(changed),
                     'changed_sample_ids':[r['sample_id'] for r in changed],
                     'changed_regions':sum(sum(a!=b for a,b in zip(r['live_choices'],r['cached_choices'])) for r in changed),
                     'input_file_mismatches':sum(r['live_input_sha256']!=r['cached_input_sha256'] for r in rows),
                     'request_mismatches':sum(not r['request_matches_saved'] for r in rows),
                     'candidate_pixel_disagreements':sum(r['candidate_pixel_disagreements'] for r in rows)}
    write(output/'timing_repeat_audit.json',{'methods':result,
          'note':'Observed repeat differences only; cause not established. Accuracy tables use the original complete test runs, never select between repeated predictions.'})
    return result


def document(metrics,remap,regions,output):
    path=PROJECT/'docs/research/实验结果汇总.md'
    original=path.read_text()
    original=re.sub(r'<!-- ABC_PROGRESS -->.*?<!-- /ABC_PROGRESS -->\n*','',original,flags=re.S)
    original=re.sub(r'## A／B／C：固定预算对照.*?## 首轮基线与已有结果\n\n','',original,flags=re.S)
    original=original.replace('；未执行 A／B／C。','；该首轮仅执行普通 SFT。')
    prefix,body=original.split('| 方法 | UMD 训练图 |',1)
    reference=metrics['s0r']['metrics']['foreground_mIoU']
    winner=max(METHODS,key=lambda k:metrics[k]['metrics']['foreground_mIoU'])
    best=metrics[winner]['metrics']['foreground_mIoU']
    if winner=='s0r':
        headline=f'**本轮普通 SFT 最高：{previous.percent(best)}；A／B／C 尚未带来净提升。**'
    else:
        headline=f"**本轮最高：{LABELS[winner]}，mIoU {previous.percent(best)}，比同条件普通 SFT 高 {100*(best-reference):.2f} 个百分点。**"
    lines=['## A／B／C：固定预算对照','',
           headline,'',
           '相同 76 张训练图、候选、300 步头预热；重新初始化 LoRA，固定联合 1,000 步。新增普通 SFT 单卡对照，统一梯度累计方式；不按 test 选轮数或调参。','',
           '| 方法 | mIoU ↑ | 相对本轮 SFT | gIoU ↑ | P@50 ↑ |',
           '|---|---:|---:|---:|---:|']
    for key in ['s0']+METHODS:
        m=metrics[key]['metrics'];score=previous.percent(m['foreground_mIoU'])
        if key in METHODS and m['foreground_mIoU']==best:score=f'**{score}**'
        lines.append(f"| {LABELS[key]} | {score} | {100*(m['foreground_mIoU']-reference):+.2f} pp | {previous.percent(m['gIoU_present_queries'])} | {previous.percent(m['P_at_50_present_queries'])} |")
    lines+=['','配对机制的净变化（各自对照）：'+ '；'.join(
        f"**{a.upper()}−{b.upper()}：{100*(metrics[a]['metrics']['foreground_mIoU']-metrics[b]['metrics']['foreground_mIoU']):+.2f} pp**" for a,b in PAIRS)+'。','',
        'B0 同时包含几何变体和组成软监督，尚不能拆分二者贡献；B1 的新增一致性未显示提升。C1 未超过普通 SFT。B0 的 **support 仅 8.43%**，功能混淆仍明显。','',
        '| 方法 | grasp | cut | scoop | contain | pound | support | wrap-grasp |',
        '|---|'+'---:|'*7]
    for key in METHODS:
        lines.append('| '+key.upper()+' | '+' | '.join(previous.percent(metrics[key]['metrics']['per_class'][n]['iou']) for n in NAMES[1:])+' |')
    lines+=['',
        '- **A0→A1**：相同编号／颜色／顺序／定义增强，增加区域与功能定义配对及预测一致性。',
        '- **B0→B1**：相同几何变体与组成软监督，增加父子区域的面积加权组成一致性；辅助头不参与测试。',
        '- **C0→C1**：同一新增残差头，从仅目标区域特征扩展到最近邻与整物体特征。两组都保留原三图共享上下文；保持一次骨干前向、零生成 token。',
        '- A／B 每张源图训练两个视图，前向预算为 SFT／C 的两倍；相应机制效果以 A0、B0 为直接对照。[训练成本](../../outputs/valen_76_abc_20261017/comparison/training_cost.json)','']
    timing={k:read(ROOT/'timing'/k/'metrics.json') for k in METHODS if (ROOT/'timing'/k/'metrics.json').exists()}
    if timing:
        lines+=['| 方法 | Valen／图 | RGB→mask／图 | P95 完整耗时 |', '|---|---:|---:|---:|']
        for key,t in timing.items():
            s=t['timing_seconds']
            lines.append(f"| {key.upper()} | {s['valen_seconds']['mean']:.3f} s | {s['end_to_end_seconds']['mean']:.3f} s | {s['end_to_end_seconds']['p95']:.3f} s |")
        lines+=['','同一 RTX 4090、batch=1、两个模型常驻，预热后测相同 200 张 RGB；包括必要预处理、PartGLEE、Valen 与合并，不含加载和结果保存。','']
        repeats=read(output/'timing_repeat_audit.json')['methods']
        if any(t['changed_predictions'] for t in repeats.values()):
            lines+=['计时复跑的预测变化：'+ '；'.join(f"{k.upper()} {t['changed_predictions']}/{t['images']} 张" for k,t in repeats.items())+
                    '。原因尚未确定；主表保持原完整评测输出。[输入与候选核对](../../outputs/valen_76_abc_20261017/comparison/timing_repeat_audit.json)','']
    lines+=['**纯前景区域判断一致率**（GT 纯度 ≥90%，6,538 个候选）：'+
            '；'.join(f"{k.upper()} {regions[k]['accuracy']*100:.2f}%" for k in METHODS)+'。','',
            '**仅作协议敏感性补充**：同一预测将 ladle GT `contain→scoop` 后，mIoU 为 '+
            '；'.join(f"{k.upper()} {remap[k]['foreground_mIoU']*100:.2f}%" for k in METHODS)+'。','',
            '各组均评测全部 2,661 张图；失败数：'+ '；'.join(f"{k.upper()} {sum(metrics[k]['failures'].values())}" for k in METHODS)+'。', '',
            '这是 seed=321 的单次实验；测试集此前已用于误差分析，不称独立盲测。mIoU 是分割指标，不是机器人操作成功率。','',
            '[逐图可切换对比](../../outputs/valen_76_abc_20261017/comparison/index.html) · [改善／退化示例](../../outputs/valen_76_abc_20261017/comparison/examples.jpg) · [训练曲线](../../outputs/valen_76_abc_20261017/comparison/training_curves.png) · [配置](../../configs/valen_76_abc) · [完整指标与审计](../../outputs/valen_76_abc_20261017/comparison/summary.json)','',
            '## 首轮基线与已有结果','']
    path.write_text(prefix+'\n'.join(lines)+'\n| 方法 | UMD 训练图 |'+body,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-gallery',action='store_true')
    parser.add_argument('--reuse-statistics',action='store_true')
    args=parser.parse_args()
    output=ROOT/'comparison';output.mkdir(exist_ok=True)
    paths=roots();metrics=load_metrics(paths)
    if args.reuse_statistics:
        remap=read(output/'ladle_remap_metrics.json')['results']
        regions=read(output/'region_statistics.json')['methods']
    else:
        audit(paths,output)
        paired_bootstrap(metrics,output)
        remap=previous.protocol_rescore(paths,output)
        regions=region_statistics(paths,output)
    if not args.skip_gallery:
        previous.RUNS={k:(LABELS[k],p) for k,p in paths.items()}
        previous.gallery(paths,metrics,output)
        (output/'index.html').write_text((Path(__file__).parent/'gallery.html').read_text(),encoding='utf-8')
        shutil.copytree(Path(__file__).parent/'assets',output/'fonts',dirs_exist_ok=True)
        examples(paths,output)
    curves(output)
    costs=training_cost(output)
    repeats=timing_repeat_audit(output)
    document(metrics,remap,regions,output)
    write(output/'summary.json',{'methods':{k:m['metrics'] for k,m in metrics.items()},
          'failures':{k:m['failures'] for k,m in metrics.items()},'fixed_final_step':1000,
          'training_cost':costs,
          'timing_repeat_audit':repeats,
          'configuration_directory':str(PROJECT/'configs/valen_76_abc')})


if __name__=='__main__': main()
