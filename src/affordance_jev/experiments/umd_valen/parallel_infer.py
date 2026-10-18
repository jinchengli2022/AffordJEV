"""Disjoint batch-one inference shards of the same fixed checkpoint."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time
from ...common import read,write,read_jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--adapter')
    p.add_argument('--gpus',default='2,3,4,5')
    a=p.parse_args()
    root=Path(a.root).resolve()
    gpus=a.gpus.split(',')
    state=read(root/'run.json')
    jobs=read_jsonl(root/'model_jobs.jsonl')
    shards=[jobs[i::len(gpus)] for i in range(len(gpus))]
    ids=[j['sample_id'] for shard in shards for j in shard]
    assert len(ids)==len(set(ids))==len(jobs)
    write(root/'run.json',{**state,'status':'running','inference_gpus':gpus})
    children=[]
    started=time.perf_counter()
    for rank,gpu in enumerate(gpus):
        log=(root/f'infer_{rank}.log').open('w')
        cmd=[sys.executable,'-m','affordance_jev.experiments.umd_valen.infer','--root',str(root),
             '--shard-index',str(rank),'--shard-count',str(len(gpus)),'--merge-workers','4']
        if a.adapter: cmd.extend(['--adapter',a.adapter])
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONUNBUFFERED':'1'}
        children.append((subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    while any(child.poll() is None for child,_ in children):
        time.sleep(15)
        status=[]
        for rank in range(len(gpus)):
            path=root/f'progress_{rank}.json'
            if path.exists():
                v=read(path)
                status.append(f"{rank}:{v['stage']} {v['completed']}/{v['total']}")
        print('SHARDS '+'; '.join(status),flush=True)
    codes=[child.returncode for child,_ in children]
    for _,log in children: log.close()
    counts=Counter(read(Path(j['output'])/'run.json')['status'] for j in jobs)
    success=all(code==0 for code in codes) and counts['complete']==len(jobs)
    write(root/'run.json',{**state,'status':'complete' if success else 'failed','counts':dict(counts),
          'inference_gpus':gpus,'batch_per_gpu':1,'shards_disjoint':True,'adapter':a.adapter,
          'batch_wall_seconds':time.perf_counter()-started,'worker_exit_codes':codes})
    print(f'COMPLETE {dict(counts)}',flush=True)
    if not success: raise SystemExit(1)


if __name__=='__main__': main()
