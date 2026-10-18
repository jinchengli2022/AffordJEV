"""One GPU per fixed method: train, infer all test images, then evaluate."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time

from ....paths import training_config
from ....common import read,write,sha
from ..data import PROJECT


def run_method(method,gpu,root):
    config_path=PROJECT/'configs/valen_76_abc'/f'{method}.json'
    config=training_config(config_path)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4'}
    status_path=root/'suite'/f'{method}.json'
    def stage(name,arguments):
        log=root/'suite'/f'{method}_{name}.log'
        write(status_path,{'method':method,'gpu':gpu,'stage':name,'status':'running','log':str(log)})
        with log.open('w') as stream:
            child=subprocess.Popen([sys.executable,'-m',*arguments],env=env,stdout=stream,stderr=subprocess.STDOUT)
            write(status_path,{'method':method,'gpu':gpu,'stage':name,'status':'running','pid':child.pid,'log':str(log)})
            code=child.wait()
        if code:
            write(status_path,{'method':method,'gpu':gpu,'stage':name,'status':'failed','exit_code':code,'log':str(log)})
            raise RuntimeError(f'{method} {name} failed: {log}')
    stage('train',['affordance_jev.experiments.umd_valen.abc.train','--config',str(config_path)])
    checkpoint=Path(config['checkpoint_dir'])/'final'
    state=read(checkpoint/'status.json')
    assert state['step']==state['total']==1000
    dest=root/'tests'/method
    stage('prepare',['affordance_jev.experiments.umd_valen.data','variant',
          '--source',str(PROJECT/'outputs/umd_novel_test'),'--output',str(dest),
          '--manifest',str(PROJECT/'data/UMD/manifests/umd_novel_test.jsonl'),
          '--style','fill','--protocol','dominant'])
    stage('infer',['affordance_jev.experiments.umd_valen.infer','--root',str(dest),'--adapter',str(checkpoint)])
    stage('evaluate',['affordance_jev.evaluation.evaluate',
          '--manifest',str(dest/'evaluation_manifest.jsonl'),'--predictions',str(dest),
          '--output',str(dest/'evaluation'),'--vis-limit','0'])
    write(status_path,{'method':method,'gpu':gpu,'stage':'complete','status':'complete',
          'checkpoint':str(checkpoint),'result':str(dest),'config_sha256':sha(config_path)})
    return method


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--methods',default='s0r,a0,a1,b0,b1,c0,c1')
    parser.add_argument('--gpus',default='6,0,1,2,3,4,5')
    args=parser.parse_args()
    root=Path(args.root).resolve()
    (root/'suite').mkdir(exist_ok=True)
    methods,gpus=args.methods.split(','),args.gpus.split(',')
    assert len(methods)==len(gpus)
    if any((root/'suite'/f'{m}.json').exists() for m in methods):
        raise FileExistsError('Refusing to rerun or overwrite an existing suite')
    started=time.time()
    write(root/'suite/status.json',{'status':'running','pid':os.getpid(),'methods':methods,'gpus':gpus,'started_unix':started})
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(methods)) as pool:
        futures={pool.submit(run_method,m,g,root):m for m,g in zip(methods,gpus)}
        for future in concurrent.futures.as_completed(futures):
            try: print('FINISHED '+future.result(),flush=True)
            except Exception as exc:
                failures.append({'method':futures[future],'error':str(exc)})
                print(str(exc),flush=True)
    write(root/'suite/status.json',{'status':'failed' if failures else 'complete','methods':methods,
          'failures':failures,'elapsed_seconds':time.time()-started})
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
