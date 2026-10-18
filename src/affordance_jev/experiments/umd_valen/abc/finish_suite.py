"""Finish already-authorized reports and timing after every training job exits."""
import os
from pathlib import Path
import subprocess
import sys
import time

from ....paths import training_config
from ....common import read,write
from ..data import PROJECT

ROOT=PROJECT/'outputs/valen_76_abc_20261017'
METHODS=['s0r','a0','a1','b0','b1','c0','c1']


def main():
    state=ROOT/'finish_status.json'
    if state.exists(): raise FileExistsError(state)
    write(state,{'status':'waiting','pid':os.getpid(),'waiting_for':METHODS})
    while True:
        states={k:read(ROOT/'suite'/f'{k}.json') if (ROOT/'suite'/f'{k}.json').exists() else {'status':'pending'} for k in METHODS}
        failed=[k for k,v in states.items() if v['status']=='failed']
        if failed:
            write(state,{'status':'failed','stage':'wait','failed_methods':failed})
            raise RuntimeError(f'Experiment failed: {failed}; no automatic restart')
        if all(v['status']=='complete' for v in states.values()): break
        time.sleep(30)
    def execute(stage,args):
        write(state,{'status':'running','stage':stage,'pid':os.getpid()})
        log=ROOT/f'{stage}.log'
        with log.open('w') as stream:
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'6','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1',
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            result=subprocess.run([sys.executable,'-m',*args],env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write(state,{'status':'failed','stage':stage,'exit_code':result.returncode,'log':str(log)})
            raise RuntimeError(f'{stage} failed: {log}')
    # C changes inference modules; A/B's extra constraints/heads are training-only.
    for key in ['s0r','c0','c1']:
        config=training_config(PROJECT/'configs/valen_76_abc'/f'{key}.json')
        execute(f'timing_{key}',['affordance_jev.experiments.umd_valen.benchmark',
            '--root',str(ROOT/'tests'/key),'--adapter',str(Path(config['checkpoint_dir'])/'final'),
            '--output',str(ROOT/'timing'/key),'--samples','200'])
    execute('report',['affordance_jev.experiments.umd_valen.abc.report'])
    write(state,{'status':'complete','methods':METHODS,'summary':str(PROJECT/'docs/research/实验结果汇总.md')})


if __name__=='__main__': main()
