"""Run or finish the seven-method experiment with explicit GPU ownership.

No historical predictions are imported. --continue adopts only processes belonging
to this new run; it does not restart failed training or change experimental settings.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from ...common import read, write, read_jsonl, write_jsonl, sha
from ...paths import PROJECT, conda_python


def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().rsplit(')',1)[1].strip().split()[0]!='Z'
    except FileNotFoundError:
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--continue',dest='resume',action='store_true')
    args=parser.parse_args()
    root=Path(args.root).resolve()
    supervisor=root/'supervisor.json'
    if supervisor.exists():
        previous=read(supervisor).get('pid')
        if previous and previous!=os.getpid() and alive(previous):
            raise RuntimeError(f'This run already has a live supervisor: PID {previous}')
    if not args.resume:
        from .data import initialize
        initialize(root)
    protocol=read(root/'protocol.json')
    for split in ['train','test']:
        assert sha(root/f'{split}_queries.jsonl')==protocol['counts'][split]['queries_sha256']
    (root/'logs').mkdir(exist_ok=True)
    write(supervisor,{'pid':os.getpid(),'root':str(root)})
    registry_path=root/'processes.json'
    registry=read(registry_path) if registry_path.exists() else {}
    children={}

    def launch(name,environment,module,arguments,gpus):
        if name in registry: return
        env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),
             'PYTHONPATH':str(PROJECT/'src')+os.pathsep+str(PROJECT.parent/'2d_affordance/src'),
             'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
             'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4',
             'NCCL_P2P_DISABLE':'1','NCCL_IB_DISABLE':'1'}
        command=[str(conda_python(environment)),'-m',module,*map(str,arguments)]
        log=root/f'logs/{name}.log'
        with log.open('w') as stream:
            p=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children[name]=p
        registry[name]={'pid':p.pid,'gpus':gpus,'command':command,'log':str(log),
                        'started_utc':datetime.now(timezone.utc).isoformat()}
        write(registry_path,registry)
        print(f'START {name} pid={p.pid} gpu={gpus}',flush=True)

    def stopped(name):
        if name not in registry: return False
        if name in children:
            code=children[name].poll()
            if code is None: return False
            if code: raise RuntimeError(f'{name} exited {code}; see {registry[name]["log"]}')
        return not alive(registry[name]['pid'])

    def ensure_complete(name,path):
        if not stopped(name): return False
        if not path.exists() or read(path).get('status')!='complete':
            raise RuntimeError(f'{name} stopped without a completed result: {path}')
        return True

    def candidate_done(split):
        jobs=read_jsonl(root/f'candidates_{split}/model_jobs.jsonl')
        statuses=Counter(read(Path(j['output'])/'run.json')['status'] for j in jobs)
        if statuses.get('pending',0):
            raise RuntimeError(f'Incomplete PartGLEE {split}: {dict(statuses)}')
        write(root/f'candidates_{split}/run.json',{'status':'complete','counts':dict(statuses)})
        return statuses

    if not args.resume:
        launch('partglee_train','partglee','affordance_jev.segmentation.partglee',
               ['--jobs',root/'candidates_train/model_jobs.jsonl'],[0])
        jobs=read_jsonl(root/'candidates_test/model_jobs.jsonl')
        for i,gpu in enumerate([1,2,3]):
            path=root/f'candidates_test/shards/{i}/jobs.jsonl'
            write_jsonl(path,jobs[i::3])
            launch(f'partglee_test_{i}','partglee','affordance_jev.segmentation.partglee',['--jobs',path],[gpu])
        launch('ooal','ooal-umd','affordance_jev.baselines.ooal',[
            'infer','--config',PROJECT/'configs/ooal_umd.json','--checkpoint',
            PROJECT/'checkpoints/ooal_umd_novel_20261016/iteration_020000.pth',
            '--manifest',root/'test_manifest.jsonl','--output',root/'methods/ooal'],[6])
        for i,gpu in enumerate([4,5]):
            launch(f'r1_{i}','affordance-r1-portable','affordance_jev.experiments.conditional_umd.infer',
                   ['r1','--root',root,'--shard-index',i,'--shard-count',2],[gpu])
    source_hashes={str(p.relative_to(PROJECT)):sha(p) for folder in ['src/affordance_jev','configs']
                   for p in (PROJECT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json']}
    write(root/'source_hashes.json',source_hashes)
    checkpoint=PROJECT/'checkpoints'/root.name
    state={'status':'running','pid':os.getpid(),'root':str(root)}
    try:
        while True:
            if stopped('partglee_train') and not (root/'training/config.json').exists():
                candidate_done('train')
                from .training import prepare
                prepare(root)
            if (root/'training/config.json').exists():
                launch('sft_warmup','valen-4b-sft','affordance_jev.experiments.umd_valen.train',
                       ['--config',root/'training/config.json','--stage','warmup'],[0])
            warm=ensure_complete('sft_warmup',root/'training/warmup/status.json')
            ooal=ensure_complete('ooal',root/'methods/ooal/run.json')
            if warm and ooal:
                common=['--standalone','--nproc_per_node=2','-m','affordance_jev.experiments.umd_valen.train',
                        '--config',root/'training/config.json','--stage','joint']
                launch('sft_smoke','valen-4b-sft','torch.distributed.run',common+['--smoke-steps',2],[0,6])
                if ensure_complete('sft_smoke',root/'training/smoke/status.json'):
                    launch('sft_joint','valen-4b-sft','torch.distributed.run',common,[0,6])
            joint=ensure_complete('sft_joint',root/'training/joint/status.json')
            if joint:
                for i,gpu in enumerate([0,6]):
                    launch(f'sft_infer_{i}','valen-4b-sft','affordance_jev.experiments.conditional_umd.infer',
                        ['valen','--root',root,'--methods','sft','--adapter',checkpoint/'final',
                         '--shard-index',i,'--shard-count',2,'--wait-candidates'],[gpu])
            # PartGLEE uses about 4 GiB and spends much of its time saving artifacts.
            # The verified Valen inference footprint fits alongside it on a 24 GiB card.
            # This streams fresh candidates instead of waiting for R1 to release a GPU.
            for i,gpu in enumerate([1,2]):
                launch(f'valen_{i}','valen-4b-sft','affordance_jev.experiments.conditional_umd.infer',
                       ['valen','--root',root,'--shard-index',i,'--shard-count',2,'--wait-candidates'],[gpu])
            r1_done=all(ensure_complete(f'r1_{i}',root/f'methods/r1/worker_{i}.json') for i in range(2))
            segmentation_done=all(stopped(f'partglee_test_{i}') for i in range(3))
            base_done=all(stopped(f'valen_{i}') for i in range(2))
            sft_done=all(stopped(f'sft_infer_{i}') for i in range(2))
            if segmentation_done and base_done and sft_done and r1_done:
                candidate_done('test')
                for method in ['fill','outline','definition','centroid','sft']:
                    for i in range(2):
                        assert read(root/f'methods/{method}/worker_{i}.json')['status']=='complete'
                state.update(status='evaluating')
                write(root/'status.json',state)
                from .report import refresh,gallery
                results=refresh(root)
                if len(results)!=7: raise RuntimeError('Incomplete seven-method evaluation')
                gallery(root)
                state.update(status='complete',results={k:v['metrics'] for k,v in results.items()})
                write(root/'status.json',state)
                print('COMPLETE: seven methods, 5232 queries each; metrics, viewer and document saved.',flush=True)
                break
            state.update(updated_utc=datetime.now(timezone.utc).isoformat(),
                stages={name:('running' if alive(info['pid']) else 'stopped') for name,info in registry.items()})
            for stage in ['warmup','smoke','joint']:
                p=root/f'training/{stage}/status.json'
                if p.exists(): state[f'sft_{stage}']=read(p)
            write(root/'status.json',state)
            time.sleep(15)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',
                     note='No automatic restart or parameter changes. Independent existing workers may still be running.')
        write(root/'status.json',state)
        traceback.print_exc()
        raise


if __name__=='__main__': main()
