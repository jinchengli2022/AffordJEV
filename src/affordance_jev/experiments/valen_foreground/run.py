"""Fresh staged smoke/full inference, with explicit GPU and Conda executables."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

from ...common import read, write, sha
from ...paths import PROJECT, PARTGLEE_ROOT, PARTGLEE_WEIGHTS, CLIP_MODEL, VALEN_MODEL, conda_python
from .data import initialize, prepare_phase


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].strip().split()[0] != 'Z'
    except FileNotFoundError:
        return False


def run_phase(root, phase, gpus, executables):
    dest = prepare_phase(root, phase)
    if (dest/'run.json').exists() and read(dest/'run.json')['status'] == 'complete':
        return
    started = time.perf_counter()
    phase_gpus = gpus[:1] if phase == 'smoke' else gpus
    write(dest/'source_hashes.json', {str(p.relative_to(PROJECT)): sha(p)
        for p in (PROJECT/'src/affordance_jev').rglob('*.py')})
    for stage in ['objects', 'foreground', 'parts', 'functions']:
        processes = []
        for index, gpu in enumerate(phase_gpus):
            finished = dest/'workers'/f'{stage}_{index}.json'
            if finished.exists() and read(finished)['status'] == 'complete':
                continue
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(PROJECT/'src'),
                   'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'}
            python = executables['partglee' if stage in ['objects', 'parts'] else 'valen']
            cmd = [str(python), '-m', 'affordance_jev.experiments.valen_foreground.workers',
                   '--root', str(dest), '--stage', stage, '--shard-index', str(index),
                   '--shard-count', str(len(phase_gpus))]
            log = dest/'logs'/f'{stage}_{index}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as stream:
                proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
            processes.append((index, proc))
            write(dest/'processes'/f'{stage}_{index}.json', {'pid': proc.pid, 'gpu': gpu, 'command': cmd,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'log': str(log)})
        try:
            while any(proc.poll() is None for _, proc in processes):
                for _, proc in processes:
                    if proc.poll() not in (None, 0):
                        raise RuntimeError(f'{phase}/{stage} worker exited {proc.returncode}; inspect logs')
                progress = [read(p) for p in sorted((dest/'progress').glob(f'{stage}_*.json'))]
                write(root/'status.json', {'status': 'running', 'phase': phase, 'stage': stage,
                    'images_done': sum(p['images_done'] for p in progress),
                    'images_total': read(dest/'protocol.json')['images'], 'workers': progress})
                time.sleep(2)
        except BaseException:
            for _, proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            for _, proc in processes:
                proc.wait()
            raise
        for index in range(len(phase_gpus)):
            marker = dest/'workers'/f'{stage}_{index}.json'
            if not marker.exists() or read(marker)['status'] != 'complete':
                raise RuntimeError(f'Missing completed worker marker: {marker}')
            counts = read(marker)['counts']
            if phase == 'smoke' and (counts.get('failed', 0) or counts.get('error', 0)):
                raise RuntimeError(f'Smoke implementation failure: {stage}: {counts}')
        print(f'COMPLETE {phase}/{stage}', flush=True)
    from .report import evaluate, gallery, update_document
    result = evaluate(dest, root)
    gallery(dest, root)
    write(dest/'run.json', {'status': 'complete', 'images': result['metrics']['images'],
                          'queries': result['metrics']['queries'], 'phase_wall_seconds': time.perf_counter()-started})
    if phase == 'full':
        update_document(root, result)
    print(f'COMPLETE {phase} metrics={result["metrics"]["mIoU_known_operations"]:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--definitions', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--partglee-root', default=str(PARTGLEE_ROOT))
    parser.add_argument('--partglee-weights', default=str(PARTGLEE_WEIGHTS))
    parser.add_argument('--clip-model', default=str(CLIP_MODEL))
    parser.add_argument('--valen-model', default=str(VALEN_MODEL))
    parser.add_argument('--partglee-python', default=str(conda_python('partglee')))
    parser.add_argument('--valen-python', default=str(conda_python('valen-4b-sft')))
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--phase', choices=['smoke', 'full'], default='full')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    lock = root/'supervisor.json'
    if lock.exists() and alive(read(lock)['pid']):
        raise RuntimeError('This experiment already has a running supervisor')
    models = {key: str(Path(getattr(args, key)).resolve(strict=True)) for key in
              ['partglee_root', 'partglee_weights', 'clip_model', 'valen_model']}
    gpus = args.gpus.split(',')
    assert len(gpus) == len(set(gpus)) and all(g.isdigit() for g in gpus)
    initialize(root, args.config, args.manifest, args.queries, args.definitions, models, Path(args.baseline).resolve())
    execution = {'gpus': gpus, 'executables': {'partglee': args.partglee_python, 'valen': args.valen_python}}
    if (root/'execution.json').exists() and read(root/'execution.json') != execution:
        raise ValueError('Cannot change shard assignment while resuming this run')
    write(root/'execution.json', execution)
    write(lock, {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        run_phase(root, 'smoke', gpus, execution['executables'])
        if args.phase == 'full':
            run_phase(root, 'full', gpus, execution['executables'])
        write(root/'status.json', {'status': 'complete' if args.phase == 'full' else 'smoke_complete',
                                 'phase': args.phase, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        write(root/'status.json', {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
