"""Fresh shared foreground/parts, then five functional Valen variants."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

from ...common import read, write, sha
from ...paths import PROJECT, PARTGLEE_ROOT, PARTGLEE_WEIGHTS, CLIP_MODEL, VALEN_MODEL, conda_python
from .data import initialize, prepare_phase
from .run import alive
from .suite_report import METHODS, evaluate, gallery, update_document


def utc():
    return datetime.now(timezone.utc).isoformat()


def run_stage(experiment, phase, stage, gpus, execution, adapter):
    root = experiment/phase
    method = stage.removeprefix('functions_') if stage.startswith('functions_') else None
    worker_stage = 'functions' if method else stage
    processes = []
    try:
        for index, gpu in enumerate(gpus):
            marker = root/'workers'/f'{stage}_{index}.json'
            if marker.exists() and read(marker)['status'] == 'complete':
                continue
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(PROJECT/'src'),
                   'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'}
            python = execution['partglee' if stage in ['objects', 'parts'] else 'valen']
            cmd = [python, '-m', 'affordance_jev.experiments.valen_foreground.workers',
                   '--root', str(root), '--stage', worker_stage, '--shard-index', str(index),
                   '--shard-count', str(len(gpus))]
            if method:
                cmd += ['--method', method]
            if method == 'sft':
                cmd += ['--adapter', str(adapter)]
            log = root/'logs'/f'{stage}_{index}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as stream:
                process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
            processes.append(process)
            write(root/'processes'/f'{stage}_{index}.json', {'pid': process.pid, 'gpu': gpu,
                'command': cmd, 'log': str(log), 'started_utc': utc()})
        while True:
            codes = [p.poll() for p in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'{phase}/{stage}: worker exited unexpectedly; inspect logs')
            progress = [read(p) for p in sorted((root/'progress').glob(f'{stage}_*.json'))]
            write(experiment/'status.json', {'status': 'running', 'phase': phase, 'stage': stage,
                'images_done': sum(p['images_done'] for p in progress),
                'images_total': read(root/'protocol.json')['images'], 'workers': progress, 'updated_utc': utc()})
            if all(code is not None for code in codes):
                break
            time.sleep(30)
        for index in range(len(gpus)):
            state = read(root/'workers'/f'{stage}_{index}.json')
            assert state['status'] == 'complete'
            if phase == 'smoke' and any(state['counts'].get(k, 0) for k in ['failed', 'error']):
                raise RuntimeError(f'Smoke failed: {stage}: {state["counts"]}')
        print(f'COMPLETE {phase}/{stage}', flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--definitions', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--partglee-root', default=str(PARTGLEE_ROOT))
    parser.add_argument('--partglee-weights', default=str(PARTGLEE_WEIGHTS))
    parser.add_argument('--clip-model', default=str(CLIP_MODEL))
    parser.add_argument('--valen-model', default=str(VALEN_MODEL))
    parser.add_argument('--partglee-python', default=str(conda_python('partglee')))
    parser.add_argument('--valen-python', default=str(conda_python('valen-4b-sft')))
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    parser.add_argument('--phase', choices=['smoke', 'full'], default='full')
    parser.add_argument('--initialize-only', action='store_true')
    args = parser.parse_args()
    root, adapter = Path(args.root).resolve(), Path(args.adapter).resolve(strict=True)
    if (root/'supervisor.json').exists() and alive(read(root/'supervisor.json')['pid']):
        raise RuntimeError('Another supervisor is already running this experiment')
    gpus = args.gpus.split(',')
    assert len(gpus) == len(set(gpus)) and all(x.isdigit() for x in gpus)
    models = {key: str(Path(getattr(args, key)).resolve(strict=True)) for key in
              ['partglee_root', 'partglee_weights', 'clip_model', 'valen_model']}
    initialize(root, args.config, args.manifest, args.queries, args.definitions, models, Path(args.baseline).resolve())
    cfg = read(root/'config.json')
    assert cfg['foreground_selector']['method'] == 'interior_point_union'
    assert 'point_union_run' not in cfg['comparisons'], 'A nine-image reference must not be used for full evaluation'
    frozen = {'methods': list(METHODS), 'gpus': gpus, 'adapter': str(adapter),
        'adapter_sha256': sha(adapter/'adapter/adapter_model.safetensors'), 'head_sha256': sha(adapter/'head.safetensors'),
        'executables': {'partglee': args.partglee_python, 'valen': args.valen_python},
        'fresh_foreground_and_parts': True, 'foreground_model': 'original Valen, never the SFT adapter',
        'new_training': False, 'SFT_training_images': 76, 'functional_rule': 'P(yes)>P(no); unchanged',
        'foreground_minimum_yes_probability': cfg['foreground_selector']['minimum_yes_probability'],
        'threshold_development': 'threshold selected after inspection of the nine-image diagnostic subset'}
    if (root/'suite.json').exists():
        assert read(root/'suite.json') == frozen, 'Cannot change models or shard assignment in an existing run'
    else:
        write(root/'suite.json', frozen)
    if args.initialize_only:
        if not (root/'status.json').exists():
            write(root/'status.json', {'status': 'prepared', 'phase': 'smoke', 'stage': 'pending', 'updated_utc': utc()})
        update_document(root)
        return
    write(root/'supervisor.json', {'pid': os.getpid(), 'started_utc': utc(), 'module': __name__})
    try:
        for phase in ['smoke'] + (['full'] if args.phase == 'full' else []):
            destination = prepare_phase(root, phase)
            if (destination/'run.json').exists() and read(destination/'run.json')['status'] == 'complete':
                continue
            started = time.perf_counter()
            current_hashes = {str(p.relative_to(PROJECT)): sha(p) for p in (PROJECT/'src/affordance_jev').rglob('*.py')}
            if (destination/'source_hashes.json').exists():
                assert read(destination/'source_hashes.json') == current_hashes, 'Source changed while resuming'
            else:
                write(destination/'source_hashes.json', current_hashes)
            phase_gpus = gpus[:1] if phase == 'smoke' else gpus
            for stage in ['objects', 'foreground', 'parts'] + [f'functions_{m}' for m in METHODS]:
                run_stage(root, phase, stage, phase_gpus, frozen['executables'], adapter)
            write(root/'status.json', {'status': 'running', 'phase': phase, 'stage': 'evaluation', 'updated_utc': utc()})
            evaluate(destination, root)
            write(root/'status.json', {'status': 'running', 'phase': phase, 'stage': 'visualization', 'updated_utc': utc()})
            gallery(destination, root)
            write(destination/'run.json', {'status': 'complete', 'images': read(destination/'protocol.json')['images'],
                'queries_per_method': read(destination/'protocol.json')['queries'], 'methods': list(METHODS),
                'phase_wall_seconds': time.perf_counter()-started, 'finished_utc': utc()})
            print(f'COMPLETE {phase}/all_five_methods', flush=True)
            if phase == 'full':
                update_document(root, read(destination/'evaluation/metrics.json'))
        write(root/'status.json', {'status': 'complete' if args.phase == 'full' else 'smoke_complete',
                                  'phase': args.phase, 'finished_utc': utc()})
    except BaseException as error:
        state = read(root/'status.json') if (root/'status.json').exists() else {}
        write(root/'status.json', {**state, 'status': 'failed', 'error': repr(error), 'finished_utc': utc()})
        raise


if __name__ == '__main__':
    main()
