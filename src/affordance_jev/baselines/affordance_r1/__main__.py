"""Run independent GPU shards, then evaluate the frozen predictions."""
import argparse
from collections import Counter
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

from ...common import read, write, read_jsonl, write_jsonl, sha, save_mask
from ...pipeline.run import sample_directory


def worker(args):
    from .predictor import Predictor
    root = Path(args.output)
    jobs = read_jsonl(args.jobs)
    cfg, definitions = read(root/'config_snapshot.json'), read(root/'affordances_snapshot.json')
    started = time.time()
    status_path = root/'workers'/f'{args.worker_id}.json'
    state = {'status': 'loading', 'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
             'images': len(jobs), 'completed': 0, 'started_unix': started}
    write(status_path, state)
    try:
        predictor = Predictor(args.model, args.sam_checkpoint, cfg, definitions)
        write(root/'workers'/f'prompts_{args.worker_id}.json', predictor.prompts)
        errors, durations = Counter(), []
        def results():
            for begin in range(0, len(jobs), cfg.get('image_batch', 1)):
                group = jobs[begin:begin+cfg.get('image_batch', 1)]
                images = []
                for job in group:
                    if set(job) != {'sample_id', 'rgb', 'rgb_sha256', 'size_wh', 'object_category'}:
                        raise ValueError('Unexpected input field; do not pass annotation/category metadata')
                    if sha(job['rgb']) != job['rgb_sha256']:
                        raise ValueError('RGB checksum changed')
                    rgb = Image.open(job['rgb']).convert('RGB')
                    if list(rgb.size) != job['size_wh']:
                        raise ValueError('RGB dimensions changed')
                    images.append(rgb)
                batch = predictor.predict_many(images, [j['object_category'] for j in group])
                for offset, (job, result) in enumerate(zip(group, batch), 1):
                    yield begin+offset, job, result
        for j, job, result in results():
            # Class/category names are explicit inputs; GT masks/presence are excluded.
            if set(job) != {'sample_id', 'rgb', 'rgb_sha256', 'size_wh', 'object_category'}:
                raise ValueError('Unexpected input field; do not pass annotation/category metadata')
            sample = sample_directory(root, job['sample_id'])
            if (sample/'run.json').is_file() and read(sample/'run.json')['status'] == 'complete':
                continue
            sample.mkdir(parents=True, exist_ok=True)
            write(sample/'run.json', {'status': 'running', 'sample_id': job['sample_id']})
            if sha(job['rgb']) != job['rgb_sha256']:
                raise ValueError('RGB checksum changed')
            class_map, logits, records, timing = result
            for record in records:
                if record['error']:
                    errors[record['error'].split(':')[0]] += 1
                save_mask(sample/'raw_masks'/f"{record['action']}.png", logits[record['class_id']-1] > 0)
                save_mask(sample/'result/masks'/f"{record['action']}.png", class_map == record['class_id'])
            (sample/'result').mkdir(exist_ok=True)
            Image.fromarray(class_map).save(sample/'result/class_map.png')
            # Float16 archive is for inspection; merging used float32 before saving.
            np.savez_compressed(sample/'sam_logits.npz', logits=logits.astype(np.float16))
            write(sample/'responses.json', records)
            write(sample/'r1_timings.json', timing)
            write(sample/'input.json', job)
            write(sample/'run.json', {'status': 'complete', 'sample_id': job['sample_id'],
                'query_errors': sum(r['error'] is not None for r in records),
                'overlap_pixels_before_merge': int(((logits > 0).sum(0) > 1).sum())})
            durations.append(timing['rgb_to_all_classes_seconds'])
            state.update(status='running', completed=j, query_errors=dict(errors), updated_unix=time.time(),
                         recent_seconds_per_rgb=float(np.mean(durations[-20:])),
                         elapsed_seconds=time.time()-started)
            write(status_path, state)
            print(f"PROGRESS shard={args.worker_id} {j}/{len(jobs)} {job['sample_id']} "
                  f"seconds={timing['rgb_to_all_classes_seconds']:.2f} errors={sum(errors.values())}", flush=True)
        state.update(status='complete', completed=len(jobs), updated_unix=time.time())
        write(status_path, state)
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', updated_unix=time.time())
        write(status_path, state)
        raise


def run(args):
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Output must be empty: {root}')
    root.mkdir(parents=True)
    rows = read_jsonl(args.manifest)
    if args.smoke:
        # One frame per object category, selected only from identities, never labels.
        first = {}
        for row in rows:
            first.setdefault(row['object_instance'].rsplit('_', 1)[0], row)
        rows = list(first.values())
    if args.limit:
        rows = rows[:args.limit]
    write_jsonl(root/'evaluation_manifest.jsonl', rows)
    cfg = read(args.config)
    write(root/'config_snapshot.json', cfg)
    write(root/'affordances_snapshot.json', read(args.definitions))
    write(root/'environment.json', {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'vllm', 'numpy']})
    source = Path(__file__).parent
    for path in source.glob('*.py'):
        target = root/'provenance'/path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    write(root/'provenance/source_sha256.json', {p.name: sha(p) for p in source.glob('*.py')})
    started = time.time()
    state = {'status': 'running', 'model': str(Path(args.model).resolve()), 'adapter': None,
             'UMD_training_images': 0, 'evaluation_manifest_sha256': sha(root/'evaluation_manifest.jsonl'),
             'source_manifest_sha256': sha(args.manifest), 'images': len(rows), 'queries': len(rows)*7,
             'started_unix': started, 'pid': os.getpid(), 'config': cfg,
             'method': 'Affordance-R1 original checkpoint + SAM2; shared dominant-label protocol adaptation'}
    write(root/'run.json', state)
    gpus = args.gpus.split(',')
    processes, logs = [], []
    try:
        for i, gpu in enumerate(gpus):
            jobs = [{**{k: r[k] for k in ['sample_id', 'rgb', 'rgb_sha256', 'size_wh']},
                     'object_category': r['object_instance'].rsplit('_', 1)[0]} for r in rows[i::len(gpus)]]
            if not jobs:
                continue
            jobpath = root/'workers'/f'jobs_{i}.jsonl'
            write_jsonl(jobpath, jobs)
            log = (root/'workers'/f'{i}.log').open('w')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
            cmd = [sys.executable, '-m', 'affordance_jev.baselines.affordance_r1', 'worker',
                   '--output', str(root), '--jobs', str(jobpath), '--worker-id', str(i),
                   '--model', args.model, '--sam-checkpoint', args.sam_checkpoint]
            processes.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        while any(p.poll() is None for p in processes):
            failed = [p for p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Inference worker failed: {[p.returncode for p in failed]}')
            states = [read(p) for p in (root/'workers').glob('[0-9]*.json')]
            done = sum(s['completed'] for s in states)
            write(root/'progress.json', {'completed': done, 'images': len(rows), 'updated_unix': time.time(),
                                       'elapsed_seconds': time.time()-started, 'workers': states})
            print(f'PROGRESS total={done}/{len(rows)} elapsed={time.time()-started:.0f}s', flush=True)
            time.sleep(20)
        if any(p.returncode for p in processes):
            raise RuntimeError('Inference worker exited unsuccessfully')
        state.update(status='complete', completed_unix=time.time(), inference_wall_seconds=time.time()-started)
        write(root/'run.json', state)
        write(root/'progress.json', {'status': 'complete', 'completed': len(rows), 'images': len(rows),
                                    'updated_unix': time.time(), 'elapsed_seconds': time.time()-started})
        if not args.smoke:
            from ...evaluation.evaluate import evaluate
            summary = evaluate(root/'evaluation_manifest.jsonl', root, root/'evaluation', vis_limit=0)
            summary['protocol'].pop('ground_truth_used_for_model_input', None)
            summary['protocol'].update(
                image_input_modality='RGB only; no depth or mask input',
                model_inputs=['RGB', 'known UMD object category', 'operation name'],
                known_object_category_provided=True, ground_truth_masks_used_for_model_input=False,
                ground_truth_function_presence_used_for_model_input=False,
                prompt='Verbatim official template and UMD category/action query',
                query_selection='All seven classes on every image',
                overlapping_masks='SAM2 maximum positive pixel logit; ties lower class ID')
            write(root/'evaluation/metrics.json', summary)
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', ended_unix=time.time())
        write(root/'run.json', state)
        raise
    finally:
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'worker'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--config')
    parser.add_argument('--definitions')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--jobs')
    parser.add_argument('--worker-id')
    args = parser.parse_args()
    (worker if args.command == 'worker' else run)(args)


if __name__ == '__main__':
    main()
