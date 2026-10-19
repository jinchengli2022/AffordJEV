"""Paired full-RGB PartGLEE proposal diagnostic, without pipeline preprocessing.

Inference jobs contain RGB only. GT is opened separately by CPU evaluation.
Keep the model's native top-50 object and top-50 part outputs, including empty,
duplicate, overlapping, and low-confidence masks. GT selects whole masks only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

from ..common import read, read_jsonl, sha, write, write_jsonl


ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ('specific', 'generic')


def now():
    return datetime.now(timezone.utc).isoformat()


def predict_joint(adapter, rgb, vocabulary):
    """One full-image forward, same resizing and threshold as production adapter."""
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        rgb = np.asarray(rgb)
        h, w = rgb.shape[:2]
        resized = adapter.resize.get_transform(rgb).apply_image(rgb)
        tensor = torch.as_tensor(resized.astype('float32').transpose(2, 0, 1))
        images = adapter.model.preprocess_image([{'image': tensor, 'height': h, 'width': w}], 'custom')
        labels = vocabulary['objects'] + vocabulary['parts']
        nobj = len(vocabulary['objects'])
        output = adapter.model.partglee(images, {}, 'custom', batch_name_list=labels, is_train=False,
            custom_object_categories_idx=list(range(nobj)),
            custom_part_categories_idx=list(range(nobj, len(labels))))
        masks = F.interpolate(output['pred_masks'], size=images.tensor.shape[-2:], mode='bilinear', align_corners=False)
        ih, iw = images.image_sizes[0]
        masks = F.interpolate(masks[:, :, :ih, :iw], size=(h, w), mode='bilinear', align_corners=False)[0]
        binary = masks > 0
        cls_score, indices = output['pred_logits'][0].sigmoid().max(-1)
        quality = cls_score * ((masks.sigmoid()*binary).flatten(1).sum(-1) / binary.flatten(1).sum(-1).clamp_min(1))
        result = binary.cpu().numpy(), quality.cpu().tolist(), [labels[i] for i in indices.cpu().tolist()]
        assert result[0].shape == (100, h, w)
        torch.cuda.synchronize()
        return (*result, time.perf_counter()-started)


def worker(args):
    import torch
    from ..segmentation.partglee import PartGLEE
    jobs = read_jsonl(args.jobs)
    out = Path(args.output)
    protocol = read(out/'protocol.json')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    started = time.perf_counter()
    adapter = PartGLEE(ROOT/'third_party/PartGLEE', ROOT/'models/partglee/PartGLEE_SwinL.pth',
                      protocol['vocabularies']['specific'], Path(args.jobs).parent)
    write(Path(args.jobs).parent/'loading.json', {'seconds':time.perf_counter()-started,
          'pid':os.getpid(), 'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'), 'images':len(jobs)})
    for i, row in enumerate(jobs, 1):
        assert set(row) == {'sample_id', 'rgb', 'rgb_sha256', 'size_wh'}
        assert sha(row['rgb']) == row['rgb_sha256']
        rgb = Image.open(row['rgb']).convert('RGB')
        assert list(rgb.size) == row['size_wh']
        # Alternate order to avoid assigning every warm first call to one variant.
        for variant in VARIANTS[::1 if i % 2 else -1]:
            folder = out/'proposals'/variant/row['sample_id']
            masks, quality, labels, elapsed = predict_joint(adapter, rgb, protocol['vocabularies'][variant])
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/'masks.npz', packed=np.packbits(masks.reshape(100,-1), axis=1),
                                shape=np.array(masks.shape, np.int32))
            write(folder/'metadata.json', {'sample_id':row['sample_id'], 'rgb_sha256':row['rgb_sha256'],
                  'variant':variant, 'input':'full original RGB', 'gt_used':False,
                  'local_foreground_selection':False, 'local_proposal_filtering':False,
                  'model_topk':{'objects':50,'parts':50}, 'mask_threshold':'logit > 0',
                  'forward_seconds':elapsed, 'qualities':quality, 'labels':labels,
                  'levels':['object']*50+['part']*50,
                  'areas':masks.sum((1,2)).tolist(), 'mask_file_sha256':sha(folder/'masks.npz')})
            if args.verify_adapter and i == 1:
                adapter.vocabulary = protocol['vocabularies'][variant]
                adapter.labels = adapter.vocabulary['objects']+adapter.vocabulary['parts']
                check, scores, names, _ = adapter.predict(rgb, 'objects')
                equal = np.array_equal(check, masks[:50]) and np.allclose(scores, quality[:50], atol=1e-6) and names == labels[:50]
                write(folder/'adapter_equivalence.json', {'equal':bool(equal), 'objects':50})
                if not equal:
                    raise AssertionError('Joint wrapper changed object predictions')
        print(f"RGB {i}/{len(jobs)} {row['sample_id']} both vocabularies saved", flush=True)
    write(Path(args.jobs).parent/'finished.json', {'status':'complete','images':len(jobs),'finished':now()})


def load_proposals(folder):
    with np.load(folder/'masks.npz', allow_pickle=False) as data:
        shape = tuple(data['shape'])
        return np.unpackbits(data['packed'], axis=1, count=int(np.prod(shape[1:]))).reshape(shape).astype(bool)


def evaluate_sample(row, out, budget):
    from ..datasets.umd import CLASS_NAMES, load_annotation
    from ..evaluation.raw_oracle import pixel_atoms, single_scores, union_oracle
    out = Path(out)
    assert sha(row['annotation']) == row['annotation_sha256']
    gt, kind = load_annotation(row['annotation'], row['size_wh'][::-1])
    assert kind == 'manual'
    results, image_results = [], []
    started = time.perf_counter()
    for variant in VARIANTS:
        folder = out/'proposals'/variant/row['sample_id']
        meta = read(folder/'metadata.json')
        assert meta['rgb_sha256'] == row['rgb_sha256'] and sha(folder/'masks.npz') == meta['mask_file_sha256']
        masks = load_proposals(folder)
        assert masks.shape == (100, *gt.shape)
        membership, counts, totals = pixel_atoms(masks, gt)
        image_results.append({'sample_id':row['sample_id'], 'variant':variant,
            'gt_pixels':int((gt>0).sum()), 'raw_covered_gt_pixels':int((masks.any(0)&(gt>0)).sum()),
            'object_covered_gt_pixels':int((masks[:50].any(0)&(gt>0)).sum()),
            'part_covered_gt_pixels':int((masks[50:].any(0)&(gt>0)).sum()),
            'unique_memberships':len(counts), 'forward_seconds':meta['forward_seconds']})
        for cid in range(1,8):
            if totals[cid] == 0:
                continue
            scores = single_scores(membership, counts, cid, int(totals[cid]))
            oracle = union_oracle(membership, counts, cid, int(totals[cid]), seconds=budget)
            if oracle['lower']+1e-6 < float(scores.max()):
                raise AssertionError('Union oracle worse than a single available mask')
            results.append({'sample_id':row['sample_id'], 'query_id':row['sample_id']+'/'+CLASS_NAMES[cid],
                'variant':variant, 'class_id':cid, 'action':CLASS_NAMES[cid], 'gt_pixels':int(totals[cid]),
                'object_best_single':float(scores[:50].max()), 'object_best_id':int(scores[:50].argmax()),
                'part_best_single':float(scores[50:].max()), 'part_best_id':int(scores[50:].argmax()+50),
                'joint_best_single':float(scores.max()), 'joint_best_id':int(scores.argmax()),
                'union_oracle':oracle})
    write(out/'evaluation/samples'/row['sample_id']/'metrics.json',
          {'images':image_results,'queries':results,'evaluation_seconds':time.perf_counter()-started})
    return image_results, results


def summarize(images, queries):
    def one(qs, ims):
        return {'images':len(ims), 'queries':len(qs),
            **{k:float(np.mean([q[k] for q in qs])) for k in ['object_best_single','part_best_single','joint_best_single']},
            'union_oracle_lower':float(np.mean([q['union_oracle']['lower'] for q in qs])),
            'union_oracle_upper':float(np.mean([q['union_oracle']['upper'] for q in qs])),
            'union_oracle_certified_queries':sum(q['union_oracle']['certified'] for q in qs),
            'union_oracle_max_gap':max(q['union_oracle']['upper']-q['union_oracle']['lower'] for q in qs),
            **{k:sum(r[k] for r in ims)/sum(r['gt_pixels'] for r in ims)
               for k in ['raw_covered_gt_pixels','object_covered_gt_pixels','part_covered_gt_pixels']},
            'mean_forward_seconds':float(np.mean([r['forward_seconds'] for r in ims]))}
    summaries = {}
    for variant in VARIANTS:
        qs = [q for q in queries if q['variant']==variant]
        ims = [r for r in images if r['variant']==variant]
        categories = sorted({r['sample_id'].split('/')[0].rsplit('_',1)[0] for r in ims})
        summaries[variant] = {'overall':one(qs,ims), 'by_category':{
            cat:one([q for q in qs if q['sample_id'].split('/')[0].rsplit('_',1)[0]==cat],
                    [r for r in ims if r['sample_id'].split('/')[0].rsplit('_',1)[0]==cat]) for cat in categories}}
    return summaries


def run(args):
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'Use a fresh experiment output: {out}')
    manifest = ROOT/'data/UMD/manifests/umd_novel_test.jsonl'
    query_file = ROOT/'outputs/umd_given_operation_20261018/test_queries.jsonl'
    rows = read_jsonl(manifest)
    if args.smoke:
        picked = {}
        for row in rows:
            picked.setdefault(row['object_instance'].rsplit('_',1)[0], row)
        rows = list(picked.values())
    valid_ids = {row['sample_id'] for row in rows}
    expected_queries = [q for q in read_jsonl(query_file) if q['sample_id'] in valid_ids]
    out.mkdir(parents=True)
    write(out/'protocol.json', {'started':now(), 'images':len(rows),'queries':len(expected_queries),
        'manifest':str(manifest),'manifest_sha256':sha(manifest), 'queries_sha256':sha(query_file),
        'vocabularies':{v:read(ROOT/('configs/umd_pipeline.json' if v=='specific' else 'configs/umd_pipeline_generic.json'))['vocabulary'] for v in VARIANTS},
        'model':str(ROOT/'models/partglee/PartGLEE_SwinL.pth'), 'budget_per_level':50,
        'input':'full RGB; no foreground selection, greying, cropping, clipping or local proposal filtering',
        'overlaps':'retained; oracle uses actual pixel union, never sums overlapping areas',
        'oracle':'best subset union per known-operation query; certified lower/upper bounds when solver times out',
        'gt':'original manual dominant labels; -1 ignored; no label remapping',
        'oracle_seconds_per_query':args.oracle_seconds,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/affordance_jev/evaluation/raw_oracle.py',ROOT/'src/affordance_jev/segmentation/partglee.py']}})
    write_jsonl(out/'evaluation_manifest.jsonl',rows)
    write_jsonl(out/'queries.jsonl',expected_queries)
    started = time.perf_counter()
    write(out/'status.json',{'status':'running','stage':'inference','images':len(rows),'pid':os.getpid(),'updated':now()})
    processes = []
    devices = args.gpus.split(',')
    try:
        for index, gpu in enumerate(devices):
            jobs = [{k:row[k] for k in ['sample_id','rgb','rgb_sha256','size_wh']} for row in rows[index::len(devices)]]
            if not jobs:
                continue
            path = out/f'workers/{index}/jobs.jsonl'
            write_jsonl(path,jobs)
            command = [sys.executable,'-m','affordance_jev.experiments.raw_candidates','worker',
                       '--output',str(out),'--jobs',str(path)]
            if args.smoke:
                command.append('--verify-adapter')
            env = {**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2',
                   'MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
            log = open(path.parent/'run.log','w')
            proc = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            log.close()
            processes.append(proc)
        failures = [p.wait() for p in processes]
        if any(failures):
            raise RuntimeError(f'Inference worker failures: {failures}')
        inference_seconds = time.perf_counter()-started
        print(f'Inference finished: {len(rows)} paired RGB, {inference_seconds:.1f}s',flush=True)
        write(out/'status.json',{'status':'running','stage':'oracle_evaluation','images':len(rows),
                               'pid':os.getpid(),'inference_seconds':inference_seconds,'updated':now()})
        images, queries = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(evaluate_sample,row,str(out),args.oracle_seconds):row for row in rows}
            for number, future in enumerate(as_completed(futures),1):
                ims, qs = future.result()
                images.extend(ims); queries.extend(qs)
                if number % 100 == 0 or number == len(rows):
                    print(f'Oracle {number}/{len(rows)}, elapsed {time.perf_counter()-started:.1f}s',flush=True)
                    write(out/'status.json',{'status':'running','stage':'oracle_evaluation','completed_images':number,
                        'images':len(rows),'pid':os.getpid(),'updated':now()})
        images.sort(key=lambda r:(r['sample_id'],r['variant']))
        queries.sort(key=lambda r:(r['query_id'],r['variant']))
        for variant in VARIANTS:
            assert {q['query_id'] for q in queries if q['variant']==variant} == {q['query_id'] for q in expected_queries}
        write_jsonl(out/'evaluation/per_image.jsonl',images)
        write_jsonl(out/'evaluation/per_query.jsonl',queries)
        summary = {'status':'complete','images':len(rows),'queries':len(expected_queries),'finished':now(),
            'inference_wall_seconds':inference_seconds,'total_wall_seconds':time.perf_counter()-started,
            'metrics':summarize(images,queries)}
        write(out/'evaluation/metrics.json',summary)
        write(out/'status.json',{k:v for k,v in summary.items() if k!='metrics'})
        print({v:summary['metrics'][v]['overall'] for v in VARIANTS},flush=True)
    except BaseException as exc:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        write(out/'status.json',{'status':'failed','error':str(exc),'updated':now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--output',required=True)
    run_parser.add_argument('--gpus',default='0,1,2,3,4,5,6')
    run_parser.add_argument('--workers',type=int,default=16)
    run_parser.add_argument('--oracle-seconds',type=float,default=3.)
    run_parser.add_argument('--smoke',action='store_true')
    work = sub.add_parser('worker')
    work.add_argument('--output',required=True)
    work.add_argument('--jobs',required=True)
    work.add_argument('--verify-adapter',action='store_true')
    args = parser.parse_args()
    (run if args.mode=='run' else worker)(args)


if __name__ == '__main__':
    main()
