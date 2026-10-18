"""Frozen-mask ablation: one unchanged RGB/mask context and one centroid image per query."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from ...common import read, write, read_jsonl, write_jsonl, sha, load_mask
from ...decision.regions import request_for
from ...paths import PROJECT, VALEN_MODEL, PARTGLEE_ROOT, PARTGLEE_WEIGHTS
from ...pipeline.artifacts import verify_prepared
from ...pipeline.run import sample_directory
from .data import empty_output, link


DOT_RADIUS = 3  # Original crop pixels; fixed before evaluation, not fitted to GT.


def centroid(mask, crop):
    """Arithmetic mass centroid, without projecting it into the mask."""
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Cannot compute the centroid of an empty mask')
    cx, cy = float(x.mean()), float(y.mean())
    px, py = int(np.floor(cx + .5)), int(np.floor(cy + .5))
    x0, y0, x1, y1 = crop
    if not (x0 <= px < x1 and y0 <= py < y1):
        raise ValueError('Candidate centroid is outside the predicted object crop')
    return {'xy_original': [cx, cy], 'xy_crop': [cx-x0, cy-y0],
            'drawn_xy_original': [px, py], 'drawn_xy_crop': [px-x0, py-y0],
            'inside_own_mask': bool(mask[py, px]), 'pixels': len(x),
            'definition': 'mean of all mask pixel x/y indices; nearest-pixel drawing; no projection'}


def point_image(rgb_crop, geometry):
    """One red point with a white rim; no region fill, outline, or numeric annotation."""
    image = rgb_crop.copy()
    x, y = geometry['drawn_xy_crop']
    ImageDraw.Draw(image).ellipse((x-DOT_RADIUS, y-DOT_RADIUS, x+DOT_RADIUS, y+DOT_RADIUS),
                                 fill=(255, 0, 0), outline=(255, 255, 255), width=1)
    return image


def centroid_request(region_id, definitions):
    request = request_for([{'id': region_id}], definitions)
    content = request['state']['messages'][0]['content']
    content[0]['text'] = (
        'Classify the visible functional part indicated by the red point in the third image. '
        'The first image is an unpainted RGB crop; the second is the same crop with numbered masks. '
        'Use these first two images as appearance and candidate-location context. '
        'Colors and numbers are annotations, not physical appearance. ' + definitions['decision_rules'])
    content[1]['image_url']['url'] = 'rgb_crop.png'
    content[2]['image_url']['url'] = 'numbered_crop.png'
    content[3]['text'] = (
        f'The third image is the unpainted RGB crop with exactly one red point at the arithmetic '
        f'centroid of candidate region {region_id}. Judge the visible functional part at this point; '
        'the point is only a location marker. There is no mask fill or boundary in this third image.')
    content[4]['image_url']['url'] = 'centroid.png'
    request['questions'][f'region_{region_id:02}']['instructions'] = (
        f'For candidate region {region_id}, select exactly one role of the visible functional part '
        'indicated by the red point in the third image, using the supplied definitions. '
        'Ignore the marker color and inspect the unpainted RGB for appearance. Use none when no role fits.')
    return request


def render_queries(sample, rgb_crop, masks, crop, definitions, candidates, save_audit=True):
    """Shared by saved evaluation inputs and live timing. No annotation argument."""
    sample = Path(sample)
    rows = []
    for candidate, mask in zip(candidates, masks):
        region_id = candidate['id']
        relative = f'centroid_queries/{region_id:02}'
        query = sample / relative
        query.mkdir(parents=True, exist_ok=True)
        for name, source in [('rgb_crop.png', sample/'foreground/rgb_crop.png'),
                             ('numbered_crop.png', sample/'regions/numbered_crop.png')]:
            if not (query/name).is_symlink():
                link(source, query/name)
        geometry = centroid(mask, crop)
        point_image(rgb_crop, geometry).save(query/'centroid.png')
        row = {'region_id': region_id, 'directory': relative, **geometry}
        if save_audit:
            write(query/'valen_request.json', centroid_request(region_id, definitions))
            names = ['rgb_crop.png', 'numbered_crop.png', 'centroid.png', 'valen_request.json']
            write(query/'prepared_assets.json', {name: sha(query/name) for name in names})
            row.update(request_sha256=sha(query/'valen_request.json'),
                       point_image_sha256=sha(query/'centroid.png'))
        rows.append(row)
    return rows


def prepare_one(row, source, output, config, definitions):
    origin = sample_directory(source, row['sample_id'])
    sample = sample_directory(output, row['sample_id'])
    verify_prepared(origin)
    if read(origin/'run.json')['status'] != 'complete':
        raise ValueError(f'Incomplete baseline: {origin}')
    if read(origin/'affordances_snapshot.json') != definitions:
        raise ValueError('Class definitions must be identical to the original fill baseline')
    sample.mkdir(parents=True)
    for name in ['rgb.png', 'foreground', 'candidates.json', 'partglee_runtime.json', 'prepare_runtime.json']:
        link(origin/name, sample/name)
    for name in ['masks', 'label_map.png', 'unassigned_foreground.png', 'numbered_crop.png']:
        link(origin/'regions'/name, sample/'regions'/name)
    candidates = read(sample/'candidates.json')['candidates']
    masks = [load_mask(sample/c['mask']) for c in candidates]
    crop = read(sample/'foreground/metadata.json')['crop_xyxy']
    rgb_crop = Image.open(sample/'foreground/rgb_crop.png').convert('RGB')
    queries = render_queries(sample, rgb_crop, masks, crop, definitions, candidates)
    for query in queries:
        q = sample/query['directory']
        assert sha(q/'rgb_crop.png') == sha(origin/'foreground/rgb_crop.png')
        assert sha(q/'numbered_crop.png') == sha(origin/'regions/numbered_crop.png')
    write(sample/'centroids.json', {'regions': queries, 'dot_radius_pixels': DOT_RADIUS,
                                   'uses_ground_truth': False, 'outside_centroids_moved': False})
    # This is an explicit aggregation manifest, never submitted as one model request.
    write(sample/'valen_request.json', {'mode': 'independent_single_centroid_queries',
          'requests': [{'region_id': q['region_id'], 'request': q['directory']+'/valen_request.json',
                        'sha256': q['request_sha256']} for q in queries]})
    write(sample/'config_snapshot.json', config)
    write(sample/'affordances_snapshot.json', definitions)
    assets = ['rgb.png', 'foreground/object_mask.png', 'foreground/rgb_crop.png',
              'regions/numbered_crop.png', 'regions/label_map.png', 'candidates.json', 'centroids.json',
              'config_snapshot.json', 'affordances_snapshot.json', 'valen_request.json']
    assets += [c['mask'] for c in candidates]
    assets += [q['directory']+'/'+name for q in queries for name in ['centroid.png','valen_request.json']]
    write(sample/'prepared_assets.json', {name: sha(sample/name) for name in assets})
    write(sample/'run.json', {'status': 'prepared', 'reference_annotations_used': False})
    job = read(origin/'job.json')
    job.update(output=str(sample), partglee_root=str(PARTGLEE_ROOT),
               partglee_weights=str(PARTGLEE_WEIGHTS), valen_model=str(VALEN_MODEL))
    write(sample/'job.json', job)
    return job, {'sample_id': row['sample_id'], 'candidates': len(candidates),
                 'centroids_outside_own_mask': sum(not q['inside_own_mask'] for q in queries),
                 'candidate_masks_and_first_two_images_unchanged': True}


def prepare(args):
    source = Path(args.source).resolve(strict=True)
    output = empty_output(args.root)
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    assert rows and len({r['sample_id'] for r in rows}) == len(rows)
    config = deepcopy(read(source/'config_snapshot.json'))
    config['valen'].update(save_compiled_inputs=True, save_compiled_tensors=False)
    definitions = read(source/'affordances_snapshot.json')
    write(output/'config_snapshot.json', config)
    write(output/'affordances_snapshot.json', definitions)
    write_jsonl(output/'evaluation_manifest.jsonl', rows)
    jobs, audits = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(lambda row: prepare_one(row, source, output, config, definitions), rows)
        for i, (job, audit) in enumerate(results, 1):
            jobs.append(job); audits.append(audit)
            if i % 100 == 0 or i == len(rows):
                print(f'PREPARE {i}/{len(rows)}', flush=True)
    write_jsonl(output/'model_jobs.jsonl', jobs)
    write_jsonl(output/'centroid_audit_per_image.jsonl', audits)
    write(output/'centroid_audit.json', {
        'images': len(rows), 'candidates': sum(a['candidates'] for a in audits),
        'centroids_outside_own_mask': sum(a['centroids_outside_own_mask'] for a in audits),
        'images_with_outside_centroid': sum(a['centroids_outside_own_mask'] > 0 for a in audits),
        'centroid_rule': 'arithmetic mean of predicted-mask pixel coordinates; no snapping to foreground',
        'dot_radius_original_pixels': DOT_RADIUS, 'calls_per_candidate': 1, 'images_per_call': 3,
        'first_two_images': 'byte-identical to the original fill baseline',
        'third_image': 'unpainted RGB crop plus one red/white centroid point, no mask fill/outline',
        'functional_definitions_unchanged': True, 'candidate_masks_unchanged': True,
        'gt_used_in_inputs': False, 'adapter': None, 'source': str(source),
        'source_manifest_sha256': sha(args.manifest),
        'source_code_sha256': sha(__file__)})
    write(output/'run.json', {'status': 'prepared', 'samples': len(rows), 'style': 'centroid',
          'protocol': 'original', 'source_candidates': str(source), 'adapter': None,
          'evaluation_manifest_sha256': sha(output/'evaluation_manifest.jsonl'),
          'reference_annotations_used_in_inference': False, 'whole_object_prohibition': False})
    print(output, flush=True)


def infer(args):
    from ...decision.valen import Valen
    from .infer import compact_merge
    root = Path(args.root).resolve()
    jobs = read_jsonl(root/'model_jobs.jsonl')[args.shard_index::args.shard_count]
    if not jobs:
        raise ValueError('Empty inference shard')
    config = read(root/'config_snapshot.json')['valen']
    model = Valen(str(VALEN_MODEL), config)
    write(root/f'model_load_{args.shard_index}.json', {'model': str(VALEN_MODEL), 'adapter': None,
          'seconds': model.load_seconds, 'model_manifest_sha256': sha(PROJECT/'models/manifest.json')})
    started = time.perf_counter()
    counts = Counter()
    for i, job in enumerate(jobs, 1):
        sample = Path(job['output'])
        if read(sample/'run.json')['status'] == 'complete':
            counts['complete'] += 1
            continue
        try:
            verify_prepared(sample)
            points = read(sample/'centroids.json')['regions']
            responses, runtimes = [], []
            for point in points:
                folder = sample/point['directory']
                model.predict(folder)
                response = read(folder/'valen_response.json')
                assert set(response['answers']) == {f"region_{point['region_id']:02}"}
                assert response['usage']['output_tokens'] == 0
                responses.append(response)
                runtimes.append(read(folder/'valen_runtime.json'))
            answers = {k: v for response in responses for k, v in response['answers'].items()}
            write(sample/'valen_response.json', {'model': 'Valen', 'answers': answers,
                  'aggregation': 'independent single-centroid requests; no cross-query conversation',
                  'usage': {'input_tokens': sum(r.get('usage',{}).get('input_tokens',0) for r in responses),
                            'output_tokens': 0},
                  'request_sha256': sha(sample/'valen_request.json')})
            runtime = dict(runtimes[0]) if runtimes else {}
            runtime.update(request_questions=len(points), forward_calls=len(points),
                           first_model_request=any(r['first_model_request'] for r in runtimes),
                           timings={'predict_seconds': sum(r['timings']['predict_seconds'] for r in runtimes),
                                    'compiled_input_save_seconds_inside_predict': sum(r['timings']['compiled_input_save_seconds_inside_predict'] for r in runtimes)},
                           timing_note='Sum of all candidate calls for this RGB; includes input dump; use warmed timing for comparison.')
            write(sample/'valen_runtime.json', runtime)
            write(sample/'run.json', {'status': 'classified', 'reference_annotations_used': False})
            compact_merge(sample)
            counts['complete'] += 1
        except Exception as exc:
            traceback.print_exc()
            write(sample/'run.json', {'status': 'failed', 'stage': 'centroid_decision', 'error': str(exc)})
            counts['failed'] += 1
        if i % 10 == 0 or i == len(jobs):
            write(root/f'progress_{args.shard_index}.json', {'completed': i, 'total': len(jobs),
                  'counts': dict(counts), 'elapsed_seconds': time.perf_counter()-started})
            print(f'CENTROID shard={args.shard_index} {i}/{len(jobs)} {dict(counts)}', flush=True)
    write(root/f'worker_{args.shard_index}.json', {'status': 'complete' if not counts['failed'] else 'failed',
          'counts': dict(counts), 'shard_index': args.shard_index, 'shard_count': args.shard_count,
          'batch_wall_seconds': time.perf_counter()-started})


def finalize(args):
    root = Path(args.root).resolve()
    jobs = read_jsonl(root/'model_jobs.jsonl')
    states = [read(Path(j['output'])/'run.json')['status'] for j in jobs]
    if any(s not in ['complete', 'failed'] for s in states):
        raise ValueError('Some inference samples have not finished')
    counts = Counter(states)
    write(root/'run.json', {**read(root/'run.json'), 'counts': dict(counts),
          'status': 'complete' if not counts['failed'] else 'complete_with_failures'})
    print(dict(counts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','infer','finalize'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', default=str(PROJECT/'outputs/umd_visual_prompt_ablation/fill'))
    parser.add_argument('--manifest', default=str(PROJECT/'data/UMD/manifests/umd_novel_test.jsonl'))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.limit < 0:
        parser.error('Invalid shard or limit')
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
