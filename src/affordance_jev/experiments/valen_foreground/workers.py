"""Four staged workers, using separate external PartGLEE and Valen environments."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from ...common import read, write, read_jsonl, sha, load_mask, save_mask
from ...pipeline.run import sample_directory
from ...visualization import overlay
from ..umd_valen.data import link
from ..conditional_umd.requests import METHODS, prepare_assets, prepare_query, decode
from ..conditional_umd.infer import save_result
from .selection import eligible_candidates, render_groups, decide
from . import point_union


def objects_path(root, sid):
    return sample_directory(root/'objects', sid)


def foreground_path(root, query):
    return sample_directory(root/'foreground', query['sample_id'])/'queries'/query['action']


def function_path(root, query, method='centroid'):
    return sample_directory(root/'methods'/method, query['sample_id'])/'queries'/query['action']


def finished(folder):
    return (folder/'run.json').exists() and read(folder/'run.json')['status'] in ['complete', 'failed']


def stage_objects(adapter, root, row, config):
    from ...segmentation.partglee import save_raw
    out = objects_path(root, row['sample_id'])
    if finished(out):
        return read(out/'run.json')['status']
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    try:
        assert sha(row['rgb']) == row['rgb_sha256']
        rgb = Image.open(row['rgb']).convert('RGB')
        assert list(rgb.size) == row['size_wh']
        rgb.save(out/'rgb.png')
        masks, qualities, labels, elapsed = adapter.predict(rgb, 'objects')
        save_raw(out, 'object_candidates', rgb, masks, qualities, labels)
        raw = read(out/'raw/objects.json')
        for r in raw['candidates']:
            r.update(level='object', mask_sha256=sha(out/r['mask']))
        write(out/'raw/objects.json', raw)
        candidates, trace = eligible_candidates(raw['candidates'], masks, config['foreground'])
        write(out/'eligible.json', {'candidates': candidates, 'filter_trace': trace,
                                   'no_center_or_area_ranking': True})
        preparation_started = time.perf_counter()
        if point_union.enabled(config):
            regions = point_union.prepare(out, rgb, candidates, config)
            groups = regions
        else:
            groups = render_groups(out, candidates, config['foreground_selector']['group_size'])
        write(out/'timing.json', {'object_forward_seconds': elapsed,
                                 'foreground_assets_seconds': time.perf_counter()-preparation_started,
                                 'stage_with_io_seconds': time.perf_counter()-started})
        write(out/'run.json', {'status': 'complete', 'raw_objects': len(masks),
                              'eligible_objects': len(candidates), 'groups': len(groups),
                              'text': ['object', 'part'], 'gt_used': False})
    except Exception as exc:
        traceback.print_exc()
        write(out/'run.json', {'status': 'failed', 'error': str(exc)})
    return read(out/'run.json')['status']


def stage_foreground(model, root, query, config):
    source = objects_path(root, query['sample_id'])
    out = foreground_path(root, query)
    if finished(out):
        return read(out/'run.json').get('reason', 'complete')
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    write(out/'query.json', query)
    point_mode = point_union.enabled(config)
    chosen, decisions, runtimes, error = ([] if point_mode else None), [], [], None
    mask = np.zeros(query['size_wh'][::-1], bool)
    try:
        if read(source/'run.json')['status'] != 'complete':
            raise RuntimeError('Object proposal stage failed')
        candidate_path = 'preprocessed/candidates.json' if point_mode else 'eligible.json'
        candidates = read(source/candidate_path)['candidates']
        judge = point_union.decide if point_mode else decide
        chosen, decisions, runtimes = judge(model, source, out, query, candidates, config['foreground_selector'])
        for c in chosen if point_mode else ([chosen] if chosen else []):
            assert sha(source/c['mask']) == c['mask_sha256']
            mask |= load_mask(source/c['mask'], mask.shape)
        decision_map = {d['region_id']: d for d in decisions}
        write(out/'decisions.json', [{**r, **decision_map[r['id']]} for r in candidates])
    except Exception as exc:
        traceback.print_exc()
        error = f'{type(exc).__name__}: {exc}'
        chosen = [] if point_mode else None
        mask[:] = False
    save_mask(out/'selected_mask.png', mask)
    rgb = Image.open(query['rgb']).convert('RGB')
    overlay(rgb, [mask], labels=['target' if chosen else '']).save(out/'overlay.jpg', quality=90)
    if point_mode:
        overlay(rgb, [load_mask(source/c['mask'], mask.shape) for c in chosen],
                labels=[str(c['id']) for c in chosen]).save(out/'accepted_regions.jpg', quality=95)
    write(out/'selection.json', {'selected': chosen, 'mask': 'selected_mask.png',
        'mode': config['foreground_selector']['method'],
        'minimum_yes_probability': config['foreground_selector']['minimum_yes_probability'],
        'mask_sha256': sha(out/'selected_mask.png'), 'reason': 'error' if error else 'selected' if chosen else 'no_target',
        'rule': (f"union of all disjoint regions with P(yes)>{config['foreground_selector']['minimum_yes_probability']}" if point_mode else
                 f"highest P(yes)>{config['foreground_selector']['minimum_yes_probability']}; tie: quality descending, raw_id ascending"), 'geometric_fallback': False,
        'query_id': query['query_id'], 'reference_annotations_used': False})
    write(out/'timing.json', {'valen_seconds': sum(r['timings']['predict_seconds'] for r in runtimes),
        'compiled_save_seconds': sum(r['timings']['compiled_input_save_seconds_inside_predict'] for r in runtimes),
        'calls': len(runtimes), 'stage_with_io_seconds': time.perf_counter()-started})
    write(out/'run.json', {'status': 'failed' if error else 'complete', 'error': error,
                          'reason': 'error' if error else 'selected' if chosen else 'no_target'})
    return read(out/'run.json')['reason']


def stage_parts(adapter, root, query, config, models):
    from ...segmentation.partglee import prepare_object_parts
    from ...pipeline.artifacts import prepare_regions
    fg = foreground_path(root, query)
    result = read(fg/'selection.json')
    index = fg/'parts.json'
    if index.exists():
        previous = read(index)
        if previous['status'] != 'ready' or (root/previous['path']/'run.json').exists():
            return previous['status']
    if not result['selected']:
        write(index, {'status': 'skipped', 'reason': result['reason'], 'path': None})
        return 'skipped'
    assert sha(fg/'selected_mask.png') == result['mask_sha256']
    # The cache is scoped to this RGB and frozen run config/weights, not operation.
    out = sample_directory(root/'parts', query['sample_id'])/result['mask_sha256']
    state = out/'run.json'
    if state.exists() and read(state)['status'] in ['prepared', 'failed']:
        status = 'ready' if read(state)['status'] == 'prepared' else 'failed'
        write(index, {'status': status, 'path': str(out.relative_to(root)), 'reused': True})
        return status
    out.mkdir(parents=True, exist_ok=True)
    try:
        source = objects_path(root, query['sample_id'])
        if not (out/'rgb.png').exists():
            link(source/'rgb.png', out/'rgb.png')
        write(out/'config_snapshot.json', config)
        write(out/'affordances_snapshot.json', read(root/'definitions.json'))
        job = {'sample_id': query['sample_id'], 'output': str(out),
               'object_mask': str(fg/'selected_mask.png'), 'reference_annotations_used': False,
               'partglee_root': models['partglee_root'], 'partglee_weights': models['partglee_weights']}
        write(out/'job.json', job)
        prepare_object_parts(adapter, job)
        metadata = read(out/'foreground/metadata.json')
        if point_union.enabled(config):
            metadata.update(source='valen_interior_point_union',
                            selected_region_ids=[c['id'] for c in result['selected']])
        else:
            metadata.update(source='valen_selected_object', selected_raw_id=result['selected']['raw_id'])
        metadata.update(selection_query_id=query['query_id'], reused_for_identical_selected_masks=True)
        write(out/'foreground/metadata.json', metadata)
        raw = read(out/'raw/parts.json')
        for r in raw['candidates']:
            r['level'] = 'part'
        write(out/'raw/parts.json', raw)
        prepare_regions(job)
        candidates = read(out/'candidates.json')
        for r in candidates['candidates']:
            r['level'] = 'part'
        write(out/'candidates.json', candidates)
        prepared = read(out/'prepared_assets.json')
        prepared['candidates.json'] = sha(out/'candidates.json')
        write(out/'prepared_assets.json', prepared)
        # Verify coordinate, clipping and disjointness before any functional scoring.
        fg_mask = load_mask(fg/'selected_mask.png', query['size_wh'][::-1])
        coverage = np.zeros(fg_mask.shape, np.uint8)
        for r in candidates['candidates']:
            m = load_mask(out/r['mask'], fg_mask.shape)
            assert not (m & ~fg_mask).any()
            coverage += m
        assert coverage.max() <= 1
        write(out/'geometry_audit.json', {'overlap_pixels': 0, 'outside_foreground_pixels': 0,
                                        'regions': len(candidates['candidates']), 'gt_used': False})
        status = 'ready'
    except Exception as exc:
        traceback.print_exc()
        status = 'failed'
        write(state, {'status': 'failed', 'error': str(exc)})
    write(index, {'status': status, 'path': str(out.relative_to(root)), 'reused': False})
    return status


def stage_functions(model, root, query, definitions, method='centroid'):
    out = function_path(root, query, method)
    if finished(out):
        return read(out/'run.json')['status']
    started = time.perf_counter()
    fg = foreground_path(root, query)
    selection, fg_state = read(fg/'selection.json'), read(fg/'run.json')
    part_index = read(fg/'parts.json')
    mask = np.zeros(query['size_wh'][::-1], bool)
    decisions, runtimes, error = [], [], None
    reason = selection['reason']
    try:
        if fg_state['status'] == 'failed':
            raise RuntimeError(fg_state['error'])
        if part_index['status'] == 'failed':
            raise RuntimeError('Part segmentation failed')
        if part_index['status'] == 'ready':
            source = root/part_index['path']
            assets = source/f'{method}_inputs'
            if not assets.exists():
                candidates = prepare_assets(source, assets, method)
            else:
                assert read(assets/'source.json')['candidate_file_sha256'] == sha(source/'candidates.json')
                candidates = read(source/'candidates.json')['candidates']
            requests = prepare_query(assets, out, query, candidates, definitions, method)
            for request in requests:
                folder = Path(request['folder'])
                model.predict(folder)
                response = read(folder/'valen_response.json')
                assert response.get('usage', {}).get('output_tokens', 0) == 0
                decisions.extend(decode(response, request['candidates']))
                runtimes.append(read(folder/'valen_runtime.json'))
            assert len(decisions) == len(candidates)
            chosen = {d['region_id'] for d in decisions if d['selected']}
            for c in candidates:
                if c['id'] in chosen:
                    mask |= load_mask(source/c['mask'], mask.shape)
            assert not (mask & ~load_mask(fg/'selected_mask.png', mask.shape)).any()
            reason = 'predicted' if candidates else 'no_parts'
    except Exception as exc:
        traceback.print_exc()
        error = f'{type(exc).__name__}: {exc}'
        mask[:] = False
        reason = 'error'
    timings = {'functional_valen_seconds': sum(r['timings']['predict_seconds'] for r in runtimes),
        'compiled_save_seconds': sum(r['timings']['compiled_input_save_seconds_inside_predict'] for r in runtimes),
        'calls': len(runtimes), 'stage_with_io_seconds': time.perf_counter()-started}
    save_result(out, query, mask, error=error, timing=timings, decisions=decisions)
    write(out/'sources.json', {'foreground': str(fg.relative_to(root)), 'parts': part_index['path'],
        'reason': reason, 'foreground_mask_sha256': selection['mask_sha256'], 'gt_used': False})
    return 'failed' if error else reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--stage', choices=['objects', 'foreground', 'parts', 'functions'], required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--method', choices=list(METHODS))
    parser.add_argument('--adapter')
    args = parser.parse_args()
    if args.method and args.stage != 'functions':
        parser.error('--method applies only to functional classification')
    if args.adapter and (args.stage != 'functions' or args.method != 'sft'):
        parser.error('--adapter applies only to functional SFT; foreground always uses original Valen')
    if args.method == 'sft' and not args.adapter:
        parser.error('SFT inference requires its trained adapter and head')
    stage_key = f'functions_{args.method}' if args.method else args.stage
    root = Path(args.root).resolve()
    config, models = read(root/'config.json'), read(root/'models.json')
    images = read_jsonl(root/'images.jsonl')[args.shard_index::args.shard_count]
    queries = defaultdict(list)
    if args.stage != 'objects':
        for q in read_jsonl(root/'queries.jsonl'):
            queries[q['sample_id']].append(q)
    started = time.perf_counter()
    audit = root/'loads'/f'{stage_key}_{args.shard_index}'
    if args.stage in ['objects', 'parts']:
        import torch
        from ...segmentation.partglee import PartGLEE
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model = PartGLEE(models['partglee_root'], models['partglee_weights'], config['vocabulary'], audit,
                         clip_model=models['clip_model'])
    else:
        from ...decision.valen import Valen
        model = Valen(models['valen_model'], config['valen'])
        if args.adapter:
            from ..umd_valen.infer import load_adapter
            load_adapter(model.model, args.adapter)
    load_seconds = time.perf_counter()-started
    write(audit/'load.json', {'seconds': load_seconds, 'models': models, 'adapter': args.adapter,
        'adapter_sha256': sha(Path(args.adapter)/'adapter/adapter_model.safetensors') if args.adapter else None,
        'head_sha256': sha(Path(args.adapter)/'head.safetensors') if args.adapter else None})
    definitions = read(root/'definitions.json')
    counts = Counter()
    for i, row in enumerate(images, 1):
        if args.stage == 'objects':
            counts[stage_objects(model, root, row, config)] += 1
        else:
            for q in queries[row['sample_id']]:
                if args.stage == 'foreground':
                    status = stage_foreground(model, root, q, config)
                elif args.stage == 'parts':
                    status = stage_parts(model, root, q, config, models)
                else:
                    status = stage_functions(model, root, q, definitions, args.method or 'centroid')
                counts[status] += 1
        progress = {'stage': stage_key, 'images_done': i, 'images_total': len(images),
                    'counts': dict(counts), 'seconds': time.perf_counter()-started}
        write(root/'progress'/f'{stage_key}_{args.shard_index}.json', progress)
        if i % 10 == 0 or i == len(images):
            print(f'PROGRESS {stage_key} shard={args.shard_index} {i}/{len(images)} {dict(counts)}', flush=True)
    write(root/'workers'/f'{stage_key}_{args.shard_index}.json', {'status': 'complete', **progress})


if __name__ == '__main__':
    main()
