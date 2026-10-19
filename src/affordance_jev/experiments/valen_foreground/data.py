"""Freeze the existing query list and isolate model inputs from annotations."""
from collections import Counter
from pathlib import Path

from ...common import read, write, read_jsonl, write_jsonl, sha


def initialize(root, config, manifest, queries, definitions, models, baseline):
    root = Path(root)
    signature = {'config': sha(config), 'manifest': sha(manifest), 'queries': sha(queries),
                 'definitions': sha(definitions), 'models': models, 'baseline': str(baseline)}
    if (root/'protocol.json').exists():
        if read(root/'protocol.json')['input_signature'] != signature:
            raise ValueError('Existing experiment has different inputs; choose a new output directory')
        return
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    cfg = read(config)
    threshold = cfg['foreground_selector']['minimum_yes_probability']
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError('foreground_selector.minimum_yes_probability must be a probability in [0, 1]')
    assert cfg['vocabulary'] == {'objects': ['object'], 'parts': ['part']}
    assert cfg['functional_method'] == 'centroid'
    assert cfg['partition']['whole_object_fraction'] is None
    from .point_union import enabled
    point_mode = enabled(cfg)
    rows, qs = read_jsonl(manifest), read_jsonl(queries)
    assert len(rows) == 2661 and len(qs) == 5232
    by_id = {r['sample_id']: r for r in rows}
    assert len(by_id) == 2661 and len({q['query_id'] for q in qs}) == 5232
    for q in qs:
        assert set(q) == {'sample_id', 'query_id', 'rgb', 'rgb_sha256', 'size_wh', 'object_category', 'class_id', 'action'}
        assert q['rgb_sha256'] == by_id[q['sample_id']]['rgb_sha256']
    write(root/'config.json', cfg)
    write(root/'definitions.json', read(definitions))
    write(root/'models.json', models)
    write_jsonl(root/'test_manifest.jsonl', rows)
    write_jsonl(root/'test_queries.jsonl', qs)
    write(root/'protocol.json', {'input_signature': signature, 'images': len(rows), 'queries': len(qs),
        'fresh_inference': True, 'training_images': 0,
        'foreground': ('disjoint connected regions, interior point yes/no, union of all accepted regions' if point_mode else
                       'independent yes/no, then single highest accepted probability'),
        'candidate_groups': cfg['foreground_selector']['group_size'], 'no_geometric_fallback': True,
        'no_gt_in_model_inputs': True, 'baseline': str(baseline),
        'foreground_minimum_yes_probability': threshold, 'foreground_threshold_comparison': 'strictly greater than',
        'functional_decision_rule': 'P(yes) > P(no); tie rejected; independent of foreground threshold',
        'metric_protocol': 'same known-operation queries; native dominant GT; background errors count; void ignored'})


def prepare_phase(root, phase):
    root = Path(root)
    dest = root/phase
    if (dest/'protocol.json').exists():
        return dest
    rows, qs = read_jsonl(root/'test_manifest.jsonl'), read_jsonl(root/'test_queries.jsonl')
    if phase == 'smoke':
        first = {}
        for q in qs:
            first.setdefault(q['object_category'], q['sample_id'])
        selected = set(first.values())
        assert len(selected) == 9
        rows = [r for r in rows if r['sample_id'] in selected]
        qs = [q for q in qs if q['sample_id'] in selected]
    for name in ['config.json', 'definitions.json', 'models.json']:
        write(dest/name, read(root/name))
    write_jsonl(dest/'evaluation_manifest.jsonl', rows)
    write_jsonl(dest/'queries.jsonl', qs)
    # PartGLEE workers receive no categories, actions, or annotations.
    write_jsonl(dest/'images.jsonl', [{k: r[k] for k in ['sample_id', 'rgb', 'rgb_sha256', 'size_wh']} for r in rows])
    write(dest/'protocol.json', {'phase': phase, 'images': len(rows), 'queries': len(qs),
        'query_counts': dict(Counter(q['action'] for q in qs)), 'queries_sha256': sha(dest/'queries.jsonl'),
        'manifest_sha256': sha(dest/'evaluation_manifest.jsonl'), 'gt_in_model_jobs': False,
        'evaluation_use': 'diagnostic development subset, not an independent holdout' if phase == 'smoke' else
                          'full test evaluation; consult experiment history for development use'})
    return dest
