"""Inspect every centroid request and compare saved predictions; never run a model."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ...common import read, write, read_jsonl, sha
from ...datasets.umd import load_annotation
from ...pipeline.artifacts import verify_prepared
from .report import label_data, relative, protocol_rescore


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    source = Path(read(root/'run.json')['source_candidates'])
    assert read(root/'run.json')['counts'] == {'complete': 2661}
    metrics = read(root/'evaluation/metrics.json')
    rows = read_jsonl(root/'evaluation_manifest.jsonl')
    assert metrics['metrics']['images'] == len(rows) == 2661
    assert metrics['metrics']['gt_present_queries'] == 5232
    assert not metrics['failures']
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(__file__).parent/'assets', output/'fonts', dirs_exist_ok=True)
    scores = {key: {r['sample_id']: r['mean_IoU_present_classes']
                    for r in read_jsonl(folder/'evaluation/per_image.jsonl')}
              for key, folder in [('baseline', source), ('centroid', root)]}
    items, queries, outside = [], 0, 0
    for index, row in enumerate(rows, 1):
        sid = row['sample_id']
        sample, original = root/'samples'/sid, source/'samples'/sid
        verify_prepared(sample)
        assert read(sample/'run.json')['status'] == 'complete'
        assert sha(sample/'candidates.json') == sha(original/'candidates.json')
        for name in ['foreground/rgb_crop.png', 'regions/numbered_crop.png', 'regions/label_map.png']:
            assert sha(sample/name) == sha(original/name), (sid, name)
        points = read(sample/'centroids.json')['regions']
        before = {r['region_id']: r for r in read(original/'decisions.json')['regions']}
        after = {r['region_id']: r for r in read(sample/'decisions.json')['regions']}
        aggregate = read(sample/'valen_response.json')
        assert aggregate['usage']['output_tokens'] == 0
        assert aggregate['request_sha256'] == sha(sample/'valen_request.json')
        candidates = []
        for point in points:
            folder = sample/point['directory']
            request, response = read(folder/'valen_request.json'), read(folder/'valen_response.json')
            qid = f"region_{point['region_id']:02}"
            assert set(request['questions']) == set(response['answers']) == {qid}
            assert response['usage']['output_tokens'] == 0
            assert response['request_sha256'] == sha(folder/'valen_request.json')
            assert response['answers'][qid] == aggregate['answers'][qid]
            assert [c['image_url']['url'] for c in request['state']['messages'][0]['content']
                    if c['type'] == 'image_url'] == ['rgb_crop.png', 'numbered_crop.png', 'centroid.png']
            for name, digest in read(folder/'prepared_assets.json').items():
                assert sha(folder/name) == digest, (sid, point['region_id'], name)
            candidates.append({**point, 'image': relative(folder/'centroid.png', output),
                'prompt': relative(folder/'valen_compiled/prompt.txt', output),
                'request': relative(folder/'valen_request.json', output),
                'response': relative(folder/'valen_response.json', output),
                'baseline': before[point['region_id']]['final_choice'],
                'prediction': after[point['region_id']]['final_choice']})
        gt, _ = load_annotation(row['annotation'], tuple(row['size_wh'][::-1]))
        items.append({'id': sid, 'category': row['object_instance'].rsplit('_', 1)[0],
            'rgb': relative(sample/'rgb.png', output), 'size': row['size_wh'],
            'crop': read(sample/'foreground/metadata.json')['crop_xyxy'],
            'gt': label_data(np.where(gt < 0, 255, gt)),
            'predictions': {key: label_data(np.asarray(Image.open(folder/'samples'/sid/'result/class_map.png')))
                            for key, folder in [('baseline', source), ('centroid', root)]},
            'scores': {key: value[sid] for key, value in scores.items()},
            'inputs': [relative(sample/name, output) for name in ['foreground/rgb_crop.png', 'regions/numbered_crop.png']],
            'candidates': candidates})
        queries += len(points)
        outside += sum(not p['inside_own_mask'] for p in points)
        if index % 250 == 0:
            print(f'REPORT {index}/{len(rows)}; audited queries={queries}', flush=True)
    expected = read(root/'centroid_audit.json')
    assert queries == expected['candidates'] and outside == expected['centroids_outside_own_mask']
    write(output/'audit.json', {'status': 'complete', 'images': len(rows), 'independent_requests': queries,
        'centroids_outside_mask': outside, 'outside_fraction': outside/queries,
        'all_request_asset_hashes_verified': True, 'first_two_images_and_masks_match_baseline': True,
        'all_requests_have_three_images_one_question': True, 'all_responses_have_zero_output_tokens': True,
        'ground_truth_used_only_after_prediction': True})
    payload = {'items': items, 'metrics': metrics['metrics']}
    (output/'data.js').write_text('window.CENTROID='+json.dumps(payload, ensure_ascii=False, separators=(',', ':'))+';\n', encoding='utf-8')
    shutil.copyfile(Path(__file__).parent/'centroid_viewer.html', output/'index.html')
    protocol_rescore({'centroid': root}, output)
    print(output/'index.html', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.root, args.output)


if __name__ == '__main__':
    main()
