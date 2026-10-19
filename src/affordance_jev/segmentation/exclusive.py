"""Coverage-preserving, connected partitions of overlapping object proposals.

This stage knows neither target classes nor ground truth. Unlike the downstream
part filter, it merges small components rather than deleting their pixels.
"""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label


CONNECTIVITY = np.ones((3, 3), dtype=bool)


def internal_point(mask):
    """Discrete largest-inscribed-circle center; ties prefer the mass centroid."""
    mask = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('Cannot locate a point in an empty region')
    cy, cx = float(yy.mean()), float(xx.mean())
    # Padding makes the image exterior count as outside, including all-true masks.
    distance = distance_transform_edt(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]
    positions = np.argwhere(distance == distance.max())
    best = int(np.argmin((positions[:, 0]-cy)**2 + (positions[:, 1]-cx)**2))
    y, x = map(int, positions[best])  # np.argwhere is stable row-major order.
    assert mask[y, x]
    return {'xy': [x, y], 'arithmetic_centroid_xy': [cx, cy],
            'distance_to_outside_pixels': float(distance[y, x]), 'inside_own_mask': True,
            'rule': 'maximum Euclidean distance to outside; ties: nearest mass centroid, then y/x'}


def partition_connected(masks, qualities, raw_ids, minimum_fragment_pixels=512):
    masks = np.asarray(masks, dtype=bool)
    qualities = np.asarray(qualities, dtype=np.float64)
    raw_ids = list(raw_ids)
    if masks.ndim != 3 or len(masks) != len(qualities) or len(masks) != len(raw_ids):
        raise ValueError('Proposal shapes/metadata disagree')
    if len(set(raw_ids)) != len(raw_ids) or not np.isfinite(qualities).all() or (qualities < 0).any():
        raise ValueError('Invalid proposal scores/IDs')
    if minimum_fragment_pixels < 1:
        raise ValueError('Minimum fragment size must be positive')
    areas = masks.sum((1, 2))
    if (areas == 0).any():
        raise ValueError('Empty proposals must be removed before partitioning')
    priorities = qualities / np.sqrt(areas)
    order = sorted(range(len(masks)), key=lambda i: (-priorities[i], -qualities[i], raw_ids[i]))
    owner = np.zeros(masks.shape[1:], dtype=np.int32)
    for i in order:
        owner[masks[i] & (owner == 0)] = i+1
    components = np.zeros_like(owner)
    fragments, next_id = [], 0
    for i in range(len(masks)):
        cc, count = label(owner == i+1, structure=CONNECTIVITY)
        components[cc > 0] = cc[cc > 0] + next_id
        sizes = np.bincount(cc.ravel())
        fragments.extend({'fragment_id': next_id+j, 'raw_id': raw_ids[i], 'pixels': int(sizes[j])}
                         for j in range(1, count+1))
        next_id += count
    sizes = np.bincount(components.ravel())
    merges = []
    small = sorted((int(sizes[i]), i) for i in range(1, len(sizes))
                   if sizes[i] < minimum_fragment_pixels)
    for _, source_id in small:
        if sizes[source_id] == 0 or sizes[source_id] >= minimum_fragment_pixels:
            continue
        patch = components == source_id
        yy, xx = np.nonzero(patch)
        y0, y1 = max(0, int(yy.min())-1), min(owner.shape[0], int(yy.max())+2)
        x0, x1 = max(0, int(xx.min())-1), min(owner.shape[1], int(xx.max())+2)
        local = components[y0:y1, x0:x1]
        ring = binary_dilation(local == source_id, structure=CONNECTIVITY) & (local != source_id) & (local > 0)
        neighbors, contacts = np.unique(local[ring], return_counts=True)
        if not len(neighbors):
            merges.append({'source_fragment': source_id, 'target_fragment': None,
                           'pixels': int(sizes[source_id]), 'reason': 'isolated_retained'})
            continue
        best = min(range(len(neighbors)), key=lambda j: (-int(contacts[j]),
                   -int(sizes[neighbors[j]]), int(neighbors[j])))
        target_id = int(neighbors[best])
        merges.append({'source_fragment': source_id, 'target_fragment': target_id,
                       'pixels': int(sizes[source_id]), 'neighbor_contact_pixels': int(contacts[best])})
        components[patch] = target_id
        sizes[target_id] += sizes[source_id]
        sizes[source_id] = 0
    surviving = [int(i) for i in np.unique(components) if i]
    surviving.sort(key=lambda i: int(np.flatnonzero(components == i)[0]))
    labels = np.zeros_like(owner)
    regions = []
    for region_id, component_id in enumerate(surviving, 1):
        mask = components == component_id
        _, count = label(mask, structure=CONNECTIVITY)
        assert count == 1
        labels[mask] = region_id
        sources, counts = np.unique(owner[mask], return_counts=True)
        regions.append({'id': region_id, 'component_id': component_id, 'area': int(mask.sum()),
                        'sources': [{'raw_id': raw_ids[int(s)-1], 'pixels': int(n)} for s, n in zip(sources, counts)],
                        'point': internal_point(mask)})
    union = masks.any(0)
    assert np.array_equal(labels > 0, union)
    trace = {'rule': 'quality / sqrt(original eligible mask area)', 'minimum_fragment_pixels': minimum_fragment_pixels,
             'connectivity': 8, 'input_masks': len(masks), 'fragments_before_merge': next_id,
             'regions_after_merge': len(regions), 'input_union_pixels': int(union.sum()),
             'output_union_pixels': int((labels > 0).sum()), 'coverage_preserved': True,
             'overlap_pixels_before': int((masks.sum(0) > 1).sum()), 'overlap_pixels_after': 0,
             'uncovered_pixels': int((labels == 0).sum()),
             'priority': [{'raw_id': raw_ids[i], 'quality': float(qualities[i]),
                          'area': int(areas[i]), 'priority': float(priorities[i])} for i in order],
             'fragments': fragments, 'merges': merges, 'reference_annotations_used': False}
    return labels, owner, regions, trace
