import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

from affordance_jev.segmentation.exclusive import internal_point, partition_connected
from affordance_jev.experiments.valen_foreground.point_union import choose, request, chinese_prompt, prepare
from affordance_jev.common import write, save_mask, read, load_mask
from affordance_jev.experiments.conditional_umd.requests import decode


class ExclusiveRegions(unittest.TestCase):
    def test_nested_overlap_quality_area_priority_and_coverage(self):
        masks = np.zeros((3, 16, 16), bool)
        masks[0, 1:15, 1:15] = True
        masks[1, 5:11, 5:11] = True
        masks[2, 1:4, 1:4] = True
        labels, owner, regions, trace = partition_connected(masks, [.8, .8, .8], [10, 20, 30], 1)
        self.assertEqual(owner[6, 6], 2)
        self.assertEqual(owner[2, 2], 3)
        self.assertTrue(np.array_equal(labels > 0, masks.any(0)))
        self.assertEqual(trace['overlap_pixels_after'], 0)
        for r in regions:
            self.assertEqual(label(labels == r['id'], np.ones((3, 3)))[1], 1)

    def test_fragment_merge_conserves_pixels_and_provenance(self):
        masks = np.zeros((2, 8, 8), bool)
        masks[0, 1:7, 1:7] = True
        masks[1, 3:5, 3:5] = True
        labels, _, regions, trace = partition_connected(masks, [.5, .5], [0, 1], 8)
        self.assertEqual(len(regions), 1)
        self.assertEqual(set(s['raw_id'] for s in regions[0]['sources']), {0, 1})
        self.assertEqual(sum(s['pixels'] for s in regions[0]['sources']), 36)
        self.assertTrue(np.array_equal(labels > 0, masks.any(0)))
        self.assertTrue(any(m.get('target_fragment') for m in trace['merges']))

    def test_disconnected_isolated_regions_and_holes_remain(self):
        masks = np.zeros((1, 12, 12), bool)
        masks[0, 0, 0] = True
        masks[0, 4:11, 4:11] = True
        masks[0, 6:9, 6:9] = False
        labels, _, regions, _ = partition_connected(masks, [1.], [0], 512)
        self.assertEqual(len(regions), 2)
        self.assertTrue(np.array_equal(labels > 0, masks[0]))
        self.assertFalse(labels[7, 7])

    def test_empty_candidates(self):
        labels, _, regions, trace = partition_connected(np.zeros((0, 5, 6), bool), [], [], 512)
        self.assertFalse(labels.any())
        self.assertEqual(regions, [])
        self.assertEqual(trace['regions_after_merge'], 0)

    def test_internal_point_ring_full_image_thin_border_and_tie(self):
        masks = [np.ones((10, 10), bool), np.eye(10, dtype=bool)]
        ring = np.ones((11, 11), bool)
        ring[3:8, 3:8] = False
        masks.append(ring)
        for mask in masks:
            p = internal_point(mask)
            x, y = p['xy']
            self.assertTrue(mask[y, x])
            self.assertEqual(p, internal_point(mask))
        self.assertEqual(internal_point(masks[0])['xy'], [4, 4])
        with self.assertRaises(ValueError):
            internal_point(np.zeros((5, 5), bool))

    def test_randomized_connected_regions_union_and_determinism(self):
        rng = np.random.default_rng(19)
        for _ in range(12):
            masks = rng.random((5, 20, 20)) > .8
            q = rng.random(5)
            a, _, rs, _ = partition_connected(masks, q, list(range(5)), 12)
            b, _, _, _ = partition_connected(masks, q, list(range(5)), 12)
            self.assertTrue(np.array_equal(a, b))
            self.assertTrue(np.array_equal(a > 0, masks.any(0)))
            for r in rs:
                x, y = r['point']['xy']
                self.assertEqual(a[y, x], r['id'])
                self.assertEqual(label(a == r['id'], np.ones((3, 3)))[1], 1)


class PointRequests(unittest.TestCase):
    def test_foreground_threshold_is_strict_and_does_not_change_functional_decoding(self):
        cs = [{'id': i} for i in [1, 2, 3, 4]]
        response = {'answers': {f'region_{i:02}': {'type': 'choice',
                    'probabilities': {'yes': p, 'no': 1-p}}
                    for i, p in [(1, .95), (2, .96), (3, .960001), (4, .99)]}}
        decisions = decode(response, cs)
        self.assertEqual([r['id'] for r in choose(cs, decisions, threshold=.96)], [3, 4])
        # The same raw scores still use the original yes/no rule for functional parts.
        self.assertEqual([d['region_id'] for d in decisions if d['selected']], [1, 2, 3, 4])

    def test_accept_all_positive_regions_and_reject_ties(self):
        cs = [{'id': i} for i in [1, 2, 3, 4]]
        ds = [{'region_id': i, 'probabilities': {'yes': p, 'no': 1-p}}
              for i, p in [(1, .8), (2, .9), (3, .5), (4, .1)]]
        self.assertEqual([r['id'] for r in choose(cs, ds)], [1, 2])
        with self.assertRaises(ValueError):
            choose(cs, ds[:-1])
        self.assertEqual(choose([], []), [])

    def test_two_images_one_question_target_membership(self):
        req = request(3, 'saw', 'cut')
        content = req['state']['messages'][0]['content']
        self.assertEqual([c['image_url']['url'] for c in content if c['type'] == 'image_url'],
                         ['rgb.png', 'point.png'])
        self.assertEqual(list(req['questions']), ['region_03'])
        self.assertIn('regardless of whether', content[0]['text'])
        self.assertIn('任意可见部位', chinese_prompt('saw', 'cut'))

    def test_saved_assets_do_not_paint_region_into_actual_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = Image.new('RGB', (40, 30), (85, 95, 110))
            rgb.save(root/'rgb.png')
            mask = np.zeros((30, 40), bool)
            mask[4:26, 5:35] = True
            save_mask(root/'raw/object_candidates/000.png', mask)
            candidate = {'raw_id': 0, 'quality': .8, 'area': int(mask.sum()),
                         'mask': 'raw/object_candidates/000.png'}
            write(root/'raw/objects.json', {'candidates': [candidate]})
            rs = prepare(root, rgb, [candidate], {'foreground_selector': {'minimum_fragment_pixels': 512}})
            self.assertEqual(len(rs), 1)
            r = rs[0]
            original = np.asarray(Image.open(root/r['input_directory']/'rgb.png'))
            marked = np.asarray(Image.open(root/r['input_directory']/'point.png'))
            self.assertTrue(np.array_equal(original, np.asarray(rgb)))
            x, y = r['point']['xy']
            yy, xx = np.nonzero(np.any(original != marked, axis=2))
            self.assertTrue(np.all(np.abs(xx-x) <= 3) and np.all(np.abs(yy-y) <= 3))
            self.assertTrue(load_mask(root/r['mask'])[y, x])
            self.assertTrue(read(root/'preprocessed/trace.json')['coverage_preserved'])


if __name__ == '__main__':
    unittest.main()
