import unittest

import numpy as np

from affordance_jev.experiments.valen_foreground.selection import eligible_candidates, choose_foreground
from affordance_jev.experiments.conditional_umd.requests import decode


class ForegroundSelection(unittest.TestCase):
    def test_exact_duplicates_only_no_geometric_ranking(self):
        masks = np.zeros((4, 10, 10), bool)
        masks[0, :3, :3] = True
        masks[1] = masks[0]
        masks[2, :3, :4] = True  # Overlap must not count as an identical mask.
        masks[3, 7:, 7:] = True  # Off-center candidate must remain eligible.
        records = [{'raw_id': i, 'quality': q, 'area': int(m.sum())}
                   for i, (m, q) in enumerate(zip(masks, [.2, .3, .2, .2]))]
        kept, trace = eligible_candidates(records, masks, {'minimum_object_pixels': 3, 'minimum_object_score': .01})
        self.assertEqual([r['raw_id'] for r in kept], [1, 2, 3])
        self.assertEqual(trace, [{'raw_id': 0, 'reason': 'identical_mask', 'retained_raw_id': 1}])

    def test_global_selection_probability_then_quality_then_raw_id(self):
        candidates = [{'id': i+1, 'raw_id': i, 'quality': q} for i, q in enumerate([.9, .1, .8, .8])]
        # Decisions may come from different groups and a different response order.
        decisions = [{'region_id': i+1, 'probabilities': {'yes': p, 'no': 1-p}}
                     for i, p in reversed(list(enumerate([.7, .8, .8, .8])))]
        self.assertEqual(choose_foreground(candidates, decisions)['raw_id'], 2)
        with self.assertRaises(ValueError):
            choose_foreground(candidates, decisions[:-1])

    def test_all_no_or_ties_abstain_even_when_object_quality_high(self):
        candidates = [{'id': 1, 'raw_id': 0, 'quality': .99}, {'id': 2, 'raw_id': 1, 'quality': .99}]
        response = {'answers': {f'region_{i:02}': {'type': 'choice', 'probabilities': {'yes': p, 'no': 1-p}}
                                for i, p in [(1, .4), (2, .5)]}}
        self.assertIsNone(choose_foreground(candidates, decode(response, candidates)))


if __name__ == '__main__':
    unittest.main()
