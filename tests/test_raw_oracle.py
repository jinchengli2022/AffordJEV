import itertools
import unittest

import numpy as np

from affordance_jev.evaluation.raw_oracle import pixel_atoms, single_scores, union_oracle


class RawOracleTests(unittest.TestCase):
    def exact(self, masks, gt, cid):
        best = 0.
        for chosen in itertools.product([False,True],repeat=len(masks)):
            pred = masks[np.array(chosen)].any(0)
            valid, target = gt >= 0, gt == cid
            best = max(best, (pred & target).sum()/((pred | target)&valid).sum())
        return best

    def test_overlapping_masks_match_exhaustive_enumeration(self):
        rng = np.random.default_rng(1901)
        for i in range(30):
            masks = rng.random((7,8,9)) < rng.uniform(.1,.7)
            gt = rng.integers(-1,3,(8,9)); gt[0,0] = 1
            membership,counts,gt_counts = pixel_atoms(masks,gt)
            exact = self.exact(masks,gt,1)
            result = union_oracle(membership,counts,1,int(gt_counts[1]),seconds=4.)
            self.assertTrue(result['certified'],result)
            self.assertAlmostEqual(result['lower'],exact,places=7)
            self.assertLessEqual(exact,result['upper']+1e-7)
            scores = single_scores(membership,counts,1,int(gt_counts[1]))
            self.assertLessEqual(float(scores.max()),result['lower']+1e-7)

    def test_zero_budget_returns_valid_bounds_with_void_and_duplicates(self):
        gt = np.array([[1,1,0],[0,1,-1]])
        masks = np.array([[[1,0,0],[0,1,1]],[[0,1,1],[0,0,0]],
                          [[1,0,0],[0,1,0]],[[0,0,1],[1,0,0]]],bool)
        a,c,g = pixel_atoms(masks,gt)
        out = union_oracle(a,c,1,int(g[1]),seconds=0.)
        exact = self.exact(masks,gt,1)
        self.assertLessEqual(out['lower'],exact+1e-8)
        self.assertGreaterEqual(out['upper']+1e-8,exact)

    def test_no_proposal_covers_target(self):
        gt = np.array([[1,0],[0,0]])
        masks = np.array([[[0,1],[0,0]],[[0,0],[0,0]]],bool)
        a,c,g = pixel_atoms(masks,gt)
        out = union_oracle(a,c,1,int(g[1]))
        self.assertEqual(out['lower'],0.)
        self.assertEqual(out['upper'],0.)
        self.assertTrue(out['certified'])


if __name__ == '__main__':
    unittest.main()
