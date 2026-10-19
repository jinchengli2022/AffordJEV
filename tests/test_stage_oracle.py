import itertools
import unittest

import numpy as np

from affordance_jev.experiments.candidate_stage_diagnosis import score_masks


class StageOracleTests(unittest.TestCase):
    def test_uncovered_gt_remains_in_denominator(self):
        gt=np.array([[1,1,1],[0,0,-1]])
        masks=np.array([[[1,0,0],[0,0,1]]],bool)
        result=score_masks(masks,gt,3.)[1]
        self.assertAlmostEqual(result['lower'],1/3)
        self.assertTrue(result['certified'])
        self.assertEqual(result['gt_pixels'],3)
        self.assertEqual(result['covered_gt_pixels'],1)

    def test_empty_stage_is_zero(self):
        gt=np.array([[1,0],[2,0]])
        for masks in [np.zeros((0,2,2),bool),np.zeros((4,2,2),bool)]:
            results=score_masks(masks,gt,3.)
            self.assertEqual(set(results),{1,2})
            self.assertTrue(all(r['lower']==r['upper']==0 for r in results.values()))

    def test_sparse_optimization_matches_full_pixel_exhaustion(self):
        rng=np.random.default_rng(1910)
        for _ in range(12):
            masks=rng.random((5,9,12))<.15
            gt=rng.integers(-1,3,(9,12));gt[0,0]=1;gt[0,1]=2
            scores=score_masks(masks,gt,3.)
            for cid,result in scores.items():
                best=0.
                for bits in itertools.product([False,True],repeat=len(masks)):
                    pred=masks[np.array(bits)].any(0)
                    target=gt==cid
                    best=max(best,float((pred&target).sum()/((pred|target)&(gt>=0)).sum()))
                self.assertAlmostEqual(result['lower'],best,places=7)
                self.assertLessEqual(best,result['upper']+1e-7)
                self.assertTrue(result['certified'])


if __name__=='__main__':
    unittest.main()
