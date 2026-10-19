"""Check conditional-query isolation, class indexing and probability decisions."""
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from affordance_jev.baselines.affordance_r1.predictor import Predictor
from affordance_jev.experiments.conditional_umd.requests import decode, request_for
from affordance_jev.baselines.affordance_r1.compare import query_score, summarize_queries


class ConditionalProtocol(unittest.TestCase):
    def test_binary_ties_reject_and_other_functions_are_allowed(self):
        answer={'answers':{'region_01':{'type':'choice','probabilities':{'yes':.5,'no':.5}},
                           'region_02':{'type':'choice','probabilities':{'yes':.8,'no':.2}}}}
        self.assertEqual([d['selected'] for d in decode(answer,[{'id':1},{'id':2}])],[False,True])
        answer['answers']['region_01']['probabilities']={'yes':.9,'no':.9}
        with self.assertRaises(ValueError): decode(answer,[{'id':1},{'id':2}])

    def test_r1_queries_only_requested_actions_with_native_ids(self):
        predictor=Predictor.__new__(Predictor)
        predictor.torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
        predictor.config={'image_side':840}
        names=['grasp','cut','scoop','contain','pound','support','wrap-grasp']
        predictor.classes=[{'id':i+1,'name':n} for i,n in enumerate(names)]
        predictor.processor=SimpleNamespace(apply_chat_template=lambda messages,**kw:messages[0]['content'][1]['text'])
        submitted=[]
        def generate(requests,*a,**kw):
            submitted.extend(requests)
            return list(range(len(requests)))
        predictor.model=SimpleNamespace(generate=generate)
        predictor.sampling=None
        seen=[]
        def segment(image, outputs, category, classes, merge):
            seen.append((outputs,[c['id'] for c in classes],merge))
            return None,None,[],{}
        predictor.segment=segment
        rgb=Image.new('RGB',(64,48))
        predictor.predict_many([rgb,rgb],['cup','saw'],[['contain','wrap-grasp'],['cut']])
        self.assertEqual(len(submitted),3)
        self.assertEqual(seen,[([0,1],[4,7],False),([2],[2],False)])
        submitted.clear(); seen.clear()
        predictor.predict_many([rgb],['cup'])
        self.assertEqual(len(submitted),7)
        self.assertEqual(seen[0][1],list(range(1,8)))
        self.assertTrue(seen[0][2])

    def test_scoring_counts_background_ignores_void_and_keeps_failures(self):
        rows=[]
        for cid in range(1,8):
            gt=np.array([[cid,0],[-1,cid]])
            rows.append(query_score(str(cid),cid,gt,np.array([[1,1],[1,0]],bool)))
        self.assertEqual(rows[0]['intersection'],1)
        self.assertEqual(rows[0]['union'],3)
        self.assertEqual(rows[0]['pred_pixels'],2)
        self.assertAlmostEqual(summarize_queries(rows)['mIoU_known_operations'],1/3)
        empty=query_score('failure',1,np.array([[1,0]]),np.zeros((1,2),bool))
        self.assertEqual(empty['iou'],0)
        self.assertEqual(empty['dice'],0)


if __name__=='__main__': unittest.main()
