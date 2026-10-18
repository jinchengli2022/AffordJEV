"""Scientific invariants: physical IDs, area conservation, and image-token layout."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from affordance_jev.common import write,save_mask
from affordance_jev.experiments.umd_valen.abc.data import NAMES,transformed_geometry
from affordance_jev.experiments.umd_valen.abc.model import canonical,js_divergence,soft_ce,region_geometry


class RegionTrainingInvariants(unittest.TestCase):
    def test_physical_identity_survives_question_and_option_permutations(self):
        expected=torch.arange(24,dtype=torch.float32).reshape(3,8)
        order=[2,0,1];keys=['role_'+str(i) for i in range(8)]
        aliases=dict(zip(keys,reversed(NAMES)))
        options=[4,1,7,0,2,6,5,3]
        questions=[];logits=[]
        for physical in order:
            displayed=[keys[i] for i in options]
            questions.append(SimpleNamespace(qid=f'region_{physical+1:02}',keys=displayed))
            logits.append(expected[physical,[NAMES.index(aliases[k]) for k in displayed]])
        actual=canonical(logits,SimpleNamespace(questions=questions),{'key_to_name':aliases})
        self.assertTrue(torch.equal(expected,actual))

    def test_geometric_children_conserve_valid_class_counts(self):
        masks=[np.zeros((20,30),bool) for _ in range(4)]
        masks[0][2:18,2:10]=True;masks[1][2:18,10:15]=True
        masks[2][2:18,16:23]=True;masks[3][2:18,24:28]=True
        gt=np.indices((20,30)).sum(0)%8;gt[4:6,5:8]=-1
        for variant in range(4):
            transformed,relations=transformed_geometry(masks,variant)
            self.assertLessEqual(np.stack(transformed).sum(0).max(),1)
            self.assertTrue(np.array_equal(np.logical_or.reduce(masks),np.logical_or.reduce(transformed)))
            def counts(items,indices):
                return sum(np.bincount(gt[items[i]&(gt>=0)],minlength=8) for i in indices)
            for relation in relations:
                self.assertTrue(np.array_equal(counts(masks,relation['original']),
                                               counts(transformed,relation['transformed'])))

    def test_composition_is_area_fraction_not_classification_confidence(self):
        a=torch.tensor([[.8,.2,0,0,0,0,0,0]])
        b=torch.tensor([[0.,0.,1.,0,0,0,0,0]])
        parent=(a+3*b)/4
        self.assertAlmostEqual(float(parent[0,2]),.75)
        self.assertAlmostEqual(float(js_divergence(parent,parent)),0.)
        self.assertGreater(float(js_divergence(parent,(a+b)/2)),0.)
        # Constant predictions can satisfy consistency, but cannot evade soft supervision.
        uniform=torch.zeros(1,8)
        correct=parent.clamp_min(1e-7).log()
        self.assertGreater(float(soft_ce(uniform,parent)),float(soft_ce(correct,parent)))

    def test_consistency_backpropagates_to_both_predictions(self):
        left=torch.tensor([[2.,0.,-1.]],requires_grad=True)
        right=torch.tensor([[-1.,1.,0.]],requires_grad=True)
        js_divergence(left.softmax(-1),right.softmax(-1)).backward()
        self.assertGreater(float(left.grad.abs().sum()),0.)
        self.assertGreater(float(right.grad.abs().sum()),0.)

    def test_first_rgb_pooling_respects_crop_and_row_major_tokens(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'foreground').mkdir()
            write(root/'foreground/metadata.json',{'crop_xyxy':[2,2,10,10]})
            masks=[np.zeros((12,12),bool) for _ in range(3)]
            masks[0][2:6,2:6]=True;masks[1][2:6,6:10]=True;masks[2][6:10,2:10]=True
            candidates=[]
            for i,mask in enumerate(masks,1):
                path=f'regions/masks/{i:02}.png';save_mask(root/path,mask)
                candidates.append({'id':i,'mask':path})
            write(root/'candidates.json',{'candidates':candidates})
            save_mask(root/'foreground/object_mask.png',np.logical_or.reduce(masks))
            state=SimpleNamespace(inputs={'input_ids':torch.tensor([[0,99,99,99,99,0,99,99,99,99]]),
                                           'image_grid_thw':torch.tensor([[1,4,4],[1,4,4]])})
            config=SimpleNamespace(image_token_id=99,vision_config=SimpleNamespace(spatial_merge_size=2))
            geometry=region_geometry(root,state,config)
            self.assertEqual(geometry['positions'].tolist(),[1,2,3,4])
            self.assertTrue(torch.equal(geometry['weights'][0,0],torch.tensor([1.,0,0,0])))
            self.assertTrue(torch.equal(geometry['weights'][1,0],torch.tensor([0.,1,0,0])))
            self.assertTrue(torch.equal(geometry['weights'][2,0],torch.tensor([0.,0,.5,.5])))
            self.assertTrue(torch.equal(geometry['weights'][0,2],torch.full((4,),.25)))
            self.assertFalse(geometry['audit']['uses_ground_truth'])
            # The real-time benchmark transfers exactly the same geometry in NPZ.
            np.savez(root/'geometry.npz',crop_xyxy=np.array([2,2,10,10]),
                     masks=np.stack(masks),foreground=np.logical_or.reduce(masks))
            (root/'candidates.json').unlink()
            transferred=region_geometry(root,state,config)
            self.assertTrue(torch.equal(transferred['positions'],geometry['positions']))
            self.assertTrue(torch.equal(transferred['weights'],geometry['weights']))


if __name__=='__main__': unittest.main()
