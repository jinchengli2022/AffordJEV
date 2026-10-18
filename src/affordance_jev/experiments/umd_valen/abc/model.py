"""Differentiable A/B losses and C's mask-aligned, non-generative readout."""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F

from ....common import read, load_mask
from .data import NAMES


def canonical(logits, state, record):
    """Return physical region order and native class-ID order, not display order."""
    key_to_name = record.get('key_to_name', {n:n for n in NAMES})
    rows = {}
    for question, scores in zip(state.questions, logits):
        names = [key_to_name[k] for k in question.keys]
        rows[int(question.qid.split('_')[-1])-1] = scores[[names.index(n) for n in NAMES]]
    if set(rows) != set(range(len(rows))):
        raise ValueError('Expected contiguous, fully supervised region IDs')
    return torch.stack([rows[i] for i in range(len(rows))])


def composition(record, device):
    counts = torch.tensor([r['gt_pixel_counts'] for r in record['regions']], device=device, dtype=torch.float32)
    areas = counts.sum(-1)
    if not bool((areas > 0).all()): raise ValueError('Empty supervised region')
    return counts/areas[:,None], areas


def soft_ce(logits, targets):
    return -(targets*logits.float().log_softmax(-1)).sum(-1).mean()


def js_divergence(p, q):
    mean = (p+q)/2
    return ((p*(p.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(-1)
            +(q*(q.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(-1)).mean()/2


def pairing_loss(logits, fractions):
    """Region/definition ranking, only for >=90%-pure native GT labels."""
    purity, labels = fractions.max(-1)
    indices = torch.where(purity >= .9)[0]
    logp = logits.float().log_softmax(-1)
    terms = []
    for i in indices:
        c = labels[i]
        negative_classes = torch.arange(8, device=logits.device) != c
        terms.append(F.softplus(1-(logp[i,c]-logp[i,negative_classes])).mean())
        others = indices[labels[indices] != c]
        if len(others):
            terms.append(F.softplus(1-(logp[i,c]-logp[others,c])).mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0


def region_geometry(media_root, state, model_config):
    """Pool the FIRST unpainted RGB image's row-major merged visual tokens.

    Uses predicted masks only. Qwen smart-resizes the full crop without padding;
    INTER_AREA yields occupancy fractions on its patch_size*merge_size grid.
    """
    root = Path(media_root)
    if (root/'candidates.json').exists():
        crop = read(root/'foreground/metadata.json')['crop_xyxy']
        x0,y0,x1,y1 = crop
        candidates = read(root/'candidates.json')['candidates']
        masks = [load_mask(root/r['mask'])[y0:y1,x0:x1] for r in candidates]
        object_mask = load_mask(root/'foreground/object_mask.png')[y0:y1,x0:x1]
    else:
        # The warmed pipeline already transfers predicted geometry in one NPZ.
        with np.load(root/'geometry.npz',allow_pickle=False) as data:
            crop = data['crop_xyxy'].tolist()
            x0,y0,x1,y1 = crop
            masks = [m[y0:y1,x0:x1] for m in data['masks']]
            object_mask = data['foreground'][y0:y1,x0:x1]
        candidates = [{'id':i+1} for i in range(len(masks))]
    t,h,w = state.inputs['image_grid_thw'][0].tolist()
    merge = model_config.vision_config.spatial_merge_size
    assert t == 1 and h%merge == 0 and w%merge == 0
    gh,gw = h//merge,w//merge
    positions = torch.where(state.inputs['input_ids'][0] == model_config.image_token_id)[0]
    all_tokens = int((state.inputs['image_grid_thw'].prod(-1)//(merge*merge)).sum())
    assert len(positions) == all_tokens
    positions = positions[:gh*gw]
    assert torch.equal(positions, torch.arange(positions[0],positions[0]+gh*gw))
    def weights(mask):
        values = cv2.resize(mask.astype(np.float32), (gw,gh), interpolation=cv2.INTER_AREA).reshape(-1)
        if mask.any() and not values.sum() > 0: raise AssertionError('Mask lost during pooling')
        return values/max(float(values.sum()),1e-12)
    object_weights = weights(object_mask)
    nearest, pools = [], []
    for i, mask in enumerate(masks):
        if len(masks)>1:
            distance = distance_transform_edt(~mask)
            j = min((j for j in range(len(masks)) if j!=i), key=lambda j:(float(distance[masks[j]].min()),j))
            neighbor = weights(masks[j])
        else:
            j,neighbor = None,np.zeros(gh*gw,np.float32)
        nearest.append(None if j is None else j+1)
        pools.append(np.stack([weights(mask),neighbor,object_weights]))
    return {'positions':positions, 'weights':torch.from_numpy(np.stack(pools)),
            'audit':{'source_image':'foreground/rgb_crop.png','grid_hw':[gh,gw],
                     'crop_xyxy':crop,'region_ids':[r['id'] for r in candidates],
                     'nearest_region_ids':nearest,'uses_ground_truth':False,
                     'token_order':'row-major merged 2x2 visual patches',
                     'mask_pixels':[int(m.sum()) for m in masks]}}


class ABCModel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.kind = config['method']
        self.auxiliary = deepcopy(model.head) if self.kind in ['b0','b1'] else None
        self.region_head = deepcopy(model.head) if self.kind in ['c0','c1'] else None
        if self.region_head is not None:
            # Exactly preserve the SFT warmup function before learning residuals.
            nn.init.zeros_(self.region_head.output[-1].weight)
            nn.init.zeros_(self.region_head.output[-1].bias)

    def encode(self, state, geometry=None):
        device = next(self.model.backbone.parameters()).device
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in state.inputs.items()}
        hidden = self.model.backbone(**inputs,use_cache=False,return_dict=True).last_hidden_state
        features = self.model._state_features(hidden,state)
        logits = [self.model.score_features(f) for f in features]
        extra = [self.model.score_features(f,head=self.auxiliary) for f in features] if self.auxiliary is not None else None
        if self.region_head is not None:
            if geometry is None: raise ValueError('C requires predicted-mask pooling weights')
            visual = hidden[0,geometry['positions'].to(device)].float()
            pooled = geometry['weights'].to(device) @ visual
            for index,question in enumerate(state.questions):
                physical = int(question.qid.split('_')[-1])-1
                target,neighbor,obj = pooled[physical]
                if self.kind == 'c0':
                    neighbor,obj = torch.zeros_like(neighbor),torch.zeros_like(obj)
                definition = features[index][0][:,2,:].float()
                role_features = torch.stack([target.expand_as(definition),neighbor.expand_as(definition),
                                             definition,obj.expand_as(definition)],dim=1)
                logits[index] = logits[index]+self.region_head.score_features(role_features)
        return logits,extra

    def forward(self, original, transformed=None):
        state,row,geometry = original
        native,_aux = self.encode(state,geometry)
        z = canonical(native,state,row)
        fractions,areas = composition(row,z.device)
        ce = F.cross_entropy(z,fractions.argmax(-1))
        zero = z.sum()*0
        pairing,invariance,soft,consistency = zero,zero,zero,zero
        if transformed is not None:
            other_state,other_row,other_geometry = transformed
            native2,aux2 = self.encode(other_state,other_geometry)
            z2 = canonical(native2,other_state,other_row)
            fractions2,areas2 = composition(other_row,z.device)
            ce = (ce+F.cross_entropy(z2,fractions2.argmax(-1)))/2
            if self.kind in ['a0','a1']:
                inverse = np.argsort(other_row['original_region_indices']).tolist()
                z2 = z2[inverse]
                if not torch.equal(fractions,fractions2[inverse]):
                    raise AssertionError('A view mapping changed physical supervision')
                pairing = (pairing_loss(z,fractions)+pairing_loss(z2,fractions))/2
                invariance = js_divergence(z.softmax(-1),z2.softmax(-1))
            elif self.kind in ['b0','b1']:
                u = canonical(_aux,state,row)
                u2 = canonical(aux2,other_state,other_row)
                soft = (soft_ce(u,fractions)+soft_ce(u2,fractions2))/2
                p,p2 = u.softmax(-1),u2.softmax(-1)
                errors = []
                for relation in other_row['relations']:
                    i,j = relation['original'],relation['transformed']
                    area = areas[i].sum()
                    if not torch.equal(area,areas2[j].sum()): raise AssertionError('Unequal union area')
                    before = (areas[i,None]*p[i]).sum(0)/area
                    after = (areas2[j,None]*p2[j]).sum(0)/area
                    errors.append(js_divergence(before[None],after[None]))
                consistency = torch.stack(errors).mean()
        loss = ce+self.config['pair_weight']*pairing+self.config['invariance_weight']*invariance
        loss = loss+self.config['composition_weight']*soft+self.config['composition_consistency_weight']*consistency
        metrics = torch.stack([ce.detach(),pairing.detach(),invariance.detach(),soft.detach(),consistency.detach()])
        return loss,metrics
