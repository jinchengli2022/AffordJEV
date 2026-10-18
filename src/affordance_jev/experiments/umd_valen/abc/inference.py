"""Install C's explicit-region scorer after the ordinary LoRA/head loader."""
import importlib
from pathlib import Path
from types import MethodType

import torch
from safetensors.torch import load_file

from ....common import read,write
from .model import ABCModel,region_geometry


def install_region_head(model, checkpoint):
    checkpoint=Path(checkpoint)
    config=read(checkpoint/'training_config.json')
    if config.get('method') not in ['c0','c1']: return
    network=ABCModel(model,config).to(next(model.parameters()).device)
    saved=load_file(checkpoint/'extra_heads.safetensors')
    expected={k for k in network.state_dict() if not k.startswith('model.')}
    if set(saved)!=expected: raise ValueError('Explicit-region checkpoint mismatch')
    network.region_head.load_state_dict({k.removeprefix('region_head.'):v for k,v in saved.items()},strict=True)
    network.eval()
    module=importlib.import_module(type(model).__module__.rsplit('.',1)[0]+'.responses')

    @torch.inference_mode()
    def predict(self,request,*,media_root='.',execution=None,temperature=1.,processor=None):
        request=request['request'] if 'request' in request else request
        compiler=self.make_compiler(media_root,execution,processor)
        state=compiler.compile({'request':request})
        geometry=region_geometry(media_root,state,self.config)
        logits,_=network.encode(state,geometry)
        answers={q.qid:module.answer(q,z,temperature[q.kind] if isinstance(temperature,dict) else temperature)
                 for q,z in zip(state.questions,logits)}
        audit_path=Path(media_root)/'valen_compiled'
        if audit_path.exists(): write(audit_path/'region_pooling.json',geometry['audit'])
        return {'model':self.model_name,'answers':answers,
                'usage':{'input_tokens':state.logical_tokens,'output_tokens':0},
                'internal_usage':{'compute_tokens':state.compute_tokens},
                'explicit_region_head':config['method']}
    # Keep the wrapper in this closure, not as a model child (which would cycle).
    model.predict=MethodType(predict,model)
