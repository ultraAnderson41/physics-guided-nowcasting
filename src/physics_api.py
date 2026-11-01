"""Thin path adapter; original model arithmetic and state_dict keys unchanged."""
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import torch
LEGACY=Path(__file__).resolve().parent/'legacy'
def load(name):
    spec=importlib.util.spec_from_file_location('release_legacy_'+name,LEGACY/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def build_model(dem=None,checkpoint=None,synthetic_flat=False):
    if dem is None and not synthetic_flat:raise ValueError('Explicit DEM required; no silent flat fallback')
    if dem is not None and not Path(dem).is_file():raise FileNotFoundError(dem)
    if checkpoint is not None and not Path(checkpoint).is_file():raise FileNotFoundError(checkpoint)
    with tempfile.TemporaryDirectory() as td:
        if dem is None:torch.save(torch.zeros(64,64),Path(td)/'dem_64x64.pt')
        else:shutil.copyfile(dem,Path(td)/'dem_64x64.pt')
        previous=Path.cwd()
        try:
            os.chdir(td);model=load('model').WeatherPredictor()
        finally:os.chdir(previous)
    if checkpoint is not None:model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    return model
def predict(model,inputs):
    if inputs.ndim!=5 or tuple(inputs.shape[1:])!=(10,2,64,64):raise ValueError('Expected [B,10,2,64,64], PWV then Radar')
    if not torch.isfinite(inputs).all() or inputs.min()<0 or inputs.max()>1:raise ValueError('Expected finite normalized inputs in [0,1]')
    values=model(inputs,pred_steps=20)
    return dict(zip(('prediction','motion','radar_source','source_cap','psi','pwv_prediction'),values))
