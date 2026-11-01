"""SYNTHETIC random weights + explicitly flat DEM; not a skill demonstration."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from physics_api import build_model,predict
torch.set_num_threads(2);torch.manual_seed(7)
model=build_model(synthetic_flat=True).eval()
with torch.inference_mode():r=predict(model,torch.rand(1,10,2,64,64))
print('SYNTHETIC ONLY', {k:tuple(v.shape) for k,v in r.items()})
