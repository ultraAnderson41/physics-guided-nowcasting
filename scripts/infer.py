"""Explicit trained checkpoint inference; no diffusion model is implemented."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--dem',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import numpy as np,torch
    from physics_api import build_model,predict
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    model=build_model(a.dem,a.checkpoint).eval();x=torch.from_numpy(np.load(a.input,allow_pickle=False)).float()
    with torch.inference_mode():r=predict(model,x)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f:np.savez_compressed(f,**{k:v.numpy() for k,v in r.items()})
if __name__=='__main__':main()
