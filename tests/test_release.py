import importlib.util, pathlib, subprocess, sys, tempfile, unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/legacy')]
from physics_api import build_model,predict,load
torch.set_num_threads(2)
class ReleaseTests(unittest.TestCase):
 def test_imports(self):
  for name in ('dataset','model','train_phase1','train_phase2','test'):load(name)
 def test_forward_and_backward(self):
  torch.manual_seed(2);m=build_model(synthetic_flat=True);r=predict(m,torch.rand(1,10,2,64,64))
  self.assertEqual(tuple(r['prediction'].shape),(1,20,2,64,64));self.assertEqual(tuple(r['motion'].shape),(1,20,2,64,64))
  for name in ('radar_source','source_cap','psi','pwv_prediction'):self.assertEqual(tuple(r[name].shape),(1,20,1,64,64))
  self.assertTrue(all(torch.isfinite(v).all() for v in r.values()));r['prediction'].mean().backward()
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.unet.parameters()))
  self.assertIsNone(m.warper.alpha.grad);self.assertIsNone(m.warper.beta.grad);self.assertIsNone(m.warper.oro_gamma.grad)
 def test_gate_original_semantics(self):
  gate=load('model').PhysicalGating;raw=torch.tensor([-2.,.2,2.],requires_grad=True);cap=torch.ones(3,requires_grad=True)
  y=gate.apply(raw,cap);self.assertTrue(torch.equal(y,torch.tensor([-1.,.2,1.])))
  y.sum().backward();self.assertTrue(torch.allclose(raw.grad,torch.sigmoid((cap-raw.abs())/.1)));self.assertIsNone(cap.grad)
 def test_cardinal_warp(self):
  w=build_model(synthetic_flat=True).warper;x=torch.zeros(1,1,64,64);x[0,0,30,30]=1
  for u,v,row,col in ((1,0,30,31),(-1,0,30,29),(0,1,31,30),(0,-1,29,30),(0,0,30,30)):
   y=w.warp_frame(x,torch.ones_like(x)*u,torch.ones_like(x)*v);self.assertEqual(int(y.flatten().argmax()),row*64+col)
 def test_metrics(self):
  import numpy as np
  metrics=load('test');x=np.zeros((64,64));x[20:30,20:30]=20
  self.assertEqual(metrics.calc_pixel_metrics(x,x,10),(1,1,0));self.assertAlmostEqual(metrics.calc_fss(x,x,10),1)
 def test_dataset_resize_channel_order_and_unchecked_gaps(self):
  import numpy as np
  from PIL import Image
  from datetime import datetime,timedelta
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td)
   for kind,value in (('pwv',0),('radar',255)):
    p=root/kind/'2025-05'/'2025-05-01';p.mkdir(parents=True)
    for i in range(31):
     name=(datetime(2025,5,1)+timedelta(minutes=12*i)).strftime('%Y-%m-%d-%H-%M-%S.png')
     Image.fromarray(np.full((66,70),value,np.uint8)).save(p/name)
   ds=load('dataset').WeatherDataset(root/'pwv',root/'radar');x,y=ds[0]
   self.assertEqual(tuple(x.shape),(10,2,64,64));self.assertEqual(tuple(y.shape),(20,2,64,64))
   self.assertTrue(torch.equal(x[:,0],torch.ones_like(x[:,0])));self.assertTrue(torch.equal(x[:,1],torch.zeros_like(x[:,1])))
   self.assertEqual(len(ds),1) # 12-minute gaps accepted by legacy design, NOT repaired.
 def test_discriminator(self):
  d=load('model').SpatiotemporalDiscriminator();x=torch.rand(2,1,64,64,requires_grad=True)
  out=d(x,torch.zeros_like(x));self.assertEqual(tuple(out.shape),(2,1,7,7));out.mean().backward();self.assertTrue(torch.isfinite(x.grad).all())
 def test_cli_help(self):
  for name in ('run_legacy.py','infer.py'):
   r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/name),'--help'],capture_output=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_missing_asset_refusal(self):
  with self.assertRaises(ValueError):build_model()
  with self.assertRaises(FileNotFoundError):build_model('nonexistent_dem.pt')
if __name__=='__main__':unittest.main()
