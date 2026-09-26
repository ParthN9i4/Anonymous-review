import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from polynomial import calibrate,convert,fit_poly
from pvit_lab.model import ViT,legacy_spec
from data_eval import subsets
from pvit_lab.data import loader
from worker import replay,load
from io_utils import sha
class Tests(unittest.TestCase):
 def test_frozen_conversion(self):
  torch.set_num_threads(2);torch.manual_seed(1);m=ViT(legacy_spec('000')).eval()
  fit,gate,sets=subsets('.',synthetic=True);p,ids=calibrate(m,loader(fit,workers=0),'cpu','budget');a,r=convert(m,p,'all');b,s=convert(m.double(),p,'all');m.float();a.float()
  result,fixture=replay(m,a,b,r,s,loader(gate,workers=0),'cpu',capture=True)
  self.assertEqual(len(result['labels']),len(gate));self.assertIn('b5_converted_scores',fixture)
  self.assertTrue(all(torch.equal(v,dict(a.named_parameters())[k]) for k,v in m.named_parameters()));self.assertFalse(any(v.requires_grad for v in a.parameters()))
  for mask in ('gelu','attention','norm'):
   a,r=convert(m,p,mask);x=next(iter(loader(gate,workers=0)))[0];r.begin(len(x),x.device);self.assertEqual(a(x).shape,(len(x),10))
 def test_power_native_parity(self):
  import native_reference as n
  torch.set_num_threads(2)
  for arm in ('P','Q'):
   native=n.ConfigurableDeiT(num_classes=10,poly_gelu=False,poly_softmax=True,poly_norm=(arm=='Q'))
   for b in native.blocks:b.attn_act=n.PowerNormAttn()
   native.eval()
   with tempfile.TemporaryDirectory() as d:
    f=Path(d)/'x.pth';torch.save({'state_dict':native.state_dict()},f)
    m=load(dict(name='test',kind='power',arm=arm,checkpoint=str(f),checkpoint_sha256=sha(f)))
    x=torch.randn(2,3,32,32)
    with torch.no_grad():self.assertTrue(torch.equal(native(x),m(x)))
if __name__=='__main__':unittest.main()
