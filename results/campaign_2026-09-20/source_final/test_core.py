import unittest
import numpy as np
from assessment import outcomes,detectors,performance
from protocol import schedule
from polynomial_numpy import reciprocal,softmax_poly
class Tests(unittest.TestCase):
 def test_plan(self):
  p=schedule();self.assertEqual([len(p[k]) for k in ('conversion','endpoints','fhe')],[26,15,12]);self.assertEqual(len({j['name'] for j in p['conversion']}),26)
 def test_reference_invalid_excluded(self):
  r=np.array([[2,0],[np.nan,0],[0,2.]]);p=np.array([[np.nan,0],[0,2],[2,0.]])
  t=outcomes(r,p,[0,0,1]);self.assertEqual(t['conversion_failure'].tolist(),[True,False,True])
 def test_all_warn_is_not_zero_risk(self):
  d=performance(np.ones(3,bool),np.array([1,0,1],bool),np.ones(3,bool));self.assertIsNone(d['accepted_failure_rate']);self.assertEqual(d['accepted_fraction'],0)
 def test_precision_can_miss_shared_error(self):
  wrong=np.array([[0.,9.]]);d,_,_=detectors(wrong,wrong,[False],False);self.assertFalse(d['range_and_precision'][0]);self.assertTrue(outcomes([[9.,0.]],wrong,[0])['conversion_failure'][0])
 def test_reciprocal_bound(self):
  p=dict(lo=1.,hi=3.,steps=2);x=np.linspace(1,3,100);err=np.abs(1-x*reciprocal(x,p));self.assertLessEqual(err.max(),.5**8+1e-14)
if __name__=='__main__':unittest.main()
