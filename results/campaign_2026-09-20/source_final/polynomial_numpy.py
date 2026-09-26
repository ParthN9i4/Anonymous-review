import numpy as np

def polynomial(x,p):return np.polynomial.polynomial.polyval((x-p['center'])/p['radius'],p['coefficients'])
def reciprocal(x,p):
 scale=2/(p['lo']+p['hi']);e=1-x*scale;y=1+e
 for _ in range(p['steps']):e=e*e;y=y*(1+e)
 return y*scale

def softmax_poly(s,p):
 z=s-s.mean(-1,keepdims=True);u=polynomial(z,p['root'])**2+p['floor'];d=u.sum(-1,keepdims=True)
 with np.errstate(over='ignore',invalid='ignore'):return u*reciprocal(d,p['reciprocal'])

def softmax_exact(s):
 z=s-s.max(-1,keepdims=True);e=np.exp(z);return e/e.sum(-1,keepdims=True)
