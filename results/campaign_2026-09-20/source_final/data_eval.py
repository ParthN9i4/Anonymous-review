import torch
from torch.utils.data import Dataset,Subset
from pvit_lab.data import datasets,loader,Indexed
from protocol import *

class Shift(Dataset):
 def __init__(self,base,kind):self.base=base;self.kind=kind
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  x,y,j=self.base[i];mean=torch.tensor([.4914,.4822,.4465])[:,None,None];std=torch.tensor([.2470,.2435,.2616])[:,None,None]
  raw=x*std+mean
  if self.kind=='noise':raw=raw+torch.randn(raw.shape,generator=torch.Generator().manual_seed(SHIFT_SEED+int(j)))*.03
  elif self.kind=='brightness':raw=raw*.7
  else:raise ValueError(self.kind)
  return (raw.clamp(0,1)-mean)/std,y,j

def subsets(root,fit_n=FIT_N,synthetic=False,evaluation=True):
 _,val,clean=datasets(root,'cifar10',synthetic=synthetic)
 if synthetic:
  return Subset(clean,range(8)),Subset(clean,range(8,16)),dict(clean=val,noise=Shift(val,'noise'),brightness=Shift(val,'brightness'))
 if not evaluation:return Subset(clean,range(fit_n)),Subset(clean,range(GATE_OFFSET,GATE_OFFSET+GATE_N)),{}
 from torchvision import datasets as D,transforms as T
 transform=T.Compose([T.ToTensor(),T.Normalize((.4914,.4822,.4465),(.2470,.2435,.2616))])
 test=Indexed(D.CIFAR10(root,train=False,download=False,transform=transform),range(TEST_N))
 shiftbase=Subset(test,range(SHIFT_N))
 return Subset(clean,range(fit_n)),Subset(clean,range(GATE_OFFSET,GATE_OFFSET+GATE_N)),dict(clean=test,noise=Shift(shiftbase,'noise'),brightness=Shift(shiftbase,'brightness'))
