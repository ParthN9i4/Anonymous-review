"""The original fixed CIFAR split. Test data is never opened by training/audit."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class Indexed(Dataset):
    def __init__(self, ds, indices): self.ds, self.indices = ds, list(map(int, indices))
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]; x,y = self.ds[j]
        return x,y,j


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def datasets(root, name='cifar10', regularized=False, download=False, synthetic=False):
    if synthetic:
        from torch.utils.data import TensorDataset
        g=torch.Generator().manual_seed(1234)
        ds=TensorDataset(torch.randn(32,3,32,32,generator=g),torch.randint(0,10,(32,),generator=g))
        return Indexed(ds,range(8,32)),Indexed(ds,range(8)),Indexed(ds,range(8,32))
    from torchvision import datasets as D, transforms as T
    cls={'cifar10':D.CIFAR10,'cifar100':D.CIFAR100}[name]
    mean=(.4914,.4822,.4465) if name=='cifar10' else (.5071,.4865,.4409)
    std=(.2470,.2435,.2616) if name=='cifar10' else (.2673,.2564,.2762)
    clean=T.Compose([T.ToTensor(),T.Normalize(mean,std)])
    aug=[T.RandomCrop(32,padding=4),T.RandomHorizontalFlip()]
    if regularized: aug.append(T.RandAugment(num_ops=2,magnitude=9))
    aug += [T.ToTensor(),T.Normalize(mean,std)]
    if regularized: aug.append(T.RandomErasing(p=.1))
    train=cls(root,train=True,download=download,transform=T.Compose(aug))
    plain=cls(root,train=True,download=download,transform=clean)
    perm=torch.randperm(50000,generator=torch.Generator().manual_seed(1234)).tolist()
    return Indexed(train,perm[5000:]),Indexed(plain,perm[:5000]),Indexed(plain,perm[5000:])


def loader(ds,batch=128,workers=2,shuffle=False,seed=42,limit=None):
    if limit: ds=Subset(ds,range(min(limit,len(ds))))
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,
        pin_memory=torch.cuda.is_available(),worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed),persistent_workers=False)
