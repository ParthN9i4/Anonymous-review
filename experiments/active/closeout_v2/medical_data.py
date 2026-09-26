"""MedMNIST loader, copied verbatim (class body) from archive/submission_v2/src/pvit/data.py.

Official 28 px images resized bilinearly to 32, RGB, normalized with mean/std 0.5, exactly
as the medical_replication_v1 checkpoints were trained and converted. Items are
(image, label, index) like pvit_lab.data.Indexed, so the same loader works.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class Medical(Dataset):
    def __init__(self,root,name,split,augment=False,indices=None):
        if split not in ('train','val','test'):raise ValueError(split)
        with np.load(Path(root)/(name+'.npz'),allow_pickle=False) as a:
            self.x=a[split+'_images'];self.y=a[split+'_labels'].reshape(-1).astype(np.int64)
        self.ids=np.arange(len(self.y)) if indices is None else np.asarray(indices)
        ops=[T.Resize((32,32),interpolation=T.InterpolationMode.BILINEAR,antialias=True)]
        if augment:ops += [T.RandomCrop(32,padding=4),T.RandomHorizontalFlip()]
        self.transform=T.Compose(ops+[T.ToTensor(),T.Normalize((.5,)*3,(.5,)*3)])
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        j=int(self.ids[i]);return self.transform(Image.fromarray(self.x[j]).convert('RGB')),int(self.y[j]),j
