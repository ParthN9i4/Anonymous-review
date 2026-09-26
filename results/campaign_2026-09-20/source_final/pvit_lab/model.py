"""Checkpoint-compatible compact ViT with explicit, instrumentable replacements.

PowerSoftmax: arXiv:2410.09457v2 equations 5-7 (paper reimplementation).
BPMax/BatchLN: operator adaptation checked against thrudgelmir/Powerformer
commit 8fc75d1fa316016bb46f9540940e05ad4608cdd6; not a BERT reproduction.
"""
from dataclasses import dataclass, asdict
import copy
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelSpec:
    num_classes: int = 10
    img_size: int = 32
    patch_size: int = 4
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0
    attention: str = 'softmax'
    activation: str = 'gelu'
    norm: str = 'ln'
    final_norm: str = 'same'
    power: int = 4
    shift: float = 0.0
    epsilon: float = 1e-3
    delta: float = 1e-6
    floor: float = 1e-3
    norm_scale: float = 1.2
    drop_path: float = 0.0


class PolyGELU(nn.Module):
    def __init__(self):
        super().__init__()
        x = torch.linspace(-3, 3, 10000)
        c = torch.linalg.lstsq(torch.stack([x*x, x, torch.ones_like(x)], 1), F.gelu(x)).solution
        self.a, self.b, self.c = [nn.Parameter(v.clone()) for v in c]

    def forward(self, x):
        return self.a*x*x + self.b*x + self.c


class Attention(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.kind, self.power = spec.attention, spec.power
        self.shift, self.eps, self.delta, self.floor = spec.shift, spec.epsilon, spec.delta, spec.floor
        assert self.power >= 2 and self.power % 2 == 0
        assert self.eps > 0 and self.delta > 0 and self.floor > 0
        if self.kind in ('raw', 'raw_relu', 'raw_l1', 'clamp', 'relu_row', 'clamp_row', 'clamp01_row'):
            self.a = nn.Parameter(torch.tensor(.05))
            self.b = nn.Parameter(torch.tensor(.5))
            self.c = nn.Parameter(torch.tensor(.25))
        if self.kind in ('bpmax', 'square_fixed'):
            tokens = (spec.img_size // spec.patch_size)**2 + 1
            self.register_buffer('running_denominator', torch.ones(1, spec.num_heads, tokens, 1))
        self.observer = None
        self.inverse = None  # Optional polynomial inverse; otherwise exact S-mode division.

    def forward(self, s):
        if self.kind == 'softmax':
            return s.softmax(-1)
        if self.kind in ('raw', 'raw_relu', 'raw_l1', 'clamp', 'relu_row', 'clamp_row', 'clamp01_row'):
            u = self.a*s*s + self.b*s + self.c
            if self.kind == 'raw': return u
            if self.kind == 'raw_relu': return F.relu(u)
            if self.kind == 'clamp': return u.clamp(0, 1)
            if self.kind == 'raw_l1':
                d = u.abs().sum(-1,keepdim=True)+1e-6
            elif self.kind == 'relu_row':
                u = F.relu(u); d = u.sum(-1, keepdim=True) + 1e-6
            elif self.kind == 'clamp01_row':
                u = u.clamp(0,1); d = u.sum(-1,keepdim=True) + 1e-6
            else:
                u = u.clamp_min(1e-6); d = u.sum(-1, keepdim=True) + 1e-8
        elif self.kind in ('power_stable', 'power_raw'):
            # With fixed epsilon these are DISTINCT functions: do not silently
            # remove max scaling at deployment and invoke homogeneity.
            x = s
            if self.kind == 'power_stable':
                c = x.abs().amax(-1, keepdim=True) + self.delta
                if self.observer: self.observer('scale', c)
                x = x/c
            u = x.pow(self.power)
            # Exactly equivalent to u/(sum(u)+eps); mean input reduces the
            # explicit token-count factor, including epsilon/L.
            d = u.mean(-1, keepdim=True) + self.eps/s.shape[-1]
            u = u/s.shape[-1]
        elif self.kind in ('square_row', 'square_fixed'):
            u = (s + self.shift).square() + self.floor
            d = u.mean(-1, keepdim=True)
            u = u/s.shape[-1]
            if self.kind == 'square_fixed':
                if self.training:
                    self.running_denominator.copy_(d.detach().amax(0, keepdim=True))
                d = self.running_denominator
        elif self.kind == 'bpmax':
            u = (s + self.shift).pow(self.power)
            if self.training:
                self.running_denominator.copy_(u.sum(-1, keepdim=True).detach().amax(0, keepdim=True))
            d = self.running_denominator + 1e-10
        else:
            raise ValueError(self.kind)
        if self.observer:
            # Broadcast fixed batch denominators so observations retain image IDs.
            self.observer('denominator', d.expand(s.shape[0], -1, -1, -1))
        return u/d if self.inverse is None else u*self.inverse(d)


class BatchLN(nn.Module):
    """Powerformer operator port: token centering, unbiased variance, last-batch
    maximum std per token. Buffer update and epsilon match inspected source.
    This is NOT BatchNorm and remains token-centering at inference.
    """
    def __init__(self, dim, tokens, scale):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.zeros(dim))
        self.register_buffer('running_denominator', torch.ones(1, tokens, 1))
        self.scale = scale

    def forward(self, x):
        z = x-x.mean(-1, keepdim=True)
        if self.training:
            d = z.var(-1, keepdim=True, unbiased=True).sqrt().amax(0, keepdim=True)*self.scale
            self.running_denominator.copy_(d.detach())
        return z/(self.running_denominator+1e-10)*self.weight+self.bias


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)); self.eps = 1e-5

    def forward(self, x):
        return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+self.eps)*self.weight


def make_norm(kind, spec):
    if kind == 'ln': return nn.LayerNorm(spec.embed_dim)
    if kind == 'bn': return nn.BatchNorm1d(spec.embed_dim)
    if kind == 'rms': return RMSNorm(spec.embed_dim)
    if kind == 'batchln':
        return BatchLN(spec.embed_dim, (spec.img_size//spec.patch_size)**2+1, spec.norm_scale)
    raise ValueError(kind)


def apply_norm(norm, x):
    if isinstance(norm, nn.BatchNorm1d): return norm(x.transpose(1, 2)).transpose(1, 2)
    return norm(x)


class Block(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d, h = spec.embed_dim, spec.num_heads
        self.norm1, self.norm2 = make_norm(spec.norm, spec), make_norm(spec.norm, spec)
        self.qkv, self.proj = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, int(d*spec.mlp_ratio)), nn.Linear(int(d*spec.mlp_ratio), d)
        self.act = PolyGELU() if spec.activation == 'quadratic' else nn.GELU()
        self.attn_act = Attention(spec)
        self.heads, self.head_dim, self.drop_prob = h, d//h, spec.drop_path
        self.observer = None; self.features = None

    def tap(self, name, x):
        if self.observer: self.observer(name, x)
        return x

    def drop(self, x):
        if not self.training or not self.drop_prob: return x
        keep = 1-self.drop_prob
        return x*torch.empty((x.shape[0],1,1),device=x.device).bernoulli_(keep)/keep

    def forward(self, x):
        self.tap('residual_in', x)
        h = self.tap('norm1_out', apply_norm(self.norm1, self.tap('norm1_in', x)))
        b,n,d = h.shape
        q,k,v = self.qkv(h).reshape(b,n,3,self.heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
        s = self.tap('scores', (q@k.transpose(-2,-1))*(self.head_dim**-.5))
        a = self.tap('attention', self.attn_act(s))
        out = self.tap('attention_branch', self.proj((a@v).transpose(1,2).reshape(b,n,d)))
        x = x+self.drop(out)
        norm2 = self.tap('norm2_out', apply_norm(self.norm2, self.tap('norm2_in', x)))
        u = self.tap('gelu_in', self.fc1(norm2))
        act = self.tap('gelu_out', self.act(u))
        x = self.tap('residual_out', x+self.drop(self.tap('mlp_branch', self.fc2(act))))
        if self.features is not None:
            self.features.update(scores=s, norm1=h, norm2=norm2, hidden=x)
        return x


class ViT(nn.Module):
    def __init__(self, spec=ModelSpec()):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        assert spec.embed_dim % spec.num_heads == 0 and spec.img_size % spec.patch_size == 0
        self.patch_embed = nn.Conv2d(3,spec.embed_dim,spec.patch_size,spec.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1,1,spec.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,(spec.img_size//spec.patch_size)**2+1,spec.embed_dim))
        nn.init.trunc_normal_(self.cls_token,std=.02); nn.init.trunc_normal_(self.pos_embed,std=.02)
        self.blocks = nn.ModuleList()
        for i in range(spec.depth):
            s=copy.deepcopy(spec); s.drop_path *= i/max(spec.depth-1,1)
            self.blocks.append(Block(s))
        self.norm = make_norm(spec.norm if spec.final_norm=='same' else spec.final_norm,spec)
        self.head = nn.Linear(spec.embed_dim,spec.num_classes)
        self.collect_features = False; self.features = {}; self.observer = None

    def forward(self, x):
        z=self.patch_embed(x).flatten(2).transpose(1,2)
        z=torch.cat([self.cls_token.expand(x.shape[0],-1,-1),z],1)+self.pos_embed
        self.features = {'embedding':z} if self.collect_features else {}
        for i,blk in enumerate(self.blocks):
            blk.features = {} if self.collect_features else None
            z=blk(z)
            if self.collect_features: self.features[str(i)] = blk.features
        if self.observer: self.observer('final_norm_in', z)
        z=apply_norm(self.norm,z)
        if self.observer: self.observer('final_norm_out', z)
        return self.head(z[:,0])


def legacy_spec(mask, num_classes=10):
    assert mask in ['000','100','010','001','110','101','011','111']
    return ModelSpec(num_classes=num_classes, activation='quadratic' if mask[0]=='1' else 'gelu',
                     attention='raw' if mask[1]=='1' else 'softmax', norm='bn' if mask[2]=='1' else 'ln')


def load_checkpoint(path, legacy_mask=None):
    # Only tensor/basic metadata checkpoints are accepted. No unsafe fallback.
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if 'state_dict' not in ck: raise ValueError('Expected state_dict checkpoint container')
    if 'model_spec' in ck: spec=ModelSpec(**ck['model_spec'])
    elif legacy_mask is not None:
        spec=legacy_spec(legacy_mask, ck['state_dict']['head.weight'].shape[0])
    else: raise ValueError('Legacy checkpoint requires an explicit mask')
    model=ViT(spec);model.load_state_dict(ck['state_dict'],strict=True)
    return model,ck


def transfer(teacher, student):
    """Value-verified transfer; norm parameters transfer only within same family.
    Cross-family norms keep fresh affine/buffers, a disclosed adaptation choice.
    """
    source,target=teacher.state_dict(),student.state_dict()
    shared,kept=[],[]
    tm,sm=dict(teacher.named_modules()),dict(student.named_modules())
    for key in target:
        parent=key.rsplit('.',1)[0]
        compatible = type(tm.get(parent)) is type(sm.get(parent))
        if key in source and source[key].shape==target[key].shape and compatible:
            target[key]=source[key].clone(); shared.append(key)
        else: kept.append(key)
    student.load_state_dict(target,strict=True)
    for key in shared:
        assert torch.equal(student.state_dict()[key],source[key]),key
    if not shared: raise ValueError('No weights transferred')
    return {'copied_keys':shared,'fresh_keys':kept,
            'copied_numel':sum(source[k].numel() for k in shared),
            'target_numel':sum(v.numel() for v in target.values())}
