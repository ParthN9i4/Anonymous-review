"""
Which polynomial substitution hurts most?
==========================================

The primary experiment is a complete 2x2x2 factorial.  The first bit controls
GELU, the second attention, and the third normalization:

  000: unchanged student + KD
  100/010/001: one substitution
  110/101/011: two substitutions
  111: all three substitutions

The teacher is a separate CE-trained reference.  Alternatives H/I are
available as a separate ``--config-set all`` comparison.

  Model A: Replace ONLY GELU → PolyGELU       (keep real softmax + LayerNorm)
  Model B: Replace ONLY softmax → PolyAttn     (keep real GELU + LayerNorm)
  Model C: Replace ONLY LayerNorm → BatchNorm  (keep real GELU + softmax)
  Model D: Replace GELU + softmax              (keep real LayerNorm)
  Model E: All three replaced                  (same as previous experiment)

Use ``--skip-no-kd`` for the paper's primary run; the CE-only arms are an
optional diagnostic and are not part of the factorial.

Usage: python substitution_ablation.py
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[WARN] scipy not available; Welch's t-tests will be skipped.")


# ── Data ─────────────────────────────────────────────────────────────

DATASET_INFO = {
    "cifar10":  {"cls": torchvision.datasets.CIFAR10,  "num_classes": 10,
                 "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    # CIFAR-100 added 2026-09-09 specifically to compare against Zimerman et
    # al. (ICML 2024), whose own vision results (Sec 5.1, Table 2) are on
    # Tiny-ImageNet and CIFAR-100 -- NOT CIFAR-10 as an earlier version of
    # docs/PROJECT_RULES.md's Section 6 stated (verified against arXiv:2311.08610
    # fulltext 2026-09-09; that line needs correcting separately).
    # Mean/std are the standard CIFAR-100 per-channel statistics.
    "cifar100": {"cls": torchvision.datasets.CIFAR100, "num_classes": 100,
                 "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762)},
}


def get_dataset(name="cifar10", batch_size=128, val_size=5000, split_seed=1234,
                loader_seed=None, num_workers=2):
    """Return (train, val, test) loaders for `name` in {cifar10, cifar100}.

    PROTOCOL NOTE (changed 2026-09-09). This function previously returned
    (train, test) only, and `train_model` selected its reported checkpoint by
    evaluating on the TEST set every epoch -- test-set leakage. Confirmed by
    external review 2026-09-04; `docs/P1_AUDIT.md` F24 had the pattern but
    mis-scoped it to two other scripts.

    `leakage_sensitivity_check.py` quantified the damage from data already on
    disk: the bias tracks variance, so the collapsing arm (std ~20) was
    inflated +8.18 pts while every stable arm moved 0.18-0.32 pts. Selection
    was MASKING the collapse, not creating it -- but the numbers still cannot
    ship as clean held-out estimates.

    Now: a fixed 45k/5k train/val split carved from the TRAINING set, with a
    split seed independent of the model seed so every arm and every seed sees
    the identical split. The test set is touched once, at the end.

    The val split uses the TEST transform (no augmentation) -- selection should
    not be made on randomly-cropped images.
    """
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASET_INFO)}")
    info = DATASET_INFO[name]
    DsCls, mean, std = info["cls"], info["mean"], info["std"]

    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # Two views of the same underlying training set: augmented for fitting,
    # clean for selection. Indices are disjoint, so no augmented copy of a
    # validation image is ever trained on.
    train_aug = DsCls('./data', train=True, download=True, transform=t_train)
    train_cln = DsCls('./data', train=True, download=True, transform=t_test)
    test = DsCls('./data', train=False, download=True, transform=t_test)

    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_sub = torch.utils.data.Subset(train_aug, train_idx)
    val_sub = torch.utils.data.Subset(train_cln, val_idx)

    # A per-run generator makes the minibatch order reproducible and identical
    # across the eight arms for a given seed.  The split itself remains fixed
    # by split_seed and is independent of the model seed.
    gen = None if loader_seed is None else torch.Generator().manual_seed(int(loader_seed))
    return (DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, generator=gen),
            DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(test, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True))


# ── Polynomial replacements ──────────────────────────────────────────
#
# These are the same ones from simple_kd_baseline.py.
# Each one replaces a nonlinear operation that CKKS cannot compute
# with a polynomial (additions + multiplications only).

class PolyGELU(nn.Module):
    """
    Replaces GELU with ax² + bx + c.

    GELU is the activation function used in every FFN block of the
    transformer. It decides which neurons "fire" and which stay quiet.

    Why polynomial? Under CKKS encryption, you cannot compute
    GELU(x) = x * Φ(x) because the Gaussian CDF Φ involves exp
    and erf — impossible with just add/multiply.

    ax² + bx + c uses only multiply and add → works under CKKS.
    Cost: 1 CKKS multiplicative level.
    """
    def __init__(self):
        super().__init__()
        # Fit a degree-2 polynomial to GELU over [-3, 3]
        x = torch.linspace(-3, 3, 10000)
        y = F.gelu(x)
        X = torch.stack([x**2, x, torch.ones_like(x)], dim=1)
        c = torch.linalg.lstsq(X, y).solution
        self.a = nn.Parameter(c[0].clone())
        self.b = nn.Parameter(c[1].clone())
        self.c = nn.Parameter(c[2].clone())

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


class PolyAttn(nn.Module):
    """
    Replaces softmax in attention with ax² + bx + c applied element-wise.

    Softmax converts attention scores into probabilities:
      attn_weights = softmax(Q @ K^T / sqrt(d))

    Under CKKS, softmax is the HARDEST operation because it needs:
      - max subtraction (comparison → ~15 CKKS levels)
      - exponentiation (transcendental → impossible exactly)
      - division by sum (inverse → ~20 levels via Goldschmidt)
    Total: ~38 levels, more than the entire CKKS budget.

    Our replacement: just apply a polynomial to each attention score
    independently. No max, no exp, no division.
    Cost: 1 CKKS level.

    The polynomial acts as a "soft gate" — KD trains it so that the
    student's attention patterns produce similar outputs to the teacher.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        return self.a * x * x + self.b * x + self.c


class PowerNormAttn(nn.Module):
    """Even-power attention with exact row normalisation and a positive floor.

    OPERATOR_ID = powernorm_evenpow_floor_v1

    NAMING DISCIPLINE -- READ BEFORE CITING. This is **not** claimed to be a
    reproduction of PowerSoftmax (Zimerman et al.). The PowerSoftmax paper is
    not in this repo (`papers/` does not exist here), so its training-time
    scaling identity, its zero-denominator treatment and its length-dependent
    division-range handling could not be checked line by line. Implementing an
    operator from a second-hand prose description and then calling it by the
    published method's name is exactly the mistake the H/I arms already made
    once (see PolyAttnGlobalDenom). Call this what it is: an even-power,
    row-normalised gate with an explicit positive floor. If a faithful
    PowerSoftmax arm is wanted, it is a SEPARATE arm, added after reading the
    paper.

    The operator (the controlled intervention specified in the 2026-09-15
    research-reset review, Sec 5.2):

        z_ij = alpha * s_ij + beta          learnable affine, per block
        u_ij = z_ij^p + delta               p EVEN and delta > 0  ->  u_ij > 0
        A_ij = u_ij / sum_j u_ij            exact row normalisation

    Three properties that matter, in order of importance to this project:

    1. ROWS SUM TO EXACTLY ONE. This restores the per-token gain control that
       softmax provides implicitly and that the raw PolyAttn gate destroys.
       Under the gain-dispersion hypothesis this is the load-bearing property,
       and it is why this arm is predicted to survive being paired with
       BatchNorm -- the cell where the raw gate collapses.

    2. THE DENOMINATOR HAS A GUARANTEED POSITIVE LOWER BOUND, sum_j u_ij >= n*delta
       for n keys, BY CONSTRUCTION rather than by observation. That is what
       makes the CKKS reciprocal budget an analytic quantity instead of an
       empirical hope: the Goldschmidt iteration count needed for a target
       relative error follows from [n*delta, max] alone (see pmode_eval.py,
       goldschmidt_min_usable_d). A calibrated-range census can only report
       what it happened to see; this bounds what CAN be seen.

    3. NON-NEGATIVITY COSTS ZERO LEVELS. An even power is polynomial; a ReLU is
       a comparison, measured at ~13 levels in this project's own ledger. Gate 1
       already showed the two are statistically indistinguishable in accuracy
       WHEN row normalisation is present (94.92 +/- 0.28 vs 94.80 +/- 0.46), so
       this is a free depth saving, not a trade.

    `delta` is a fixed hyperparameter, deliberately NOT learnable: a learned
    floor could be driven toward zero by training, which would silently destroy
    property 2 -- and this repo has already shipped one inert learnable offset
    that never moved off its initial value (fix5_evenpower_ablation.py's c).
    Raising delta tightens the denominator bound but pushes attention toward
    uniformity when the powered scores are small; that cost is real and must be
    measured, not assumed away.
    """

    OPERATOR_ID = "powernorm_evenpow_floor_v1"

    def __init__(self, power: int = 2, delta: float = 1e-2):
        super().__init__()
        if power % 2 != 0 or power < 2:
            raise ValueError(f"power must be a positive EVEN integer, got {power}")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0 to bound the denominator, got {delta}")
        self.power = int(power)
        self.delta = float(delta)
        # Affine on the score, so the block can learn where to place the
        # even power's minimum. Initialised to the identity-ish (1, 0) rather
        # than to the legacy (0.05, 0.5, 0.25) quadratic: this is a different
        # operator and inheriting the old init would confound the comparison.
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        # Diagnostics for the gain-dispersion measurement. Buffers, so they
        # survive state_dict and can be read off a trained checkpoint.
        self.register_buffer("last_rowmass_min", torch.tensor(float("nan")))
        self.register_buffer("last_rowmass_max", torch.tensor(float("nan")))

    def forward(self, scores):
        z = self.alpha * scores + self.beta
        u = z.pow(self.power) + self.delta
        denom = u.sum(dim=-1, keepdim=True)
        with torch.no_grad():
            d = denom.detach()
            self.last_rowmass_min.fill_(float(d.min()))
            self.last_rowmass_max.fill_(float(d.max()))
        return u / denom


class PolyAttnGlobalDenom(nn.Module):
    """Polynomial attention with a single global fixed denominator.

    NOT A BPMax REPRODUCTION. Renamed from `PolyAttnBatchMax` on 2026-09-14
    after checking the operator against the Powerformer paper (Park, Lee & Lee,
    ACL 2025, Sec 3.1 p.11092), which defines

        BPMax(x) = (x+c)^p / max_i sum_l (x_{i,j,k,l} + c)^p  ->  (x+c)^p / R_d

    Indices j (head) and k (query) are FREE in that expression, so BPMax's
    denominator is one value per head per query row. Two deviations:

    | | Powerformer BPMax | this module |
    |---|---|---|
    | numerator | (x + c)^p, one free parameter | a*s^2 + b*s + c, three free |
    | denominator reduction | max over batch i only; head j and query k kept | max over batch, head AND query -> ONE scalar |
    | init in the (x+c)^2 family? | yes by construction | no: b^2 = 0.25 != 4ac = 0.05 |

    Either deviation alone would disqualify the "reproduction" label, so this
    arm tests a global-scalar denominator -- a strictly cheaper operator -- and
    nothing about Powerformer's actual method. A faithful arm is a NEW arm
    (reduce with `poly.sum(-1).amax(dim=0)`, keeping head and query), not a
    relabel of this one.

    Added 2026-09-10 to make the Powerformer comparison honest.

    Config F (softmax+norm, using plain PolyAttn) collapses on CIFAR-10 --
    32.95% vs a 75.40% teacher, seed 42 -- while our PolyAttn has NO
    denominator at all. This arm adds one back, to separate "the denominator
    matters" from "the regime matters".

    It keeps Powerformer's train/inference SPLIT, which is the part worth
    borrowing: the denominator is computed per batch during training and
    tracked into a running buffer that becomes a fixed constant at eval -- the
    same split BatchNorm uses, and verbatim from Sec 3.1, "During training, the
    denominator is computed per batch; during inference, it is replaced with a
    fixed constant R_d computed in advance." At inference it is a plaintext
    scalar, so it costs ZERO CKKS levels, unlike Fix 2's row division (measured
    at 13 levels, experiment_results/encrypted_primitives.json).

    Rows do NOT sum to one, so this arm sits outside Proposition 1 by design.
    Gate 1 showed row normalisation is the load-bearing component (94.80 with
    vs 63.65 +/- 39.88 without), so if THIS arm trains while F collapses, the
    explanation for F is the missing denominator. If it collapses too, the
    explanation is the regime (random init + single soft-label KD vs fine-tuned
    init + 4-point distillation), which is the stronger finding.

    Regime context, verified from the PDF (Table 5, p.11097, average over
    RTE/MRPC/SST-2): Baseline 82.86, BPMax 83.01, Batch LN 83.81, Both 83.08.
    Powerformer's own combination is WORSE than Batch LN alone, and worse on
    all three tasks individually (71.48->70.52, 87.91->86.76, 92.05->91.97).
    So a sub-additive attention x normalisation interaction is present in their
    numbers too -- mild under a fine-tuned BERT recipe, catastrophic under ours.
    Cite that as the precedent for the regime axis; do not claim their
    substitutions were simply benign.
    """

    def __init__(self, eps: float = 1e-6, momentum: float = 0.1):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))
        self.eps = eps
        self.momentum = momentum
        # Buffer, not parameter: tracked like a BatchNorm running stat, and it
        # must survive state_dict save/load for the inference constant to be
        # the one training actually produced.
        self.register_buffer("running_denom", torch.tensor(1.0))
        # Diagnostic, not a safety device. Counts training batches whose global
        # max row sum fell below eps -- i.e. every row in the batch was
        # non-positive. Before 2026-09-14 that case silently divided by a
        # negative number; the clamp now prevents it, but whether it EVER fired
        # is the question that decides if completed H/I runs are usable, and
        # without a counter that question is unanswerable after the fact.
        self.register_buffer("neg_denom_batches", torch.tensor(0, dtype=torch.long))

    def forward(self, scores):
        poly = self.a * scores * scores + self.b * scores + self.c
        if self.training:
            # Max over batch, head AND query of the per-row sums -- see the
            # deviation table above; this is a global scalar, not BPMax's R_d.
            raw = poly.sum(dim=-1).max().detach()
            # Clamp the divisor, not just the buffer update. Fixed 2026-09-14.
            # `poly` is a free quadratic and CAN go negative, so `raw` is the
            # max of quantities that may all be negative. The previous code
            # tracked `raw.clamp(min=eps)` into the buffer but divided by
            # `raw + eps` directly, so a batch whose every row sum was negative
            # divided by a negative number -- flipping the sign of the entire
            # attention map -- and a raw near -eps divided by ~0. Eval was
            # always safe (the buffer is clamped); only training could fire.
            denom = raw.clamp(min=self.eps)
            with torch.no_grad():
                if raw < self.eps:
                    self.neg_denom_batches += 1
                self.running_denom.mul_(1.0 - self.momentum).add_(
                    self.momentum * denom)
        else:
            denom = self.running_denom.clamp(min=self.eps)
        return poly / (denom + self.eps)


# ── Configurable DeiT-Tiny ──────────────────────────────────────────
#
# This single model class can use either standard or polynomial
# operations. We control which ones to replace via boolean flags.

class ConfigurableDeiT(nn.Module):
    """
    DeiT-Tiny where you choose which operations to replace.

    Args:
        poly_gelu:    If True, use PolyGELU instead of nn.GELU
        poly_softmax: If True, use PolyAttn instead of softmax
        poly_norm:    If True, use BatchNorm1d instead of LayerNorm
    """
    def __init__(self, num_classes=10, img_size=32, patch_size=4,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 poly_gelu=False, poly_softmax=False, poly_norm=False,
                 drop_path_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.poly_softmax = poly_softmax
        self.poly_norm = poly_norm

        # Patch embedding — same for all variants
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Build transformer blocks.
        # drop_path_rate > 0 enables stochastic depth, linearly scaled across
        # depth as in DeiT/timm. Default 0.0 keeps this class bit-identical to
        # its pre-2026-09-10 behaviour, so the substitution-factorial results
        # remain comparable; only strong_baseline.py opts in.
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(ConfigurableBlock(
                embed_dim, num_heads, mlp_ratio,
                poly_gelu=poly_gelu,
                poly_softmax=poly_softmax,
                poly_norm=poly_norm,
                drop_path=dpr[i],
            ))

        # Final norm + head
        if poly_norm:
            self.norm = nn.BatchNorm1d(embed_dim)
        else:
            self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        # Final norm
        if isinstance(self.norm, nn.BatchNorm1d):
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm(x)

        return self.head(x[:, 0])


class ConfigurableBlock(nn.Module):
    """One transformer block with configurable operations."""

    def __init__(self, dim, num_heads, mlp_ratio,
                 poly_gelu=False, poly_softmax=False, poly_norm=False,
                 drop_path=0.0):
        super().__init__()

        # Stochastic depth. nn.Identity() when drop_path == 0.0, so the
        # default path is numerically identical to before this was added.
        if drop_path > 0.0:
            from timm.layers import DropPath
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        # Norm layers: LayerNorm (standard) or BatchNorm1d (polynomial)
        if poly_norm:
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        self.use_bn = poly_norm

        # Attention
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # poly_softmax: False | True ("plain" PolyAttn) | "globaldenom" (a single
        # fixed denominator). String form added 2026-09-10; bool form unchanged,
        # so every pre-existing config is bit-identical.
        self.poly_softmax = poly_softmax
        if poly_softmax == "globaldenom":
            self.attn_act = PolyAttnGlobalDenom()
        elif poly_softmax == "powernorm":
            self.attn_act = PowerNormAttn()
        elif poly_softmax:
            self.attn_act = PolyAttn()

        # FFN
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        if poly_gelu:
            self.act = PolyGELU()
        else:
            self.act = nn.GELU()

    def _norm(self, norm_layer, x):
        """Apply norm — handles the transpose needed for BatchNorm1d."""
        if self.use_bn:
            return norm_layer(x.transpose(1, 2)).transpose(1, 2)
        else:
            return norm_layer(x)

    def forward(self, x):
        # Attention
        h = self._norm(self.norm1, x)
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.poly_softmax:
            attn = self.attn_act(attn)
        else:
            attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.drop_path(self.proj(out))

        # FFN
        h = self._norm(self.norm2, x)
        x = x + self.drop_path(self.fc2(self.act(self.fc1(h))))

        return x


# ── Training ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def _set_determinism(enabled: bool):
    """Make training reproducible enough that a re-trained teacher is the SAME
    teacher its students distilled from.

    This file previously set no determinism flags at all, while the BloodMNIST
    path did (fix2_bloodmnist.py:102-103). That asymmetry is why a re-trained
    CIFAR teacher could not be used as a matched control for historical
    students: the original teacher was never checkpointed AND could not be
    reconstructed. Both halves are fixed -- this flag, plus the teacher
    checkpoint written in main().

    Costs throughput. Disable only for exploratory runs, never for a cohort
    whose arms must be comparable to each other.
    """
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cudnn.benchmark = not bool(enabled)


def _assert_unsubstituted(model, label):
    """The 000 cell must be a genuinely unchanged student.

    Checking the config tuple is not enough -- it only proves what we asked
    for, not what got built. A silently substituted 000 arm would bias every
    contrast that uses it, which is most of the cube.
    """
    bad = [type(m).__name__ for m in model.modules()
           if type(m).__name__ in ("PolyGELU", "PolyAttn", "PolyAttnGlobalDenom",
                                    "PolyAttnNormed", "TokenBatchNorm")]
    if bad:
        raise SystemExit(f"[FATAL] {label} is the 000 cell but contains "
                         f"polynomial/BN modules: {sorted(set(bad))}")
    if not any(isinstance(m, nn.LayerNorm) for m in model.modules()):
        raise SystemExit(f"[FATAL] {label} is the 000 cell but has no LayerNorm")
    if any(isinstance(m, nn.BatchNorm1d) for m in model.modules()):
        raise SystemExit(f"[FATAL] {label} is the 000 cell but has BatchNorm1d")


def train_model(model, train_loader, val_loader, test_loader, device, epochs=100,
                teacher=None, kd_alpha=0.1, kd_temp=4.0, label="",
                checkpoint_path=None, checkpoint_meta=None):
    """Train, select on VAL, evaluate on TEST exactly once.

    Returns a dict, not a scalar. The reported headline number is
    `test_at_best_val`: the test accuracy of the checkpoint that had the best
    VALIDATION accuracy. `test_final` is reported alongside as a
    selection-free sensitivity check (see leakage_sensitivity_check.py).
    """
    import copy
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()

    best_val, best_state, best_epoch = -1.0, None, -1
    # Per-epoch trajectory. Required by docs/PREREGISTRATION.md to separate
    # numerical failure / failure to learn / late deterioration. Final
    # accuracies alone cannot distinguish "never escaped chance" from "trained,
    # then collapsed", and the distinction cannot be recovered after the run.
    val_curve, loss_curve = [], []
    print(f"\n  Training: {label}")
    print(f"  KD: {'yes' if teacher else 'no'}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            ce_loss = F.cross_entropy(logits, labels)

            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=-1),
                    F.softmax(t_logits / kd_temp, dim=-1),
                    reduction='batchmean')
                loss = kd_alpha * kd_loss + (1 - kd_alpha) * ce_loss
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        # SELECTION HAPPENS ON VAL ONLY. The test set is not touched in this loop.
        val_acc = evaluate(model, val_loader, device)
        mean_loss = total_loss / max(len(train_loader), 1)
        val_curve.append(round(float(val_acc), 4))
        loss_curve.append(None if not math.isfinite(mean_loss)
                          else round(float(mean_loss), 6))
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            # Keep the saved selection on CPU.  This avoids retaining a full
            # GPU copy for every completed arm and makes checkpoints portable.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_loader):.4f}"
                  f"  val={val_acc:.2f}%  best_val={best_val:.2f}%")

    # Test set touched exactly twice, after training: once for the
    # val-selected checkpoint, once for the final one.
    test_final = evaluate(model, test_loader, device)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_at_best_val = evaluate(model, test_loader, device)

    checkpoint_written = None
    if checkpoint_path and best_state is not None:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        checkpoint = {
            "state_dict": best_state,
            "result": {
                "test_at_best_val": float(test_at_best_val),
                "test_final": float(test_final),
                "best_val": float(best_val),
                "best_epoch": int(best_epoch + 1),
            },
            "label": label,
            "checkpoint_rule": "best validation accuracy",
            "git_commit": _git_commit(),
            "metadata": checkpoint_meta or {},
        }
        tmp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)
        checkpoint_written = checkpoint_path

    print(f"  > test @ best-val (epoch {best_epoch+1}): {test_at_best_val:.2f}%"
          f"   |  test @ final epoch: {test_final:.2f}%   |  best val: {best_val:.2f}%")
    result = {
        "test_at_best_val": test_at_best_val,
        "test_final": test_final,
        "best_val": best_val,
        "best_epoch": best_epoch + 1,
    }
    if checkpoint_written:
        result["checkpoint_path"] = checkpoint_written
    # Curves travel with the result and are split into a sidecar file by
    # main(), so the headline JSON stays small enough to read by eye.
    result["_curves"] = {"val_acc": val_curve, "train_loss": loss_curve}
    return result


# ── Main ─────────────────────────────────────────────────────────────


PRIMARY_CONFIGS = [
    # (name,                 poly_gelu, poly_softmax, poly_norm)
    ("0: unchanged KD",         False,     False,        False),
    ("A: GELU only",             True,      False,        False),
    ("B: softmax only",          False,     True,         False),
    ("C: norm only",             False,     False,        True),
    ("D: GELU+softmax",          True,      True,         False),
    ("E: all three",             True,      True,         True),
    ("F: softmax+norm",          False,     True,         True),
    ("G: GELU+norm",             True,      False,        True),
]

# The mechanism test. PowerNormAttn crossed with the normaliser: the
# gain-dispersion hypothesis says row normalisation is the load-bearing
# property, so PN+BatchNorm ("011-equivalent") should SURVIVE where the raw
# gate's 011 collapses. PN+LayerNorm is the control that says whether the
# operator trains at all. If PN+BatchNorm collapses too, the hypothesis is
# falsified in the first experiment, which is the point of running it first.
POWERNORM_CONFIGS = [
    ("P: powernorm+LN",   False, "powernorm", False),
    ("Q: powernorm+BN",   False, "powernorm", True),
]

# Secondary global-denominator alternatives.  Not points in the standard
# 2x2x2 factorial because their attention operator is different.
ALTERNATIVE_CONFIGS = [
    ("H: globaldenom+norm",  False, "globaldenom", True),
    ("I: globaldenom all three", True, "globaldenom", True),
]

# Backwards-compatible public name used by smoke tests and older scripts.
CONFIGS = PRIMARY_CONFIGS
CONFIG_MASKS = {
    "0: unchanged KD": "000", "A: GELU only": "100",
    "B: softmax only": "010", "C: norm only": "001",
    "D: GELU+softmax": "110", "E: all three": "111",
    "F: softmax+norm": "011", "G: GELU+norm": "101",
}


def _summarize(per_seed: dict, metric: str = "test_at_best_val") -> dict:
    """mean/std/min/max over seeds for one {seed: result} mapping.

    `train_model` returns a dict per seed since 2026-09-09 (val-based
    selection). `metric` picks which field aggregates:
      test_at_best_val -- headline; test acc of the best-VAL checkpoint
      test_final       -- selection-free sensitivity check
      best_val         -- the selection signal itself

    Both test metrics are always summarised so the sensitivity check ships
    with every result rather than needing a separate run.
    """
    keys = sorted(per_seed)
    accs = np.array([per_seed[s][metric] for s in keys], dtype=float)
    out = {
        "metric": metric,
        "mean": float(accs.mean()),
        "std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "min": float(accs.min()),
        "max": float(accs.max()),
        "per_seed": {str(s): float(per_seed[s][metric]) for s in keys},
        "n_seeds": int(len(accs)),
    }
    if metric == "test_at_best_val":
        fin = np.array([per_seed[s]["test_final"] for s in keys], dtype=float)
        out["sensitivity_test_final"] = {
            "mean": float(fin.mean()),
            "std": float(fin.std(ddof=1)) if len(fin) > 1 else 0.0,
            "per_seed": {str(s): float(per_seed[s]["test_final"]) for s in keys},
            "note": "no-selection comparator; see leakage_sensitivity_check.py",
        }
        out["best_val_per_seed"] = {str(s): float(per_seed[s]["best_val"]) for s in keys}
        out["best_epoch_per_seed"] = {str(s): int(per_seed[s]["best_epoch"]) for s in keys}
    return out



def _git_commit():
    """Short HEAD of the checkout that is producing these results.

    Stamped into every artifact. Two scripts' Config E numbers were merged into
    one table once already and nothing in either file recorded which code made
    it; a queued array task reads this .py from disk at task-start, so a pull
    mid-array can also split one experiment across two versions silently.
    Never let a failure here kill a run -- provenance is worth recording, not
    worth crashing over.
    """
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        if out.returncode != 0:
            return "UNKNOWN (git rev-parse failed)"
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5,
                               cwd=os.path.dirname(os.path.abspath(__file__)))
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit
    except Exception:
        return "UNKNOWN (git unavailable)"


def _pop_curves(teacher_per_seed, kd_per_seed, nokd_per_seed):
    """Move every `_curves` entry out of the result dicts into one nested map.

    DESTRUCTIVE by design: the trajectories must end up in exactly one file.
    Leaving them in both would put ~100 floats per arm back into the headline
    JSON, which is the thing this split exists to prevent. Call it once, after
    the last `_save_partial` -- partial files deliberately keep the curves
    inline so a job killed at the --time wall stays salvageable.
    """
    curves = {"teacher": {}, "with_kd": {}, "no_kd": {}}
    for seed, res in teacher_per_seed.items():
        if isinstance(res, dict) and "_curves" in res:
            curves["teacher"][str(seed)] = res.pop("_curves")
    for bucket, store in (("with_kd", kd_per_seed), ("no_kd", nokd_per_seed)):
        for nm, bysd in store.items():
            for seed, res in bysd.items():
                if isinstance(res, dict) and "_curves" in res:
                    curves[bucket].setdefault(nm, {})[str(seed)] = res.pop("_curves")
    return curves


def _file_prefix(args):
    tag = f"_{args.run_tag}" if args.run_tag else ""
    return f"substitution_ablation_{args.dataset}{tag}_seeds{'_'.join(str(s) for s in args.seeds)}"


def _partial_path(args):
    return os.path.join(args.results_dir, _file_prefix(args) + ".PARTIAL.json")


def _save_partial(args, teacher_per_seed, kd_per_seed, nokd_per_seed):
    """Checkpoint results after every completed run.

    Added 2026-09-10 after a near-miss: this script writes its final JSON only
    after ALL runs finish (5 seeds x 8 models = 40 runs at 100 epochs). Under
    SLURM, hitting the --time wall kills the job and every completed run is
    lost with it. A partial file costs milliseconds and makes a timed-out job
    salvageable instead of worthless. It is overwritten each time and
    superseded by the final JSON, so it is safe to delete once the real
    results file exists.
    """
    try:
        os.makedirs(args.results_dir, exist_ok=True)
        done = (len(teacher_per_seed)
                + sum(len(d) for d in kd_per_seed.values())
                + sum(len(d) for d in nokd_per_seed.values()))
        payload = {
            "STATUS": "PARTIAL -- job still running or was killed before finishing",
            "git_commit": _git_commit(),
            "runs_completed": done,
            "args": vars(args),
            "dataset": args.dataset,
            "teacher_per_seed": {str(k): v for k, v in teacher_per_seed.items()},
            "with_kd_per_seed": {nm: {str(k): v for k, v in d.items()}
                                 for nm, d in kd_per_seed.items()},
            "no_kd_per_seed": {nm: {str(k): v for k, v in d.items()}
                               for nm, d in nokd_per_seed.items()},
        }
        tmp = _partial_path(args) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=float)
        os.replace(tmp, _partial_path(args))   # atomic; never a half-written file
    except Exception as e:                      # never let checkpointing kill a run
        print(f"  [WARN] partial-save failed (continuing): {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Single/pair/triple substitution ablation on CIFAR-10/CIFAR-100.")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=list(DATASET_INFO.keys()),
                        help="cifar10 (original) or cifar100 (added 2026-09-09 "
                             "to compare against Zimerman et al., ICML 2024, "
                             "whose vision results are on CIFAR-100/Tiny-ImageNet).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds to run. Default [42] reproduces the original "
                             "single-seed run; pass 42 43 44 45 46 for the "
                             "multi-seed version the paper needs.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers; keep <= --cpus-per-task on SLURM.")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20,
                        help="Fraction of the visible MIG slice PyTorch may allocate.")
    parser.add_argument("--results-dir", type=str, default="./experiment_results")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints_substitution")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Non-empty tag added to result/checkpoint filenames; "
                             "use e.g. clean8 to avoid overwriting historical runs.")
    parser.add_argument("--config-set",
                        choices=["primary", "all", "000", "powernorm"],
                        default="primary",
                        help="primary = the clean eight-arm factorial; all also "
                             "runs the two global-denominator alternatives; 000 runs "
                             "only the unchanged KD control; powernorm runs the "
                             "even-power row-normalised mechanism test (P/Q); "
                             "unchanged KD control.")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="cudnn deterministic kernels (default on). Required "
                             "for a cohort whose arms must be comparable and for "
                             "a re-trained teacher to match its students.")
    parser.add_argument("--no-deterministic", dest="deterministic",
                        action="store_false",
                        help="faster, but arms are no longer bit-comparable.")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save best-validation checkpoints for every arm.")
    parser.add_argument("--no-kd-arm", action="store_true", default=True,
                        help="Run the CE-only (no-KD) arm as well. On by default.")
    parser.add_argument("--skip-no-kd", dest="no_kd_arm", action="store_false",
                        help="Skip the CE-only arm to halve runtime.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        if not 0.0 < args.gpu_memory_fraction <= 1.0:
            raise ValueError("--gpu-memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    num_classes = DATASET_INFO[args.dataset]["num_classes"]
    print(f"Device        : {device}")
    print(f"Dataset       : {args.dataset}  ({num_classes} classes)")
    print(f"Seeds         : {args.seeds}")
    print(f"Epochs        : {args.epochs}   Batch: {args.batch_size}")
    print(f"Workers       : {args.num_workers}   GPU fraction: {args.gpu_memory_fraction}")
    configs = list(PRIMARY_CONFIGS)
    if args.config_set == "all":
        configs += list(ALTERNATIVE_CONFIGS)
    elif args.config_set == "000":
        configs = [PRIMARY_CONFIGS[0]]
    elif args.config_set == "powernorm":
        # Paired against the existing 010/011 cells from the clean8 run: same
        # seeds, same split, same KD recipe, same checkpoint rule, so the only
        # difference is the attention operator.
        configs = list(POWERNORM_CONFIGS)
    _set_determinism(args.deterministic)
    print(f"Config set    : {args.config_set} ({len(configs)} student arms)")
    print(f"Determinism   : cudnn.deterministic={args.deterministic}")
    print(f"CE-only arm   : {'yes' if args.no_kd_arm else 'SKIPPED'}")
    t_start = time.time()

    teacher_per_seed = {}
    kd_per_seed = {name: {} for name, _, _, _ in configs}
    nokd_per_seed = {name: {} for name, _, _, _ in configs}

    for seed in args.seeds:
        print(f"\n{'#'*64}\n# SEED {seed}\n{'#'*64}")

        # Build fresh loaders for every arm through this helper.  Reusing one
        # shuffled DataLoader would advance its generator during the teacher
        # run, giving later arms a different minibatch order.  Fresh loaders
        # with the same seed make the comparison reproducible and matched.
        def loaders_for_seed():
            return get_dataset(args.dataset, batch_size=args.batch_size,
                               loader_seed=seed, num_workers=args.num_workers)

        # ── Step 1: teacher for THIS seed ────────────────────────────
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        teacher = ConfigurableDeiT(
            num_classes=num_classes, poly_gelu=False, poly_softmax=False, poly_norm=False)
        train_loader, val_loader, test_loader = loaders_for_seed()
        teacher_res = train_model(
            teacher, train_loader, val_loader, test_loader, device, args.epochs,
            label=f"Teacher (seed {seed}): standard GELU + softmax + LayerNorm",
            checkpoint_path=(os.path.join(args.ckpt_dir, _file_prefix(args) +
                                           f"_teacher_seed{seed}.pth")
                             if args.save_checkpoints else None),
            checkpoint_meta={"dataset": args.dataset, "seed": seed,
                             "arm": "teacher", "architecture": "ConfigurableDeiT"})
        teacher_per_seed[seed] = teacher_res
        _save_partial(args, teacher_per_seed, kd_per_seed, nokd_per_seed)

        # ── Step 2: each substitution, WITH KD ───────────────────────
        for name, pg, ps, pn in configs:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            student = ConfigurableDeiT(
                num_classes=num_classes, poly_gelu=pg, poly_softmax=ps, poly_norm=pn)
            if CONFIG_MASKS.get(name) == "000":
                _assert_unsubstituted(student, f"{name} seed {seed}")
            replaced = []
            if pg: replaced.append("GELU->PolyGELU")
            if ps: replaced.append("softmax->PolyAttn")
            if pn: replaced.append("LayerNorm->BatchNorm")
            train_loader, val_loader, test_loader = loaders_for_seed()
            res = train_model(
                student, train_loader, val_loader, test_loader, device, args.epochs,
                teacher=teacher,
                label=f"{name} seed {seed} [{', '.join(replaced)}]",
                checkpoint_path=(os.path.join(args.ckpt_dir, _file_prefix(args) +
                                               f"_{name.split(':', 1)[0]}_seed{seed}.pth")
                                 if args.save_checkpoints else None),
                checkpoint_meta={"dataset": args.dataset, "seed": seed,
                                 "arm": name, "mask": CONFIG_MASKS.get(name),
                                 "architecture": "ConfigurableDeiT"})
            kd_per_seed[name][seed] = res
            _save_partial(args, teacher_per_seed, kd_per_seed, nokd_per_seed)

        # ── Step 3: each substitution, WITHOUT KD ────────────────────
        if args.no_kd_arm:
            for name, pg, ps, pn in configs:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                student = ConfigurableDeiT(
                    num_classes=num_classes, poly_gelu=pg, poly_softmax=ps, poly_norm=pn)
                if CONFIG_MASKS.get(name) == "000":
                    _assert_unsubstituted(student, f"{name} seed {seed} (no KD)")
                train_loader, val_loader, test_loader = loaders_for_seed()
                res = train_model(
                    student, train_loader, val_loader, test_loader, device, args.epochs,
                    label=f"{name} seed {seed} (no KD)",
                    checkpoint_path=(os.path.join(args.ckpt_dir, _file_prefix(args) +
                                                   f"_{name.split(':', 1)[0]}_seed{seed}_nokd.pth")
                                     if args.save_checkpoints else None),
                    checkpoint_meta={"dataset": args.dataset, "seed": seed,
                                     "arm": name, "mask": CONFIG_MASKS.get(name),
                                     "kd": False, "architecture": "ConfigurableDeiT"})
                nokd_per_seed[name][seed] = res
                _save_partial(args, teacher_per_seed, kd_per_seed, nokd_per_seed)

    # ── Aggregate ────────────────────────────────────────────────────
    summary = {
        "Teacher": _summarize(teacher_per_seed),
        "with_kd": {name: _summarize(kd_per_seed[name]) for name, _, _, _ in configs},
    }
    if args.no_kd_arm:
        summary["no_kd"] = {name: _summarize(nokd_per_seed[name])
                            for name, _, _, _ in configs}

    # Welch's t-test of each KD config against the teacher (needs >1 seed).
    # scipy's t-distribution, NOT the normal approximation -- see the
    # verify_fixes.py bug fixed 2026-08-17, which understated p by 3-8x at n=5.
    if HAVE_SCIPY and len(args.seeds) > 1:
        # Tested on the headline metric (val-selected test accuracy).
        M = "test_at_best_val"
        t_accs = np.array([teacher_per_seed[s][M] for s in sorted(teacher_per_seed)])
        summary["tests_vs_teacher"] = {}
        for name, _, _, _ in configs:
            c_accs = np.array([kd_per_seed[name][s][M]
                               for s in sorted(kd_per_seed[name])])
            r = scipy_stats.ttest_ind(c_accs, t_accs, equal_var=False)
            summary["tests_vs_teacher"][name] = {
                "t": float(r.statistic), "p": float(r.pvalue),
                "diff_pp": float(c_accs.mean() - t_accs.mean()),
                "metric": M,
            }

    # ── Print table ──────────────────────────────────────────────────
    n = len(args.seeds)
    print(f"\n{'='*78}")
    print(f"RESULTS  ({args.dataset.upper()}, {n} seed{'s' if n > 1 else ''}: {args.seeds})")
    print(f"{'='*78}")
    tm, ts = summary["Teacher"]["mean"], summary["Teacher"]["std"]
    print(f"  Teacher (standard): {tm:.2f}% +/- {ts:.2f}%")
    print()
    hdr = f"  {'Configuration':<22} {'With KD':>16}"
    if args.no_kd_arm:
        hdr += f" {'No KD':>16}"
    hdr += f" {'Drop vs teacher':>16}"
    print(hdr)
    print(f"  {'-'*22} {'-'*16}" + (f" {'-'*16}" if args.no_kd_arm else "") + f" {'-'*16}")
    for name, _, _, _ in configs:
        k = summary["with_kd"][name]
        row = f"  {name:<22} {k['mean']:>8.2f}+/-{k['std']:<6.2f}"
        if args.no_kd_arm:
            nk = summary["no_kd"][name]
            row += f" {nk['mean']:>8.2f}+/-{nk['std']:<6.2f}"
        row += f" {tm - k['mean']:>15.2f}%"
        print(row)

    if summary.get("tests_vs_teacher"):
        print("\n  Welch's t vs teacher (two-sided, scipy t-distribution):")
        for name, t in summary["tests_vs_teacher"].items():
            star = " *" if t["p"] < 0.05 else ""
            print(f"    {name:<22} dAcc={t['diff_pp']:+7.2f}pp  p={t['p']:.4f}{star}")

    if n == 1:
        print("\n  [WARN] Single seed. The project's headline finding is that seed")
        print("         variance is enormous (Config E: 11-63% across seeds on")
        print("         CIFAR-10). A single-seed 'substitution X is safe' claim is")
        print("         an evidential double standard. Re-run with")
        print("         --seeds 42 43 44 45 46 before citing this in the paper.")

    # ── Save: JSON to experiment_results/ (NOT a .pth -- *.pth is gitignored,
    #    which is exactly why this script's results were absent from the repo
    #    and the README's Config A-E table had no artifact behind it) ────
    os.makedirs(args.results_dir, exist_ok=True)
    seed_tag = "_".join(str(s) for s in args.seeds)
    out_path = os.path.join(
        args.results_dir,
        _file_prefix(args) + ".json")

    # ── Split per-epoch trajectories into a sidecar ──────────────────
    # Mirrors verify_fixes.py's verification_curves_*.json. Keeping ~100 floats
    # per arm out of the headline file leaves it readable, while preserving the
    # evidence docs/PREREGISTRATION.md needs to tell "never escaped chance"
    # apart from "trained, then collapsed".
    curves = _pop_curves(teacher_per_seed, kd_per_seed, nokd_per_seed)
    curves_path = os.path.join(
        args.results_dir, _file_prefix(args) + "_curves.json")
    tmp_c = curves_path + ".tmp"
    with open(tmp_c, "w") as f:
        json.dump({"git_commit": _git_commit(), "dataset": args.dataset,
                   "seeds": args.seeds, "epochs": args.epochs,
                   "metric": "validation accuracy (%) per epoch",
                   "config_masks": {nm: CONFIG_MASKS.get(nm)
                                    for nm, _, _, _ in configs},
                   "curves": curves}, f, indent=2)
    os.replace(tmp_c, curves_path)

    payload = {
        "git_commit": _git_commit(),
        "args": vars(args),
        "dataset": DATASET_INFO[args.dataset]["cls"].__name__,
        "num_classes": num_classes,
        "architecture": "ConfigurableDeiT (see substitution_ablation.py:130)",
        "kd_alpha": 0.1,
        "kd_temp": 4.0,
        "kd_t2_scaling": False,
        "optimizer": "AdamW lr=1e-3 wd=0.05, CosineAnnealingLR",
        "checkpoint_rule": "best VALIDATION accuracy observed during training (see get_dataset docstring)",
        "config_set": args.config_set,
        "config_masks": {name: CONFIG_MASKS.get(name) for name, _, _, _ in configs},
        # NOTE (fixed 2026-09-09): train_model returns a per-seed dict, not a
        # scalar, since the val-based-selection fix. `float(v)` on that dict
        # raised TypeError unconditionally -- caught before this was ever run;
        # store the dict as-is (json.dump serializes it fine).
        "teacher_per_seed": {str(k): v for k, v in teacher_per_seed.items()},
        "with_kd_per_seed": {nm: {str(k): v for k, v in d.items()}
                             for nm, d in kd_per_seed.items()},
        "no_kd_per_seed": ({nm: {str(k): v for k, v in d.items()}
                            for nm, d in nokd_per_seed.items()}
                           if args.no_kd_arm else None),
        "summary": summary,
        "wall_clock_seconds": time.time() - t_start,
    }
    tmp_out = out_path + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    os.replace(tmp_out, out_path)
    print(f"\nResults JSON: {out_path}")
    print(f"Curves JSON : {curves_path}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Checkpoints    : {'saved under ' + args.ckpt_dir if args.save_checkpoints else 'disabled'}")


if __name__ == '__main__':
    main()
