"""Model class definitions copied verbatim from native_source.py; training imports omitted."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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