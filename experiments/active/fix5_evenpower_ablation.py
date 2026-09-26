"""
fix5_evenpower_ablation.py
==========================
GATE 1 for the SaTML 2027 submission: is Fix 2 the *cheapest* route to a
trainable polynomial attention, or merely the first one we tried?

THE QUESTION
------------
Fix 2 is `attn = ReLU(poly(s)) / (rowsum + eps)`. It buys two properties:

    non-negativity   <- ReLU        (a COMPARISON; ~13-14 CKKS levels via the
                                     minimax sign polynomial, Lee et al.
                                     IEEE TDSC 2022, degrees {15,15,27}, alpha=13)
    sum-to-one       <- row division (Goldschmidt; ~2 levels/iter x 3-5 iters)

Both are expensive, and neither has been shown to be *necessary*. An even
power is non-negative by construction at zero extra cost, and a fixed
denominator is a plaintext scalar that folds away entirely. So this script
runs the 2x2 that separates the two costs:

                        | denominator = row-sum   | denominator = fixed 1/T
    --------------------+-------------------------+-------------------------
    non-neg via ReLU    | Fix2_Normalized (known) | Fix2_FixedDenom
    non-neg via square  | EvenPow_RowSum          | EvenPow_FixedDenom

plus Teacher and Config_E as the anchors already reported at 5 seeds.

WHAT EACH CELL DECIDES
----------------------
  EvenPow_RowSum  ~= Fix2_Normalized  -> the ReLU is dead weight. Drop ~14
                                         levels per block for free. Fix 2
                                         becomes a strictly worse variant of
                                         our own method.
  Fix2_FixedDenom ~= Fix2_Normalized  -> the division is dead weight, and
                                         Proposition 1 (convex-combination
                                         bound) is NOT the operative mechanism
                                         -- scale control alone suffices.
  EvenPow_FixedDenom recovers          -> both costs are dead weight. The
                                         attention activation collapses to ONE
                                         multiplicative level and the paper's
                                         contribution changes shape entirely.
  Only Fix2_Normalized recovers        -> Fix 2 is justified, Proposition 1 is
                                         load-bearing, and the cost is the
                                         price of the mechanism. Strongest
                                         outcome for the current narrative.

READ THIS BEFORE INTERPRETING THE RESULT
----------------------------------------
The fixed-denominator arms do NOT sum to one, so they are not convex
combinations and Proposition 1 does not apply to them. If they train fine,
the paper's single formal anchor stops explaining the phenomenon it was
introduced to explain. That is a real risk of running this experiment, and it
is the reason it is Gate 1 rather than a nice-to-have: the intro cannot be
written until the answer is known.

COMPARABILITY
-------------
Architecture, data pipeline, KD loss, optimizer, schedule, clipping and
checkpoint-selection rule are IMPORTED from fix2_bloodmnist.py rather than
restated, so no hyperparameter can silently drift between this ablation and
the published 5-seed numbers. (That drift is exactly what confounded the
step5 / step5b cold-vs-warm comparison -- see 01_CORRECTIONS.md.) The new
arms are installed by swapping `blk.attn_act` on an otherwise stock model;
`--selftest` asserts the swap leaves the rest of the network bit-identical.

Teacher checkpoints are reused from ./checkpoints_bloodmnist if present, so
the teacher is the SAME shared teacher (seed 42) the existing results used.

USAGE
    python experiments/active/fix5_evenpower_ablation.py --selftest              # ~30 s, CPU, no data
    python experiments/active/fix5_evenpower_ablation.py --seeds 42 --epochs 6 \
        --arms EvenPow_FixedDenom,Fix2_Normalized             # smoke, ~10 min
    CUDA_VISIBLE_DEVICES=2 python experiments/active/fix5_evenpower_ablation.py  # full 2x2, 5 seeds

"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fix2_bloodmnist import (
    CONFIGS as FIX2_CONFIGS,
    DeiTTiny,
    PolyAttnNormed,
    compute_loss,
    evaluate,
    get_bloodmnist_loaders,
    load_checkpoint,
    set_seed,
)

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[WARN] scipy not available; Welch's t-tests will be skipped.")


# =====================================================================
# 1. THE NEW ATTENTION ACTIVATIONS
# =====================================================================

class EvenPowerAttn(nn.Module):
    """Non-negativity for free: attn = ((a*s + b)^2 + c^2) / denom.

    A perfect square is non-negative on all of R, so the ReLU that Fix 2 uses
    to restore what exp() gave for free is simply unnecessary. Both are
    degree-2 in `s`: ONE ct-ct multiplication, the same multiplicative depth
    as Fix 2's `a*s^2 + b*s + c`, but with no comparison anywhere in the
    circuit.

    DEFECT, found 2026-09-08 by external review, CONFIRMED by direct test.
    `c` was intended as "a learnable positive offset available without
    breaking the guarantee". It is not learnable as initialised. It enters as
    `c*c`, whose derivative 2c is EXACTLY ZERO at the initial value c=0, so
    the parameter receives no gradient and never leaves 0 for the entire run.

        >>> c = torch.tensor(0.0, requires_grad=True)
        >>> (((a*s + b)**2 + c*c).sum()).backward(); c.grad
        tensor(0.)

    Every arm in results/ablations/fix5_evenpower_ablation.json therefore
    ran the function class `(a*s + b)^2`, with NO offset term.

    This does NOT invalidate the Gate 1 result. Both compared arms are equally
    offset-free, so the finding -- row normalisation is load-bearing, the
    non-negativity mechanism is not -- is unaffected. What changes is the
    DESCRIPTION: report the arm as `(a*s + b)^2`, not as `(a*s+b)^2 + c^2`,
    and do not claim a learnable offset was explored.

    `c` is retained at 0.0 deliberately so the shipped code reproduces the
    published numbers exactly. To actually study an offset, initialise it
    nonzero (the gradient is alive anywhere except 0) and re-run -- that is a
    NEW experiment, not a correction to this one.

    INITIALIZATION. Matched to PolyAttnNormed's (a, b, c) = (0.05, 0.5, 0.25)
    on the quadratic and constant terms: a_sq = sqrt(0.05) reproduces the s^2
    coefficient exactly and b_sq = 0.5 reproduces the constant exactly. The
    linear term cannot also match -- a perfect square forces
    2*a_sq*b_sq = 0.2236 where Fix 2 starts at 0.5 -- because a square has a
    double root. This is a property of the function class under test, not a
    tuning choice; it is recorded in the results JSON.

    denom_mode:
        "rowsum" -- poly.sum(-1) + eps. Restores sum-to-one, so rows are a
                    convex combination and Proposition 1 applies. Encrypted
                    cost: a ciphertext division (Goldschmidt, ~2 levels per
                    iteration, 3-5 iterations).
        "fixed"  -- multiply by the constant 1/T, T = sequence length. T is
                    fixed by the architecture (65 tokens = 64 patches + CLS),
                    so at inference this is a plaintext scalar that folds into
                    the value projection: ZERO levels. Rows do NOT sum to one;
                    this arm is deliberately outside Proposition 1.
    """

    def __init__(self, denom_mode: str = "rowsum", eps: float = 1e-6):
        super().__init__()
        if denom_mode not in ("rowsum", "fixed"):
            raise ValueError(f"Unknown denom_mode: {denom_mode}")
        self.denom_mode = denom_mode
        self.eps = eps
        self.a = nn.Parameter(torch.tensor(0.05 ** 0.5))   # -> s^2 coeff 0.05
        self.b = nn.Parameter(torch.tensor(0.5))           # -> constant 0.25
        # INERT BY CONSTRUCTION -- see class docstring. d(c^2)/dc = 2c = 0 at
        # c=0, so this parameter never trains. Kept at 0.0 so the shipped code
        # reproduces the published Gate 1 numbers exactly; the function class
        # actually evaluated is (a*s + b)^2, with no offset.
        self.c = nn.Parameter(torch.tensor(0.0))           # offset, enters as c^2

    def forward(self, scores):
        inner = self.a * scores + self.b
        poly = inner * inner + self.c * self.c
        if self.denom_mode == "rowsum":
            return poly / (poly.sum(dim=-1, keepdim=True) + self.eps)
        return poly * (1.0 / scores.shape[-1])


class PolyAttnFixedDenom(nn.Module):
    """Fix 2 with the division removed: attn = ReLU(poly(s)) * (1/T).

    The control that isolates the ReLU from the row division. Keeps Fix 2's
    comparison and its exact initialization; replaces only the denominator.
    Non-negative but not sum-to-one, so also outside Proposition 1.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.05))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.25))

    def forward(self, scores):
        poly = self.a * scores * scores + self.b * scores + self.c
        return F.relu(poly) * (1.0 / scores.shape[-1])


# =====================================================================
# 2. ARM TABLE + ESTIMATED CKKS DEPTH
# =====================================================================
#
# Depth figures are ESTIMATES from the literature, not measurements, and are
# tagged as such in the results JSON. They exist so the accuracy result is
# never read without its cost. Per ATTENTION ACTIVATION, per block; multiply
# by depth=6 for the backbone. Sources: minimax sign polynomial 13-14 levels
# (Lee et al., IEEE TDSC 2022); Goldschmidt division ~2 levels/iteration at
# 3-5 iterations. See 01_CORRECTIONS.md item 10 -- ReLU is NOT CKKS-native,
# and no arm here may be described as "cheaper overall" on accuracy alone.

def _depth(poly, relu, div):
    total_lo = poly + (13 if relu else 0) + (6 if div else 0)
    total_hi = poly + (14 if relu else 0) + (10 if div else 0)
    return {
        "poly_levels": poly,
        "relu_levels": [13, 14] if relu else 0,
        "division_levels": [6, 10] if div else 0,
        "attn_act_levels_per_block": [total_lo, total_hi],
        "status": "estimated_from_literature_not_measured",
    }


ARMS = {
    "Teacher": {
        "attn_module": None, "kd": False, "stock": "standard",
        "depth": None,
        "note": "LayerNorm + softmax + GELU. Not FHE-evaluable; accuracy anchor only.",
    },
    "Config_E": {
        "attn_module": None, "kd": True, "stock": "poly",
        "depth": _depth(1, False, False),
        "note": "Unbounded polynomial attention. Collapse control.",
    },
    "Fix2_Normalized": {
        "attn_module": lambda: PolyAttnNormed(), "kd": True, "stock": "poly",
        "depth": _depth(1, True, True),
        "note": "Published Fix 2. Pays for BOTH comparison and division.",
    },
    "Fix2_FixedDenom": {
        "attn_module": lambda: PolyAttnFixedDenom(), "kd": True, "stock": "poly",
        "depth": _depth(1, True, False),
        "note": "Isolates the ReLU: keeps comparison, drops division.",
    },
    "EvenPow_RowSum": {
        "attn_module": lambda: EvenPowerAttn("rowsum"), "kd": True, "stock": "poly",
        "depth": _depth(1, False, True),
        "note": "Isolates the division: drops comparison, keeps sum-to-one.",
    },
    "EvenPow_FixedDenom": {
        "attn_module": lambda: EvenPowerAttn("fixed"), "kd": True, "stock": "poly",
        "depth": _depth(1, False, False),
        "note": "Drops both. One multiplicative level. Cheapest arm possible.",
    },
}

DEFAULT_ARMS = ["Fix2_Normalized", "Fix2_FixedDenom",
                "EvenPow_RowSum", "EvenPow_FixedDenom"]


def build_model(arm_name: str, device: str) -> nn.Module:
    """Stock DeiTTiny with `attn_act` swapped for the arm's module.

    Constructing with attn_type="poly" keeps Block.forward on its non-softmax
    branch; only the activation object is replaced. Everything else -- patch
    embed, qkv, proj, MLP, BatchNorm, PolyGELU, init -- is the architecture
    that produced the published numbers. `--selftest` verifies this.
    """
    spec = ARMS[arm_name]
    cfg = FIX2_CONFIGS["Teacher"] if arm_name == "Teacher" else {
        "norm": "batchnorm", "attn": spec["stock"], "gelu": "poly",
    }
    model = DeiTTiny(
        img_size=32, patch_size=4, in_channels=3, num_classes=8,
        embed_dim=192, depth=6, num_heads=3,
        norm_type=cfg["norm"], attn_type=cfg["attn"], gelu_type=cfg["gelu"],
    ).to(device)

    if spec["attn_module"] is not None:
        for blk in model.blocks:
            blk.attn_act = spec["attn_module"]().to(device)
    return model


# =====================================================================
# 3. SIMPLEX INSTRUMENTATION
# =====================================================================

@torch.no_grad()
def measure_row_sums(model, loader, device, max_batches: int = 4):
    """Per-layer attention row-sum statistics on a probe batch.

    This is the measurement that tests Proposition 1 directly. Row sums of
    1.0 mean the layer is a convex combination and the output is bounded by
    the value vectors. Anything else quantifies how far the arm sits outside
    that bound -- which, paired with test accuracy, is what tells us whether
    the bound is doing the work or merely correlating with something that is.
    """
    stats, handles = {}, []

    def hook(idx):
        def fn(_module, _inp, out):
            rs = out.sum(dim=-1)
            s = stats.setdefault(idx, {"mean": [], "min": [], "max": []})
            s["mean"].append(float(rs.mean()))
            s["min"].append(float(rs.min()))
            s["max"].append(float(rs.max()))
        return fn

    for i, blk in enumerate(model.blocks):
        if getattr(blk, "attn_act", None) is not None:
            handles.append(blk.attn_act.register_forward_hook(hook(i)))
    if not handles:
        return {}

    model.eval()
    for bi, (imgs, _) in enumerate(loader):
        if bi >= max_batches:
            break
        model(imgs.to(device))
    for h in handles:
        h.remove()

    return {
        f"layer_{i}": {
            "row_sum_mean": float(np.mean(s["mean"])),
            "row_sum_min": float(np.min(s["min"])),
            "row_sum_max": float(np.max(s["max"])),
            "deviation_from_simplex": float(abs(np.mean(s["mean"]) - 1.0)),
        }
        for i, s in sorted(stats.items())
    }


# =====================================================================
# 4. TRAINING (hyperparameters inherited, model injected)
# =====================================================================

def train_one(arm_name, model, seed, train_loader, val_loader, test_loader,
              teacher_model=None, epochs: int = 150, lr: float = 1e-3,
              wd: float = 0.05, device: str = "cuda",
              ckpt_dir: str = "./checkpoints_fix5", print_every: int = 25):
    """Mirrors fix2_bloodmnist.train_one, but takes a prebuilt model.

    KD loss is the IMPORTED compute_loss called with the same explicit
    T=4.0 / alpha=0.1, so no loss-formula drift can occur between this
    ablation and the published fix2 runs. Checkpoint selection is on balanced
    validation accuracy, identical to the published runs.

    NOTE (corrected 2026-08-19): an earlier version of this docstring blamed
    the step5-vs-step5b confound on a "T^2-scaling" difference. That is false
    -- both scripts' kd_loss() apply temperature**2 identically
    (step5_kd_poly_gelu.py:130-169, step5b_coldstart_kd.py:110-128). The real
    confounds are lr (5e-5 vs 1e-4), warmup length (2 vs 5 epochs), and a
    bundled activation-transition schedule with phase-gated checkpoint
    selection in step5 that step5b does not have. Those differences live in
    the training loop, not the loss, so importing compute_loss does not by
    itself rule them out -- this script avoids them by holding lr, warmup,
    schedule, and checkpoint rule fixed across all arms.
    """
    set_seed(seed)
    use_kd = teacher_model is not None and ARMS[arm_name]["kd"]
    if use_kd:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, best_bal, best_epoch, best_state = 0.0, 0.0, -1, None
    nonfinite_epochs = []
    print(f"\n=== Training {arm_name} seed={seed} ===")
    print(f"    {ARMS[arm_name]['note']}")
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.squeeze(-1).long().to(device)

            optimizer.zero_grad()
            s_logits = model(imgs)
            if use_kd:
                with torch.no_grad():
                    t_logits = teacher_model(imgs)
                loss, _, _ = compute_loss(s_logits, t_logits, labels, T=4.0, alpha=0.1)
            else:
                loss, _, _ = compute_loss(s_logits, None, labels)

            if not torch.isfinite(loss):
                # Squares can overflow where Fix 2's ReLU clamped. Record and
                # skip rather than poisoning the weights with a NaN step --
                # a silently NaN'd arm would read as "collapse" and be wrong.
                nonfinite_epochs.append(epoch)
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        val_acc, val_bal = evaluate(model, val_loader, device)
        if val_bal > best_bal:
            best_bal, best_val, best_epoch = val_bal, val_acc, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % print_every == 0 or epoch == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"val_acc={val_acc*100:.2f}% | val_bal={val_bal*100:.2f}% | "
                  f"best_bal={best_bal*100:.2f}%@{best_epoch+1} | "
                  f"{time.time()-t_start:.0f}s")

    if best_state is None:
        print(f"  [FAIL] {arm_name} seed={seed}: no finite epoch. Recorded as divergence.")
        return {"arm": arm_name, "seed": seed, "test_acc": 0.0, "test_bal_acc": 0.0,
                "diverged": True, "nonfinite_batches": len(nonfinite_epochs)}

    model.load_state_dict(best_state)
    test_acc, test_bal_acc = evaluate(model, test_loader, device)
    row_sums = measure_row_sums(model, test_loader, device)
    elapsed = time.time() - t_start

    print(f"  FINAL: test_acc={test_acc*100:.2f}% | test_bal={test_bal_acc*100:.2f}% "
          f"| {elapsed:.0f}s")
    if row_sums:
        dev = [v["deviation_from_simplex"] for v in row_sums.values()]
        print(f"  row-sum deviation from simplex: "
              f"mean={np.mean(dev):.4f} max={np.max(dev):.4f}")
    if nonfinite_epochs:
        print(f"  [WARN] {len(nonfinite_epochs)} non-finite batches skipped.")

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{arm_name}_seed{seed}.pt")
    torch.save({"state_dict": best_state, "arm": arm_name, "seed": seed,
                "test_acc": test_acc, "test_bal_acc": test_bal_acc,
                "best_val": best_val, "best_val_bal": best_bal,
                "best_epoch": best_epoch}, ckpt_path)

    return {
        "arm": arm_name, "seed": seed,
        "test_acc": test_acc, "test_bal_acc": test_bal_acc,
        "best_val_acc": best_val, "best_val_bal": best_bal,
        "best_epoch": best_epoch, "total_epochs": epochs,
        "time_seconds": elapsed, "diverged": False,
        "nonfinite_batches": len(nonfinite_epochs),
        "row_sum_stats": row_sums, "ckpt_path": ckpt_path,
    }


# =====================================================================
# 5. SELF-TEST (CPU, no data, no GPU)
# =====================================================================

def selftest():
    """Functional assertions. Run before committing GPU hours."""
    print("=" * 72)
    print("SELF-TEST")
    print("=" * 72)
    n = 0

    def check(label, cond):
        nonlocal n
        assert cond, f"FAILED: {label}"
        n += 1
        print(f"  ok  {label}")

    torch.manual_seed(0)
    scores = torch.randn(2, 3, 65, 65) * 10.0   # deliberately wide

    # -- non-negativity, the core claim of the even-power arm -------------
    for mode in ("rowsum", "fixed"):
        m = EvenPowerAttn(mode)
        with torch.no_grad():
            m.c.fill_(0.7)
        out = m(scores)
        check(f"EvenPower[{mode}] non-negative everywhere", bool((out >= 0).all()))
        check(f"EvenPower[{mode}] finite", bool(torch.isfinite(out).all()))

    m = EvenPowerAttn("rowsum")
    check("EvenPower is exactly (a*s+b)^2 + c^2 (degree 2, one ct-ct mult)",
          torch.allclose((m.a * scores + m.b) ** 2 + m.c ** 2,
                         m(scores) * (((m.a * scores + m.b) ** 2 + m.c ** 2)
                                      .sum(-1, keepdim=True) + m.eps), atol=1e-4))

    # -- initialization matches Fix 2 on s^2 and constant -----------------
    m0 = EvenPowerAttn("rowsum")
    check("init: s^2 coefficient == Fix 2's 0.05",
          abs(float(m0.a) ** 2 - 0.05) < 1e-6)
    check("init: constant term == Fix 2's 0.25",
          abs(float(m0.b) ** 2 - 0.25) < 1e-6)
    check("init: linear term is 0.2236, NOT Fix 2's 0.5 (square has a double root)",
          abs(2 * float(m0.a) * float(m0.b) - 0.2236) < 1e-3)

    # -- simplex behaviour: who is inside Proposition 1, who is not -------
    rs = EvenPowerAttn("rowsum")(scores).sum(-1)
    check("EvenPow_RowSum rows sum to 1 (inside Prop. 1)",
          torch.allclose(rs, torch.ones_like(rs), atol=1e-4))
    check("Fix2_Normalized rows sum to 1 (inside Prop. 1)",
          torch.allclose(PolyAttnNormed()(scores).sum(-1),
                         torch.ones(2, 3, 65), atol=1e-4))
    fd = EvenPowerAttn("fixed")(scores).sum(-1)
    check("EvenPow_FixedDenom rows do NOT sum to 1 (outside Prop. 1, by design)",
          not torch.allclose(fd, torch.ones_like(fd), atol=1e-2))
    check("Fix2_FixedDenom non-negative", bool((PolyAttnFixedDenom()(scores) >= 0).all()))

    # -- the swap does not perturb the rest of the architecture -----------
    set_seed(123)
    stock = DeiTTiny(img_size=32, patch_size=4, in_channels=3, num_classes=8,
                     embed_dim=192, depth=6, num_heads=3, norm_type="batchnorm",
                     attn_type="poly_normed", gelu_type="poly")
    set_seed(123)
    swapped = build_model("Fix2_Normalized", "cpu")
    check("swapped model has identical parameter count to stock poly_normed",
          sum(p.numel() for p in stock.parameters())
          == sum(p.numel() for p in swapped.parameters()))
    check("swapped model has identical state_dict keys",
          set(stock.state_dict().keys()) == set(swapped.state_dict().keys()))
    swapped.load_state_dict(stock.state_dict())
    stock.eval(); swapped.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        check("swapped model is numerically identical to stock poly_normed",
              torch.allclose(stock(x), swapped(x), atol=1e-6))

    # -- every arm trains ------------------------------------------------
    for arm in ARMS:
        set_seed(7)
        mdl = build_model(arm, "cpu")
        out = mdl(x)
        check(f"{arm}: forward shape (4, 8)", tuple(out.shape) == (4, 8))
        check(f"{arm}: forward finite", bool(torch.isfinite(out).all()))
        out.sum().backward()
        g = [p.grad for p in mdl.parameters() if p.grad is not None]
        check(f"{arm}: backward produces finite gradients",
              len(g) > 0 and all(torch.isfinite(t).all() for t in g))

    # -- depth bookkeeping -----------------------------------------------
    check("EvenPow_FixedDenom is the cheapest arm (1 level)",
          ARMS["EvenPow_FixedDenom"]["depth"]["attn_act_levels_per_block"] == [1, 1])
    check("Fix2_Normalized is the most expensive arm",
          ARMS["Fix2_Normalized"]["depth"]["attn_act_levels_per_block"][1]
          == max(a["depth"]["attn_act_levels_per_block"][1]
                 for a in ARMS.values() if a["depth"]))

    print(f"\n{n} assertions passed.\n")
    return n


# =====================================================================
# 6. SUMMARY
# =====================================================================

def summarize(all_results: dict) -> dict:
    summary = {"per_arm": {}, "tests": {}, "depth_accounting": {}}
    for name, runs in all_results.items():
        if not runs:
            continue
        accs = np.array([r["test_acc"] for r in runs])
        bal = np.array([r["test_bal_acc"] for r in runs])
        summary["per_arm"][name] = {
            "test_acc_mean": float(accs.mean()),
            "test_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "test_acc_seeds": accs.tolist(),
            "test_bal_acc_mean": float(bal.mean()),
            "test_bal_acc_std": float(bal.std(ddof=1)) if len(bal) > 1 else 0.0,
            "test_bal_acc_seeds": bal.tolist(),
            "n_diverged": int(sum(r.get("diverged", False) for r in runs)),
            "n_seeds": int(len(accs)),
            "row_sum_stats_last_seed": runs[-1].get("row_sum_stats", {}),
        }
        summary["depth_accounting"][name] = ARMS[name]["depth"]

    if HAVE_SCIPY:
        ref = "Fix2_Normalized"
        for name in all_results:
            if name == ref or not all_results.get(name) or not all_results.get(ref):
                continue
            a = np.array([r["test_acc"] for r in all_results[name]])
            b = np.array([r["test_acc"] for r in all_results[ref]])
            ba = np.array([r["test_bal_acc"] for r in all_results[name]])
            bb = np.array([r["test_bal_acc"] for r in all_results[ref]])
            if len(a) > 1 and len(b) > 1:
                t_acc, p_acc = scipy_stats.ttest_ind(a, b, equal_var=False)
                t_bal, p_bal = scipy_stats.ttest_ind(ba, bb, equal_var=False)
                summary["tests"][f"{name}_vs_{ref}"] = {
                    "acc_t": float(t_acc), "acc_p": float(p_acc),
                    "acc_diff_pp": float((a.mean() - b.mean()) * 100),
                    "bal_t": float(t_bal), "bal_p": float(p_bal),
                    "bal_diff_pp": float((ba.mean() - bb.mean()) * 100),
                }
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 84)
    print("GATE 1 RESULTS: even-power / fixed-denominator ablation (BloodMNIST)")
    print("=" * 84)
    print(f"\n{'Arm':<22}{'test_acc':>18}{'bal_acc':>18}{'attn levels/blk':>18}{'div':>6}")
    print("-" * 84)
    for name, s in summary["per_arm"].items():
        d = summary["depth_accounting"].get(name)
        lv = "n/a (softmax)" if d is None else str(d["attn_act_levels_per_block"])
        flag = "  !" if s["n_diverged"] else ""
        print(f"{name:<22}"
              f"{s['test_acc_mean']*100:>11.2f} +/-{s['test_acc_std']*100:>5.2f}"
              f"{s['test_bal_acc_mean']*100:>11.2f} +/-{s['test_bal_acc_std']*100:>5.2f}"
              f"{lv:>18}{s['n_diverged']:>4}{flag}")

    if summary["tests"]:
        print("\nWelch's t-tests vs Fix2_Normalized (two-sided):")
        for k, t in summary["tests"].items():
            star = "  *" if t["acc_p"] < 0.05 else ""
            print(f"  {k:<40} dAcc={t['acc_diff_pp']:+6.2f}pp  "
                  f"p={t['acc_p']:.4f}{star}")

    print("\nSimplex deviation (mean |row_sum - 1| across layers, last seed):")
    for name, s in summary["per_arm"].items():
        rs = s.get("row_sum_stats_last_seed") or {}
        if rs:
            devs = [v["deviation_from_simplex"] for v in rs.values()]
            print(f"  {name:<22} {np.mean(devs):.4f}")

    # ---- interpretation ------------------------------------------------
    per = summary["per_arm"]
    if "Fix2_Normalized" not in per:
        print("\n(No Fix2_Normalized reference arm in this run; skipping verdict.)")
        print("=" * 84)
        return

    ref = per["Fix2_Normalized"]["test_bal_acc_mean"]
    tol = 0.02   # 2pp balanced accuracy = "matches"
    matches = {n: (s["test_bal_acc_mean"] >= ref - tol)
               for n, s in per.items() if n in DEFAULT_ARMS}

    print("\nVerdict:")
    if matches.get("EvenPow_FixedDenom"):
        print("  -> BOTH costs are unnecessary. The attention activation drops to ONE")
        print("     multiplicative level. Fix 2 is not the method; it is an expensive")
        print("     special case. Proposition 1 does NOT explain the recovery -- the")
        print("     fixed-denominator arm is outside the simplex and trains anyway.")
        print("     ACTION: the paper's formal anchor must be replaced before the")
        print("     intro is written. Scale control, not convexity, is the mechanism.")
    elif matches.get("EvenPow_RowSum") and not matches.get("Fix2_FixedDenom"):
        print("  -> The DIVISION is load-bearing; the ReLU is not. Proposition 1")
        print("     survives intact. Drop the comparison: ~13-14 levels per block")
        print("     saved at no accuracy cost. This strengthens the paper.")
    elif matches.get("Fix2_FixedDenom") and not matches.get("EvenPow_RowSum"):
        print("  -> The COMPARISON is load-bearing; the division is not. Proposition 1")
        print("     is not the operative mechanism -- non-negativity alone suffices.")
        print("     Reframe the theory section around boundedness, not convexity.")
    elif not any(matches.get(a) for a in DEFAULT_ARMS if a != "Fix2_Normalized"):
        print("  -> Fix 2 is JUSTIFIED. Every cheaper route fails; both properties are")
        print("     necessary together. Proposition 1 is load-bearing and the cost is")
        print("     the price of the mechanism. Strongest outcome for the current")
        print("     narrative -- but report the depth table anyway (Flaw 2).")
    else:
        print("  -> Mixed result. Both single-property arms match Fix 2 but the")
        print("     zero-cost arm does not: the properties are partially redundant.")
        print("     Report all four cells; do not compress this into one claim.")

    print("\n  NOTE: depth figures above are literature estimates, NOT measured.")
    print("  No arm may be called 'cheaper overall' until the ReLU consistency")
    print("  check (Flaw 1) closes the train-vs-deploy mismatch.")
    print("=" * 84)


# =====================================================================
# 7. MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--arms", type=str, default=",".join(DEFAULT_ARMS),
                        help="comma-separated; 'all' includes Teacher and Config_E")
    parser.add_argument("--teacher-seed", type=int, default=42)
    parser.add_argument("--teacher-ckpt-dir", type=str,
                        default="./checkpoints_bloodmnist",
                        help="reuses the SHARED teacher from the published runs")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints_fix5")
    parser.add_argument("--results-dir", type=str, default="./results/ablations")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    arms = list(ARMS) if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"Unknown arm(s): {unknown}. Choose from {list(ARMS)}")

    print(f"[selftest] running before GPU commit ...")
    selftest()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Arms   : {arms}")
    print(f"Seeds  : {args.seeds}   Epochs: {args.epochs}")

    train_loader, val_loader, test_loader = get_bloodmnist_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root)
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    # Shared teacher -- reused, never retrained, so KD signal is identical
    # to the published Fix 2 / Config E runs.
    teacher_ckpt = os.path.join(args.teacher_ckpt_dir,
                                f"Teacher_seed{args.teacher_seed}.pt")
    teacher_model = None
    if os.path.exists(teacher_ckpt):
        teacher_model, obj = load_checkpoint(teacher_ckpt, device)
        print(f"Loaded shared teacher seed={args.teacher_seed}: "
              f"test_acc={obj['test_acc']*100:.2f}% bal={obj['test_bal_acc']*100:.2f}%")
    else:
        raise SystemExit(
            f"No teacher checkpoint at {teacher_ckpt}.\n"
            "Run fix2_bloodmnist.py first so this ablation distills from the SAME\n"
            "teacher as the published numbers. Training a fresh teacher here would\n"
            "reintroduce exactly the confound 01_CORRECTIONS.md item on cold-start\n"
            "KD warns about.")

    all_results = {a: [] for a in arms}
    for arm in arms:
        for seed in args.seeds:
            ckpt = os.path.join(args.ckpt_dir, f"{arm}_seed{seed}.pt")
            if args.skip_train and os.path.exists(ckpt):
                o = torch.load(ckpt, map_location="cpu", weights_only=False)
                print(f"\n[SKIP] {arm} seed={seed} (cached "
                      f"test_acc={o['test_acc']*100:.2f}%)")
                all_results[arm].append({
                    "arm": arm, "seed": seed, "test_acc": o["test_acc"],
                    "test_bal_acc": o["test_bal_acc"], "diverged": False,
                    "row_sum_stats": {}, "ckpt_path": ckpt})
                continue

            set_seed(seed)
            model = build_model(arm, device)
            all_results[arm].append(train_one(
                arm, model, seed, train_loader, val_loader, test_loader,
                teacher_model=teacher_model, epochs=args.epochs,
                lr=args.lr, wd=args.wd, device=device, ckpt_dir=args.ckpt_dir))

    summary = summarize(all_results)
    print_summary(summary)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "fix5_evenpower_ablation.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args), "results": all_results, "summary": summary,
            "arm_notes": {k: v["note"] for k, v in ARMS.items()},
            "dataset": "BloodMNIST", "num_classes": 8, "img_size": 32,
            "patch_size": 4, "depth": 6, "embed_dim": 192, "num_heads": 3,
            "n_tokens": 65, "kd_T": 4.0, "kd_alpha": 0.1, "grad_clip": 5.0,
            "teacher": "shared, seed 42, reused from fix2_bloodmnist runs",
            "depth_disclaimer": (
                "Level counts are literature estimates (minimax ReLU 13-14, "
                "Lee et al. TDSC 2022; Goldschmidt ~2 levels/iter x 3-5 iters), "
                "not measurements. Do not publish as measured."),
        }, f, indent=2, default=float)
    print(f"\nResults JSON: {out_path}")


if __name__ == "__main__":
    main()
