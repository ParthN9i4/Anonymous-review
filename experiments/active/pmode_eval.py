"""
pmode_eval.py -- the S -> P comparison for a trained substituted checkpoint.
============================================================================
Added 2026-09-14.

FOUR EXECUTION ROLES, and this script covers the middle boundary
----------------------------------------------------------------
  O  the ordinary reference model (real softmax / GELU / LayerNorm)
  S  the trained SUBSTITUTED model, exact PyTorch arithmetic
  P  the SAME checkpoint with every forbidden operation inside the declared
     boundary replaced by its intended deployment approximation, in plaintext
  C  actual ciphertext execution of that same computation

O -> S mixes architecture and training. S -> P isolates approximation and
compilation. P -> C isolates encrypted arithmetic. This script measures S -> P
on identical weights and identical inputs, which is the comparison that turns
the repo's domain-coverage audit into a statement about predictions.

WHAT "P-MODE" HAS TO MEAN
-------------------------
Evaluating exact PyTorch division inside something called "P-mode" does not
test the reciprocal approximation -- it tests nothing at all. P here replaces
BOTH forbidden operations in the Fix 2 attention:

  1. the row reciprocal 1/rowsum  ->  Goldschmidt iteration, the exact
     recurrence measured in measure_encrypted_primitives.py:119, run in
     plaintext so its divergence is attributable to the approximation and not
     to CKKS noise;
  2. the non-negativity step (ReLU or clamp) -> a finite LUT, the repo's own
     declared deployment surrogate (lut_precision_sweep.py:90). A ReLU is a
     comparison; under CKKS it is not free and must not be smuggled in exactly.

An external final softmax may be excluded ONLY if that boundary is declared.
`--boundary` records the declaration in the output.

THE DOMAIN RESULT THIS IS DESIGNED TO CASH IN
---------------------------------------------
The obvious criterion -- "d' = d*scale must lie in (0, 2) so that |e0| < 1" --
is NECESSARY BUT NOT SUFFICIENT, and taking it for the whole story is an error
this file's own self-test caught. At the BloodMNIST scale, d = 1e-6 gives
d' = 2e-8, comfortably inside (0, 2), and the k=8 iteration still returns 10.60
instead of 1e6.

The recurrence telescopes, which gives the exact behaviour:

    (1 - e0) * prod_{i=0..k} (1 + e0^(2^i)) = 1 - e0^(2^(k+1))
    =>  x_k = (1 - e0^(2^(k+1))) / d'        relative error = (1 - d')^(2^(k+1))

Two consequences, both verified numerically against this file's self-test with
the repo's own calibration rule (scale = 1/(max*1.05), lut_precision_sweep.py:261):

  1. THE FAILURE IS QUIET. As d -> 0, x_k -> 2^(k+1), so the reciprocal returns
     the CONSTANT 2^(k+1)*scale, independent of d. It does not blow up; it
     silently stops being a reciprocal. At the BloodMNIST scale that constant
     is 10.5981 at k=8 (measured: 10.5981) and 2713.12 at k=16 (measured:
     2709.44).

  2. THE USABLE RANGE IS NARROW AND THE MARGIN IS ~ZERO. Accuracy to `tol`
     needs d >= -ln(tol) / (2^(k+1) * scale). At k=8 (depth 17) and tol=1e-14
     that is d >= 3.04, while the smallest BloodMNIST row sum actually observed
     is 2.95 -- just below, measuring 9.8e-15, i.e. sitting exactly on the
     tolerance line. Raising k only raises the ceiling linearly in 2^k while
     depth grows as 2k+1: recovering 1/d at d = 1e-6 needs k >= 25, i.e. 51
     levels, more than the entire CKKS budget. "Use more iterations" is not an
     available fix.

Scale-independence holds for the reason the closed form shows: rescaling moves
the saturation ceiling and the denominators together. BOTH Fix 2 variants reach
this regime -- clamp vs ReLU changes the attention output, not the violation.

Crucially, a violated row does NOT automatically mean a wrong prediction: a
zero numerator times a finite inaccurate reciprocal is still zero in exact
arithmetic, so an exact zero row breaks coverage without by itself breaking the
output. That is why exact zero rows and tiny positive denominators are counted
SEPARATELY, and why the headline number is the violation-conditioned flip rate
against the non-violating flip rate -- not the violation count on its own.

Usage
-----
  # numerical core, no checkpoint and no dataset required
  python experiments/active/pmode_eval.py --self-test

  # the real comparison (on the H200, where the checkpoints live)
  python experiments/active/pmode_eval.py \
      --checkpoint checkpoints_substitution_bloodmnist/F:\\ softmax+norm_seed42.pt \
      --iters 8 --relu-bits 12 --out results/ablations/pmode_bloodmnist_f_s42.json
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn


# ── The approximation primitives ─────────────────────────────────────

def goldschmidt_reciprocal(d, scale, iters):
    """1/d by Goldschmidt on the scaled input d' = d*scale, in plaintext.

    Mirrors measure_encrypted_primitives.py:119 term for term:

        e_0 = 1 - d'        x_0 = 1 + e_0
        e_{i+1} = e_i^2     x_{i+1} = x_i * (1 + e_i^2)
        1/d = scale * x_k

    Depth 2*iters + 1 under CKKS. |e_0| < 1 (i.e. d' in (0, 2)) is required for
    the residual to contract at all, but it is NOT enough at a finite budget:
    see goldschmidt_rel_err and goldschmidt_saturation for the exact behaviour.
    Run in plaintext on purpose -- the failure is a property of the
    approximation, not of ciphertext noise, so it must show up identically here.
    """
    d_scaled = d * scale
    e = 1.0 - d_scaled
    x = 1.0 + e
    for _ in range(iters):
        e = e * e
        x = x * (1.0 + e)
    return x * scale


def in_convergence_window(d, scale):
    """Bool mask: d' = d*scale strictly inside (0, 2), so |e_0| < 1.

    NECESSARY BUT NOT SUFFICIENT. Kept separate from `goldschmidt_usable`
    precisely because treating it as the criterion is the mistake this file's
    self-test caught: d = 1e-6 at the BloodMNIST scale gives d' = 2e-8, which
    IS inside (0, 2), and the k=8 iteration still returns 10.60 instead of 1e6.
    """
    d_scaled = d * scale
    return (d_scaled > 0.0) & (d_scaled < 2.0)


def goldschmidt_rel_err(d, scale, iters):
    """Exact relative error of the k-iteration result: (1 - d*scale)^(2^(k+1)).

    The recurrence telescopes --

        (1 - e_0) * prod_{i=0..k} (1 + e_0^(2^i)) = 1 - e_0^(2^(k+1))

    so x_k = (1 - e_0^(2^(k+1))) / d', and since the exact answer is 1/d', the
    relative error is exactly e_0^(2^(k+1)) with e_0 = 1 - d'. Verified against
    the measured values: d=2.95 predicts 9.768e-15 and measures 9.83e-15.

    The error therefore depends on the ITERATION BUDGET, not only on the
    interval. Accuracy needs 2^(k+1) * d' >> 1, i.e. a denominator that is
    small relative to 1/scale needs exponentially many iterations.
    """
    e0 = 1.0 - d * scale
    if torch.is_tensor(e0):
        return e0.abs() ** (2 ** (iters + 1))
    return abs(e0) ** (2 ** (iters + 1))


def goldschmidt_saturation(scale, iters):
    """What the k-iteration reciprocal returns as d -> 0: 2^(k+1) * scale.

    This is the quiet part of the failure. A denominator too small for the
    budget does NOT blow up -- x_k -> 2^(k+1), a CONSTANT INDEPENDENT OF d.
    The reciprocal silently stops being a reciprocal. Verified: at the
    BloodMNIST scale this is 10.5981 for k=8 (measured 10.5981) and 2713.12
    for k=16 (measured 2709.44).

    It also bounds what more iterations can buy: to represent 1/d honestly the
    ceiling must exceed 1/d, so k >= log2(1/(d*scale)) - 1 while DEPTH grows as
    2k+1. At d = 1e-6 that is k >= 25, i.e. 51 levels -- more than the whole
    CKKS budget. "Use more Goldschmidt iterations" is not an available fix.
    """
    return (2.0 ** (iters + 1)) * scale


def goldschmidt_usable(d, scale, iters, tol=1e-6):
    """Bool mask: the k-iteration result meets `tol` relative error at d.

    This is the predicate that should gate a deployment claim, not the (0, 2)
    interval. Equivalent closed form of the lower bound:

        d >= -ln(tol) / (2^(k+1) * scale)
    """
    return goldschmidt_rel_err(d, scale, iters) <= tol


def goldschmidt_min_usable_d(scale, iters, tol=1e-6):
    """Smallest denominator the k-iteration reciprocal represents to `tol`."""
    return -math.log(tol) / ((2.0 ** (iters + 1)) * scale)


class BoundedLUT:
    """Uniform quantisation of a scalar function on a FIXED [lo, hi].

    A finite table spans a fixed domain; inputs outside it are CLAMPED, and
    more bits does not widen the range -- it only subdivides it. That is the
    trap job 473 fell into (lut_precision_sweep.py:185): a mis-sized domain is
    invisible on the bit axis and looks exactly like a precision floor. The
    clamp rate is therefore reported next to the bit width, always.
    """

    def __init__(self, fn, lo, hi, bits):
        self.lo, self.hi, self.bits = float(lo), float(hi), int(bits)
        self.n = 1 << int(bits)
        grid = torch.linspace(self.lo, self.hi, self.n)
        self.table = fn(grid)
        self.clamped = 0
        self.seen = 0

    def __call__(self, x):
        out_of_range = (x < self.lo) | (x > self.hi)
        self.clamped += int(out_of_range.sum())
        self.seen += int(x.numel())
        xc = x.clamp(self.lo, self.hi)
        idx = ((xc - self.lo) / (self.hi - self.lo) * (self.n - 1)).round().long()
        return self.table.to(x.device)[idx]

    @property
    def clamp_rate(self):
        return self.clamped / self.seen if self.seen else 0.0


# ── The P-mode attention operator ────────────────────────────────────

class PModeAttn(nn.Module):
    """Fix 2 attention with BOTH forbidden operations approximated.

    Constructed from a trained PolyAttnNormed of either variant -- the
    coefficients are copied, never re-initialised, because the point is to
    evaluate the deployed weights rather than a fresh fit. `variant` records
    which operator the checkpoint actually trained ('relu' or 'clamp'); it
    changes the non-negativity step and the epsilon, and it is the distinction
    that OPERATOR_ID exists to keep straight.
    """

    def __init__(self, src, variant, scale, iters, relu_bits=None,
                 relu_range=(-10.0, 10.0), exact_recip=False, tol=1e-6):
        super().__init__()
        self.a = nn.Parameter(src.a.detach().clone(), requires_grad=False)
        self.b = nn.Parameter(src.b.detach().clone(), requires_grad=False)
        self.c = nn.Parameter(src.c.detach().clone(), requires_grad=False)
        self.variant = variant
        self.eps = 1e-6 if variant == "relu" else 1e-8
        self.floor = 0.0 if variant == "relu" else 1e-6
        self.scale, self.iters = float(scale), int(iters)
        self.tol = float(tol)   # relative-error budget the reciprocal must meet
        self.exact_recip = bool(exact_recip)   # S-mode switch, for the paired run
        self.relu_lut = None
        if relu_bits is not None:
            fn = (torch.relu if variant == "relu"
                  else (lambda t: t.clamp(min=1e-6)))
            self.relu_lut = BoundedLUT(fn, relu_range[0], relu_range[1], relu_bits)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "rows": 0,
            "rows_exact_zero_denom": 0,     # numerator identically zero
            "rows_tiny_positive_denom": 0,  # positive but below the window
            "rows_out_of_domain": 0,        # rel err > tol at THIS budget
            "rows_outside_0_2_window": 0,   # the weaker, necessary-only test
            "denom_min": float("inf"),
            "denom_max": float("-inf"),
        }

    def forward(self, scores):
        poly = self.a * scores * scores + self.b * scores + self.c
        if self.relu_lut is not None:
            nonneg = self.relu_lut(poly)
        elif self.variant == "relu":
            nonneg = torch.relu(poly)
        else:
            nonneg = poly.clamp(min=1e-6)

        denom = nonneg.sum(dim=-1, keepdim=True) + self.eps

        with torch.no_grad():
            d = denom.detach().flatten()
            self.stats["rows"] += int(d.numel())
            self.stats["denom_min"] = min(self.stats["denom_min"], float(d.min()))
            self.stats["denom_max"] = max(self.stats["denom_max"], float(d.max()))
            # Usability at THIS iteration budget, not the bare (0,2) interval.
            inside = goldschmidt_usable(d, self.scale, self.iters, self.tol)
            self.stats["rows_out_of_domain"] += int((~inside).sum())
            self.stats["rows_outside_0_2_window"] += int(
                (~in_convergence_window(d, self.scale)).sum())
            # An EXACT zero row (all-negative polynomial, ReLU'd to nothing) is
            # a different object from a merely small positive denominator: its
            # numerator is zero too, so the output stays zero whatever the
            # reciprocal returns. Counting them together would overstate the
            # damage.
            exact_zero = d <= self.eps * 1.000001
            self.stats["rows_exact_zero_denom"] += int(exact_zero.sum())
            self.stats["rows_tiny_positive_denom"] += int(
                ((~inside) & (~exact_zero)).sum())

        if self.exact_recip:
            return nonneg / denom
        return nonneg * goldschmidt_reciprocal(denom, self.scale, self.iters)


# ── Wiring ───────────────────────────────────────────────────────────

def swap_attention(model, variant, scale, iters, relu_bits, relu_range,
                   exact_recip, tol=1e-6):
    """Replace every trained PolyAttnNormed in `model` with a PModeAttn.

    Walks the module tree rather than rebuilding the model, so the swap cannot
    silently change anything else about the architecture.
    """
    swapped = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if type(child).__name__ == "PolyAttnNormed":
                new = PModeAttn(child, variant, scale, iters, relu_bits,
                                relu_range, exact_recip, tol)
                setattr(parent, name, new.to(next(model.parameters()).device))
                swapped.append(new)
    return swapped


def calibrate_scale(model, loader, device, max_batches=40, margin=1.05):
    """scale = 1 / (max observed row-sum * margin), the repo's own rule.

    Calibration reads TRAINING data only. Under the MLaaS threat model the
    model owner fixes the circuit's constants from data they hold; the client's
    inputs arrive encrypted. Calibrating on test would both leak and
    misrepresent what is deployable (lut_precision_sweep.py:185).
    """
    seen = []

    def hook(mod, inp, out):
        seen.append(float(out.detach().flatten().max()))

    handles = [m.register_forward_hook(
                   lambda mo, i, o, _m=m: seen.append(
                       float((_m.a * i[0] ** 2 + _m.b * i[0] + _m.c)
                             .clamp(min=0).sum(dim=-1).max())))
               for m in model.modules() if type(m).__name__ == "PolyAttnNormed"]
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            model(batch[0].to(device))
    for h in handles:
        h.remove()
    if not seen:
        raise SystemExit("[FATAL] calibration observed no PolyAttnNormed rows")
    return 1.0 / (max(seen) * margin), max(seen)


# ── Self-test: the numerical core, no checkpoint or dataset needed ────

def self_test():
    """Verify the numerical core. Needs no checkpoint and no dataset.

    Every expectation here is a CLOSED FORM checked against the iteration, not
    a golden value copied from a previous run -- so if the recurrence is ever
    edited, this fails rather than quietly re-baselining.
    """
    ok = True
    max_rowsum = 46.01        # BloodMNIST measured (encrypted_primitives.json)
    scale = 1.0 / (max_rowsum * 1.05)
    K = 8
    dtype = torch.float64

    def gs(d, sc=scale, k=K):
        return float(goldschmidt_reciprocal(torch.tensor([d], dtype=dtype), sc, k))

    print("GOLDSCHMIDT RECIPROCAL -- plaintext, repo recurrence\n")
    print(f"  calibration          : max row-sum {max_rowsum}, "
          f"scale = 1/(max*1.05) = {scale:.6e}")
    print(f"  convergence window   : d*scale in (0,2)  ->  d < {2/scale:.2f}  "
          f"(NECESSARY ONLY)")
    print(f"  usable at k={K}, 1e-14 : d >= "
          f"{goldschmidt_min_usable_d(scale, K, 1e-14):.3f}   <- the real criterion")
    print(f"  saturation as d -> 0 : {goldschmidt_saturation(scale, K):.4f} "
          f"(a CONSTANT, independent of d)\n")

    print(f"  {'d':>10} {'d*scale':>11} {'in(0,2)':>8} {'usable':>7} "
          f"{'1/d exact':>12} {'Goldschmidt':>12} {'rel err':>9} {'predicted':>10}")
    print("  " + "-" * 88)
    for d in (2.95, 10.0, 46.01, 6.5e-5, 1e-6):
        t = torch.tensor([d], dtype=dtype)
        got, exact = gs(d), 1.0 / d
        rel = abs(got - exact) / abs(exact)
        pred = float(goldschmidt_rel_err(t, scale, K))
        win = bool(in_convergence_window(t, scale)[0])
        use = bool(goldschmidt_usable(t, scale, K, 1e-14)[0])
        # The closed form must predict the measured error.
        agree = (abs(rel - pred) <= 1e-9 + 0.02 * pred)
        ok &= agree
        print(f"  {d:>10.6g} {d*scale:>11.6g} {str(win):>8} {str(use):>7} "
              f"{exact:>12.6g} {got:>12.6g} {rel:>9.2e} {pred:>10.2e}"
              f"{'' if agree else '   <- MISPREDICTED'}")
    print(f"\n  closed form (1-d*scale)^(2^(k+1)) predicts every measured "
          f"error ... {'OK' if ok else 'FAIL'}")

    # The point of the correction: inside (0,2) is NOT sufficient.
    t = torch.tensor([1e-6], dtype=dtype)
    win = bool(in_convergence_window(t, scale)[0])
    use = bool(goldschmidt_usable(t, scale, K, 1e-14)[0])
    sep = win and not use
    ok &= sep
    print(f"\n  d=1e-6 is INSIDE (0,2) [{win}] yet NOT usable at k={K} [{not use}]"
          f" ... {'OK' if sep else 'FAIL'}")
    print("    (this is why the interval must never be the deployment criterion)")

    # Saturation: the returned value is the constant, not a big number.
    print(f"\n  saturation 2^(k+1)*scale predicts the returned value as d -> 0:")
    for k in (2, 4, 8, 16):
        pred = goldschmidt_saturation(scale, k)
        got = gs(1e-6, k=k)
        hit = abs(got - pred) <= 0.002 * pred
        ok &= hit
        print(f"    k={k:<3} predicted {pred:>10.4f}   measured {got:>10.4f}"
              f"   {'OK' if hit else 'FAIL'}")

    # Scale independence -- ceiling and denominators move together.
    print(f"\n  scale independence (d=1e-6, k={K}): the answer tracks the ceiling")
    for mx in (4.601, 46.01, 460.1):
        sc = 1.0 / (mx * 1.05)
        got, pred = gs(1e-6, sc), goldschmidt_saturation(sc, K)
        hit = abs(got - pred) <= 0.002 * pred
        ok &= hit
        print(f"    max row-sum {mx:>8.3f} -> ceiling {pred:>9.4f}  "
              f"returned {got:>9.4f}   {'OK' if hit else 'FAIL'}  (exact 1e6)")

    # Depth cost of "just use more iterations".
    need_k = math.ceil(math.log2(1e6 / scale) - 1)
    print(f"\n  to represent 1/d at d=1e-6 the ceiling must exceed 1e6:")
    print(f"    k >= {need_k}  ->  CKKS depth 2k+1 = {2*need_k+1} levels "
          f"(the whole budget, and then some)")
    ok &= (goldschmidt_saturation(scale, need_k) >= 1e6)

    # The deployed margin, on the numbers this project measured.
    lo = goldschmidt_min_usable_d(scale, K, 1e-14)
    print(f"\n  DEPLOYED MARGIN at k={K}, tol=1e-14:")
    print(f"    smallest usable denominator   {lo:.4f}")
    print(f"    smallest BloodMNIST row sum   2.9500  (encrypted_primitives.json)")
    print(f"    margin                        {2.95 - lo:+.4f}  <- NEGATIVE; "
          f"measured rel err at 2.95 is {gs(2.95) and abs(gs(2.95)-1/2.95)*2.95:.2e}")

    # An exact zero row stays zero whatever the reciprocal returns.
    prod = float(torch.zeros(1, dtype=dtype)
                 * goldschmidt_reciprocal(torch.tensor([1e-6], dtype=dtype), scale, K))
    ok &= (prod == 0.0)
    print(f"\n  exact zero row: 0 * (saturated reciprocal) = {prod}  "
          f"{'OK' if prod == 0.0 else 'FAIL'}")
    print("    so a zero row breaks DOMAIN COVERAGE without, by itself, breaking")
    print("    the output -- counted separately from a tiny POSITIVE denominator,")
    print("    which does corrupt a nonzero numerator.")

    print(f"\n  {'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="verify the numerical core; needs no checkpoint or data")
    ap.add_argument("--checkpoint", help="trained SUBSTITUTED student (.pt)")
    ap.add_argument("--variant", choices=("relu", "clamp"), default="relu",
                    help="which Fix 2 operator the checkpoint trained: "
                         "relu = polyattn_normed_relu_v1 (fix2_bloodmnist), "
                         "clamp = polyattn_normed_clamp_v1 (verify_fixes)")
    ap.add_argument("--iters", type=int, default=8,
                    help="Goldschmidt iterations; CKKS depth = 2*iters + 1")
    ap.add_argument("--relu-bits", type=int, default=None,
                    help="LUT bits for the non-negativity step. OMITTING THIS "
                         "leaves an exact comparison in the circuit, which is "
                         "not deployable -- the run is then labelled partial.")
    ap.add_argument("--relu-range", type=float, nargs=2, default=(-10.0, 10.0))
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="relative-error budget the reciprocal must meet for a "
                         "row to count as usable. The (0,2) interval is also "
                         "reported, but it is necessary-only and must not be "
                         "used as the deployment criterion.")
    ap.add_argument("--boundary", default="attention only; final softmax external",
                    help="declared approximation boundary, recorded in the output")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.self_test:
        raise SystemExit(self_test())
    if not a.checkpoint:
        raise SystemExit("need --checkpoint, or --self-test")

    from fix2_bloodmnist import load_checkpoint, get_bloodmnist_loaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, obj = load_checkpoint(a.checkpoint, device)
    train_loader, val_loader, test_loader = get_bloodmnist_loaders(
        batch_size=a.batch_size, data_root=a.data_root)

    scale, max_rowsum = calibrate_scale(model, train_loader, device)
    print(f"calibrated on TRAIN: max row-sum {max_rowsum:.4f}  "
          f"scale {scale:.6e}  domain d < {2/scale:.2f}")

    def run(exact_recip):
        mods = swap_attention(model, a.variant, scale, a.iters, a.relu_bits,
                              tuple(a.relu_range), exact_recip, a.tol)
        for m in mods:
            m.reset_stats()
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for imgs, y in test_loader:
                logits.append(model(imgs.to(device)).cpu())
                labels.append(y.squeeze(-1).long())
        return torch.cat(logits), torch.cat(labels), mods

    # S: same swap, exact division -- so the ONLY difference from P is the
    # approximated reciprocal (and the LUT, if enabled). Rebuilding the model
    # for S would reintroduce a reimplementation difference.
    s_logits, labels, _ = run(exact_recip=True)
    p_logits, _, mods = run(exact_recip=False)

    s_pred, p_pred = s_logits.argmax(-1), p_logits.argmax(-1)
    flips = (s_pred != p_pred)
    err = (p_logits - s_logits).abs()

    stats = {k: sum(m.stats[k] for m in mods)
             for k in ("rows", "rows_exact_zero_denom",
                       "rows_tiny_positive_denom", "rows_out_of_domain",
                       "rows_outside_0_2_window")}
    stats["denom_min"] = min(m.stats["denom_min"] for m in mods)
    stats["denom_max"] = max(m.stats["denom_max"] for m in mods)

    out = {
        "checkpoint": a.checkpoint,
        "operator_variant": a.variant,
        "operator_id": f"polyattn_normed_{a.variant}_v1",
        "declared_boundary": a.boundary,
        "exact_comparison_left_in_circuit": a.relu_bits is None,
        "goldschmidt": {
            "iters": a.iters, "ckks_depth": 2 * a.iters + 1,
            "scale": scale, "calibration_max_rowsum": max_rowsum,
            "convergence_window_upper_bound": 2 / scale,
            "tol": a.tol,
            "min_usable_denominator":
                goldschmidt_min_usable_d(scale, a.iters, a.tol),
            "saturation_value_as_d_to_0":
                goldschmidt_saturation(scale, a.iters),
            "note": ("A denominator below min_usable_denominator does not blow "
                     "up: the reciprocal returns saturation_value_as_d_to_0, a "
                     "constant independent of d."),
        },
        "relu_lut": (None if a.relu_bits is None else
                     {"bits": a.relu_bits, "range": list(a.relu_range),
                      "clamp_rate": max(m.relu_lut.clamp_rate for m in mods)}),
        "accuracy": {
            "S_exact_recip": float((s_pred == labels).float().mean()),
            "P_approx_recip": float((p_pred == labels).float().mean()),
        },
        "agreement": {
            "prediction_agreement": float((~flips).float().mean()),
            "n_flips": int(flips.sum()), "n_images": int(len(labels)),
        },
        "logit_error": {"max": float(err.max()), "mean": float(err.mean())},
        "domain": stats,
    }
    if a.relu_bits is None:
        out["_WARNING"] = ("PARTIAL P-MODE: --relu-bits was not given, so the "
                           "non-negativity comparison was evaluated exactly. A "
                           "comparison is not free under CKKS; this run does "
                           "not measure a deployable circuit.")
    print(json.dumps(out, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, a.out)
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
