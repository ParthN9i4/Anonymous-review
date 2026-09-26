"""Per-site primitive error probes and model-level metrics for closeout_v2.

Two primitive views are measured at every converted site:
* source path: the deployed polynomial applied to the ORIGINAL model's inputs at that site
  (the distribution the fit targets; computable before deployment from gate data);
* converted path: the replacement module's actual output vs the exact operation on the
  inputs it actually receives inside the converted network (includes upstream drift).

Also recorded per site: the grid sup-error of the fit (the check papers usually report),
the fraction of inputs outside the fitted interval, and for attention the empirical
reciprocal relative error |1 - D R(D)| and the row-mass deviation |sum_j a_ij - 1|.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import polynomial as P

KEEP = 50_000  # per-site sample of absolute errors kept for quantiles


class SiteStats:
    def __init__(self):
        self.n = 0; self.sum = 0.0; self.max = 0.0; self.outside = 0; self.nonfinite = 0
        self.sample = []; self.kept = 0

    def add(self, err, outside=None):
        err = err.detach().double().flatten()
        fin = torch.isfinite(err)
        self.nonfinite += int((~fin).sum())
        e = err[fin]
        self.n += int(err.numel())
        if e.numel():
            self.sum += float(e.sum()); self.max = max(self.max, float(e.max()))
            if self.kept < KEEP:
                take = e[torch.randperm(e.numel())[:min(e.numel(), KEEP - self.kept)]].cpu().numpy()
                self.sample.append(take); self.kept += take.size
        if outside is not None:
            self.outside += int(outside.sum())

    def summary(self):
        s = np.concatenate(self.sample) if self.sample else np.array([])
        finite_n = self.n - self.nonfinite
        return dict(n=self.n, nonfinite=self.nonfinite,
                    mean_abs=self.sum / finite_n if finite_n else None,
                    max_abs=self.max if finite_n else None,
                    p99_abs=float(np.quantile(s, .99)) if s.size else None,
                    p999_abs=float(np.quantile(s, .999)) if s.size else None,
                    outside_fraction=self.outside / self.n if self.n else None)


def _gelu_exact(x):
    return F.gelu(x)  # erf form (PyTorch default approximate='none')


def _poly_softmax_parts(s, p):
    z = s - s.mean(-1, keepdim=True)
    u = P.torch_poly(z, p['root']).square() + p['floor']
    d = u.sum(-1, keepdim=True)
    return z, u, d


class Probes:
    """Registers hooks on the source model and the converted model."""

    def __init__(self, base, conv, params, converted_sites):
        self.params, self.sites = params, set(converted_sites)
        self.stats = {}; self.handles = []
        base_mods, conv_mods = dict(base.named_modules()), dict(conv.named_modules())
        for name in self.sites:
            kind = params[name]['kind']
            self.handles.append(base_mods[name].register_forward_pre_hook(
                lambda m, a, name=name, kind=kind: self._source(name, kind, m, a[0])))
            self.handles.append(conv_mods[name].register_forward_hook(
                lambda m, a, out, name=name, kind=kind: self._converted(name, kind, m, a[0], out)))

    def _st(self, key):
        return self.stats.setdefault(key, SiteStats())

    @torch.no_grad()
    def _source(self, name, kind, mod, x):
        p = self.params[name]
        if kind == 'gelu':
            q = p['poly']; out = (x < q['lo']) | (x > q['hi']) | ~torch.isfinite(x)
            self._st((name, 'source')).add((P.torch_poly(x, q) - _gelu_exact(x)).abs(), out)
        elif kind == 'norm':
            h = x - x.mean(-1, keepdim=True); v = h.square().mean(-1, keepdim=True) + mod.eps
            q = p['poly']; out = (v < q['lo']) | (v > q['hi'])
            self._st((name, 'source')).add((P.torch_poly(v, q) - v.rsqrt()).abs(), out)
        elif kind == 'attention':
            z, u, d = _poly_softmax_parts(x, p)
            out = (z < p['root']['lo']) | (z > p['root']['hi'])
            approx = u * P.reciprocal(d, p['reciprocal'])
            self._st((name, 'source')).add((approx - x.softmax(-1)).abs(), out)
            rec = p['reciprocal']; dout = (d < rec['lo']) | (d > rec['hi'])
            self._st((name, 'source_reciprocal_rel')).add((1 - d * P.reciprocal(d, rec)).abs(), dout)
            self._st((name, 'source_numerator_rel')).add(
                ((u - p['floor']) / torch.exp(z - p['public_exp_shift']) - 1).abs())

    @torch.no_grad()
    def _converted(self, name, kind, mod, x, y):
        p = self.params[name]
        if kind == 'gelu':
            q = p['poly']; out = (x < q['lo']) | (x > q['hi']) | ~torch.isfinite(x)
            self._st((name, 'converted')).add((y - _gelu_exact(x)).abs(), out)
        elif kind == 'norm':
            exact = F.layer_norm(x, x.shape[-1:], mod.weight, mod.bias, mod.eps)
            self._st((name, 'converted')).add((y - exact).abs())
        elif kind == 'attention':
            self._st((name, 'converted')).add((y - x.softmax(-1)).abs())
            self._st((name, 'converted_row_mass')).add((y.sum(-1) - 1).abs())
            z, u, d = _poly_softmax_parts(x, p)
            rec = p['reciprocal']; dout = (d < rec['lo']) | (d > rec['hi'])
            self._st((name, 'converted_reciprocal_rel')).add((1 - d * P.reciprocal(d, rec)).abs(), dout)

    def remove(self):
        for h in self.handles:
            h.remove()

    def summary(self):
        per = {}
        for (name, view), st in sorted(self.stats.items()):
            per.setdefault(name, {})[view] = st.summary()
        for name in per:
            p = self.params[name]
            if p['kind'] in ('gelu', 'norm'):
                per[name]['fit'] = dict(lo=p['poly']['lo'], hi=p['poly']['hi'], degree=p['poly']['degree'],
                                        grid_sup_error=p['poly']['grid_max_abs_error'])
            else:
                r = p['reciprocal']
                per[name]['fit'] = dict(root_lo=p['root']['lo'], root_hi=p['root']['hi'], root_degree=p['root']['degree'],
                                        root_grid_sup_error=p['root']['grid_max_abs_error'],
                                        reciprocal_lo=r['lo'], reciprocal_hi=r['hi'], reciprocal_steps=r['steps'],
                                        reciprocal_bound=r['relative_bound_on_interval'])
        return per


def aggregate_sites(per):
    """Network-level primitive summaries: worst site and mean over sites."""
    def collect(view, field):
        v = [s[view][field] for s in per.values() if view in s and s[view].get(field) is not None]
        return v
    out = {}
    for view in ('source', 'converted', 'source_reciprocal_rel', 'converted_reciprocal_rel',
                 'converted_row_mass', 'source_numerator_rel'):
        for field in ('mean_abs', 'p99_abs', 'max_abs', 'outside_fraction'):
            v = collect(view, field)
            if v:
                out[f'{view}_{field}_worst_site'] = float(max(v))
                out[f'{view}_{field}_site_mean'] = float(np.mean(v))
    sup = [s['fit'].get('grid_sup_error', s['fit'].get('root_grid_sup_error')) for s in per.values()]
    if sup:
        out['grid_sup_error_worst_site'] = float(max(sup))
    bnd = [s['fit']['reciprocal_bound'] for s in per.values() if 'reciprocal_bound' in s['fit']]
    if bnd:
        out['reciprocal_bound_worst_site'] = float(max(bnd))
    return out


# ---------------------------------------------------------------- model-level metrics
def _log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def clopper_pearson_upper(k, n, alpha=0.05):
    """One-sided (1-alpha) upper bound on a binomial rate; exact, dependency-free."""
    if n == 0:
        return None
    if k >= n:
        return 1.0

    def cdf(p):  # P(X <= k | n, p)
        if p <= 0:
            return 1.0
        if p >= 1:
            return 0.0
        lp, lq = math.log(p), math.log1p(-p)
        terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
                 for i in range(k + 1)]
        m = max(terms)
        return math.exp(m) * sum(math.exp(t - m) for t in terms)

    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def agreement(ref, conv, labels, alpha=0.05):
    """Prediction-level comparison of converted vs reference logits. Invalid = failure."""
    r = np.asarray(ref, dtype=np.float64); c = np.asarray(conv, dtype=np.float64); y = np.asarray(labels)
    rf = np.isfinite(r).all(1); cf = np.isfinite(c).all(1); both = rf & cf
    n = int(rf.sum())
    flip = np.zeros(len(r), bool); flip[both] = r[both].argmax(1) != c[both].argmax(1)
    changed = flip | (rf & ~cf)                      # invalid converted output counts as a change
    cw = np.zeros(len(r), bool); cw[both] = (r[both].argmax(1) == y[both]) & (c[both].argmax(1) != y[both])
    kl = np.full(len(r), np.nan)
    if both.any():
        lr, lc = _log_softmax(r[both]), _log_softmax(c[both])
        kl[both] = (np.exp(lr) * (lr - lc)).sum(1)
    wc = np.zeros(len(r), bool); wc[both] = (r[both].argmax(1) != y[both]) & (c[both].argmax(1) == y[both])
    tv = np.full(len(r), np.nan); cerr = np.full(len(r), np.nan); margin = np.full(len(r), np.nan)
    if both.any():
        tv[both] = .5 * np.abs(np.exp(_log_softmax(r[both])) - np.exp(_log_softmax(c[both]))).sum(1)
        d = c[both] - r[both]
        cerr[both] = np.abs(d - d.mean(1, keepdims=True)).max(1)        # invariant to a shared logit offset
        srt = np.sort(r[both], axis=1); margin[both] = srt[:, -1] - srt[:, -2]
    sufficient = both & (2 * np.nan_to_num(cerr, nan=np.inf) < np.nan_to_num(margin, nan=-np.inf))
    k = int(changed[rf].sum())
    return dict(n_reference_finite=n, converted_invalid=int((rf & ~cf).sum()),
                wrong_to_correct=int(wc.sum()),
                probability_tv_mean=float(np.nanmean(tv)) if both.any() else None,
                probability_tv_p99=float(np.nanquantile(tv, .99)) if both.any() else None,
                centered_logit_error_max=float(np.nanmax(cerr)) if both.any() else None,
                centered_logit_error_median=float(np.nanmedian(cerr)) if both.any() else None,
                margin_sufficient_count=int(sufficient.sum()),
                margin_sufficient_scope=('2*max centered logit error < reference top-2 margin guarantees an '
                                         'unchanged argmax for that example; failing it does not imply a flip'),
                prediction_changes=k, change_rate=k / n if n else None,
                change_rate_cp_upper=clopper_pearson_upper(k, n, alpha),
                correct_to_wrong=int(cw.sum()),
                kl_mean_finite=float(np.nanmean(kl)) if both.any() else None,
                kl_p99_finite=float(np.nanquantile(kl, .99)) if both.any() else None,
                reference_accuracy=100 * float((r.argmax(1) == y)[rf].mean()) if n else None,
                converted_accuracy_invalid_as_failure=100 * float(((c.argmax(1) == y) & cf).mean()))
