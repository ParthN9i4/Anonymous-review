"""Frozen plan for the closeout_v2 campaign. Change only by creating a new version.

Everything here is fixed BEFORE any v2 outcome is observed. `launch_v2.sh` hashes
this file and PREREGISTRATION_v2.md into the run manifest.

Splits (CIFAR-10, the project's fixed 45k/5k split from pvit_lab.data.datasets):
  fit    = clean training images [0, 2048)      -- polynomial fitting (same as v1)
  gate   = clean training images [2048, 4096)   -- pre-deployment screen inputs (same as v1)
  dev    = the 5,000-image validation split     -- all v2 sweeps and screen development
  test   = CIFAR-10 test, 10,000 images         -- opened only by `confirm` jobs, after the
                                                   screen is frozen (screen_frozen.json exists)
"""

SEEDS = [42, 43, 44]
DEV_SEED = 42                      # screen thresholds are chosen on this seed only
FIT_N = 2048
GATE_OFFSET = 2048
GATE_N = 2048
TEST_N = 10000

# v1 profiles, copied from source_final/polynomial.py (PROFILES) for reference.
V1 = {'budget': dict(gelu_degree=7, norm_degree=7, exp_root_degree=4, reciprocal_steps=2),
      'accurate': dict(gelu_degree=15, norm_degree=15, exp_root_degree=8, reciprocal_steps=4)}

# Deployment requirement used to label a configuration acceptable (fixed a priori).
ACCEPT_FLIP_RATE = 0.01            # <=1% prediction flips vs the reference on the split
SECONDARY_FLIP_RATE = 0.05         # also reported, never used to select thresholds
CP_ALPHA = 0.05                    # Clopper-Pearson one-sided 95% upper bound

ATTENTION_VARIANTS = ['poly', 'exact_numerator', 'exact_reciprocal', 'exact_both']
GELU_DEGREES = [7, 9, 11, 15, 23, 31]
GELU_DOMAINS = ['minmax', 'central']          # central = [p0.1, p99.9] of fit activations
ROOT_DEGREES = [4, 8, 12]
RECIPROCAL_STEPS = [2, 4, 6, 8, 10]
BLOCKS = list(range(6))
CENTRAL_QUANTILES = (0.001, 0.999)
SAMPLE_PER_SITE = 200_000                     # activation sample kept per site for quantiles


def _job(kind, seed, split, **kw):
    name = '_'.join([kind, f's{seed}'] + [f'{k}-{v}' for k, v in kw.items()] + [split])
    return dict(name=name, kind=kind, seed=seed, split=split, **kw)


def e1_jobs():
    """Oracle attribution: which part of polynomial attention loses accuracy?"""
    out = []
    for s in SEEDS:
        for profile in ('budget', 'accurate'):
            for mask in ('attention', 'all'):
                for variant in ATTENTION_VARIANTS:
                    out.append(_job('e1', s, 'dev', profile=profile, mask=mask, variant=variant))
    return out


def e2_configs():
    """Dose-response configurations (also the confirm-pass configurations)."""
    cfg = []
    for d in GELU_DEGREES:
        for dom in GELU_DOMAINS:
            cfg.append(dict(profile='budget', mask='gelu', gelu_degree=d, domain=dom))
    for r in ROOT_DEGREES:
        for k in RECIPROCAL_STEPS:
            cfg.append(dict(profile='budget', mask='attention', exp_root_degree=r, reciprocal_steps=k))
    return cfg


def e2_jobs(split='dev'):
    return [_job('e2' if split == 'dev' else 'confirm', s, split, **c) for s in SEEDS for c in e2_configs()]


def e3_jobs():
    """Layer localization: convert one block only, or all blocks but one."""
    out = []
    for s in SEEDS:
        for mask in ('gelu', 'attention'):
            for b in BLOCKS:
                out.append(_job('e3', s, 'dev', profile='budget', mask=mask, only_block=b))
                out.append(_job('e3', s, 'dev', profile='budget', mask=mask, except_block=b))
    return out


# MedMNIST extension (E1 and E2 only). Checkpoints: arm 000 of the existing
# medical_replication_v1 run; fit/gate indices and float64 evaluation exactly as in
# archive/submission_v2/scripts/medical_conversion.py; dev = the official validation split.
MEDICAL = ['bloodmnist', 'pathmnist', 'dermamnist']
MEDICAL_FIT_RNG = 20260921


def med_e1_jobs():
    return [dict(j, name=f"med_{d}_{j['name']}", dataset=d, dtype='float64')
            for d in MEDICAL for j in e1_jobs()]


def med_e2_jobs():
    return [dict(j, name=f"med_{d}_{j['name']}", dataset=d, dtype='float64')
            for d in MEDICAL for j in e2_jobs('dev')]


def pilot_jobs():
    """Real-checkpoint correctness gate, run before any array (one job per dataset)."""
    return [dict(name=f'pilot_{d}', kind='pilot', seed=42, split='dev', dataset=d,
                 dtype='float32' if d == 'cifar10' else 'float64', images=64)
            for d in ['cifar10'] + MEDICAL]


# F1: continuously encrypted MLP sub-block on the headline failing configuration.
# ~13 s per token-chain on 4 CPU threads (measured locally, N=16384): 32 images x 65 tokens x 2 blocks
# = 4,160 chains, split into 32 jobs of 2 images (~30 min each).
F1 = dict(seed=43, profile='budget', mask='gelu', blocks=[0, 5], images=32, images_per_job=2,
          ring_degree=16384, modulus_bits=[60] + [40] * 7 + [60], scale_bits=40,
          smoke_max_seconds_per_token=60.0, reduced_images=8)

# F0: measured depth ledger.
F0 = dict(settings=[dict(ring_degree=16384, modulus_bits=[60] + [40] * 7 + [60], scale_bits=40),
                    dict(ring_degree=32768, modulus_bits=[60] + [40] * 18 + [60], scale_bits=40)])


def f1_jobs():
    n = F1['images'] // F1['images_per_job']
    return [dict(name=f"f1_b{b}_part{i}", kind='f1', block=b, part=i, first=i * F1['images_per_job'],
                 count=F1['images_per_job']) for b in F1['blocks'] for i in range(n)]


def schedule():
    return dict(pilot=pilot_jobs(), e1=e1_jobs(), e2=e2_jobs('dev'), e3=e3_jobs(), confirm=e2_jobs('test'),
                med_e1=med_e1_jobs(), med_e2=med_e2_jobs(),
                f0=[dict(name=f"f0_n{s['ring_degree']}", kind='f0', setting=i) for i, s in enumerate(F0['settings'])],
                f1=f1_jobs())
