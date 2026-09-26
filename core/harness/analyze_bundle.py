from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp


parser = argparse.ArgumentParser(description="Recompute descriptive tables and figures from a metrics-closeout archive. No GPU or checkpoint files required.")
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "full")
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "paper_analysis")
args = parser.parse_args()
ROOT = args.root
RESULTS = ROOT / "results"
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["0", "A", "B", "C", "D", "E", "F", "G"]
ARM_LABEL = {
    "0": "000 unchanged",
    "A": "100 GELU",
    "B": "010 attention",
    "C": "001 norm",
    "D": "110 GELU+attention",
    "E": "111 all three",
    "F": "011 attention+norm",
    "G": "101 GELU+norm",
}
ARM_COLORS = {
    "0": "#4C78A8",
    "A": "#72B7B2",
    "B": "#F58518",
    "C": "#54A24B",
    "D": "#ECA82C",
    "E": "#E45756",
    "F": "#B279A2",
    "G": "#59A14F",
}


def load_reports():
    reports = []
    for directory in sorted(RESULTS.iterdir()):
        path = directory / "metrics.json"
        if path.is_file():
            reports.append((directory, json.loads(path.read_text())))
    return reports


def identity(name: str, kind: str):
    seed = int(name.rsplit("seed", 1)[1])
    if kind == "legacy":
        arm = name[len("legacy_") :].rsplit("_seed", 1)[0]
        family = "factorial"
    elif kind == "power":
        arm = name[len("power_") :].rsplit("_seed", 1)[0]
        family = "power"
    else:
        arm = name.rsplit("_seed", 1)[0]
        family = "baseline"
    return family, arm, seed


def finite_numbers(census, suffix, field):
    values = []
    for key, record in census.items():
        value = record.get(field)
        if key.endswith(suffix) and isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return values


records = []
tail_rows = []
denominator_rows = []
for directory, report in load_reports():
    name = report["name"]
    family, arm, seed = identity(name, report["kind"])
    validation = report["validation"]
    census = json.loads((directory / "validation_census.json").read_text())
    prediction = np.load(directory / "validation_predictions.npz")
    logits = prediction["logits"].astype(np.float64)
    labels = prediction["labels"].astype(int)
    finite = np.isfinite(logits).all(1)
    finite_logits = logits[finite]
    finite_labels = labels[finite]
    shifted_logits = finite_logits - finite_logits.max(axis=1, keepdims=True)
    losses = logsumexp(shifted_logits, axis=1) - shifted_logits[
        np.arange(len(finite_labels)), finite_labels
    ]

    row_means = [v["mean"] for k, v in census.items()
                 if re.fullmatch(r"blocks\.\d+\.abs_row_mass", k)]
    attention_relative = finite_numbers(
        census, ".attention_branch.relative_to_block_input", "rms"
    )
    mlp_relative = finite_numbers(census, ".mlp_branch.relative_to_block_input", "rms")
    residual_rms = finite_numbers(census, ".residual_out", "rms")
    score_maximum = finite_numbers(census, ".scores", "maximum") + [
        -v for v in finite_numbers(census, ".scores", "minimum")
    ]
    denominator_minimum = finite_numbers(census, ".denominator", "minimum")
    denominator_maximum = finite_numbers(census, ".denominator", "maximum")

    negative_attention = sum(
        v["negative_finite"]
        for k, v in census.items()
        if re.fullmatch(r"blocks\.\d+\.attention", k)
    )
    finite_attention = sum(
        v["finite"]
        for k, v in census.items()
        if re.fullmatch(r"blocks\.\d+\.attention", k)
    )

    records.append(
        {
            "name": name,
            "family": family,
            "arm": arm,
            "seed": seed,
            "accuracy": validation["accuracy_invalid_as_failure"],
            "balanced_accuracy": validation["balanced_accuracy_present_classes"],
            "macro_f1": validation["macro_f1_present_classes"],
            "invalid_images": validation["nonfinite_images"],
            "finite_coverage": validation["finite_coverage"],
            "conditional_nll": validation["nll_finite"],
            "conditional_ece": validation["ece15_finite"],
            "conditional_brier": validation["brier_finite"],
            "negative_attention_fraction": negative_attention / finite_attention
            if finite_attention
            else np.nan,
            "maximum_block_mean_abs_row_mass": max(row_means),
            "maximum_attention_branch_relative_rms": max(attention_relative),
            "maximum_mlp_branch_relative_rms": max(mlp_relative),
            "maximum_residual_rms": max(residual_rms),
            "maximum_abs_score": max(score_maximum),
            "maximum_abs_finite_logit": float(np.abs(finite_logits).max()),
            "batch1_p50_ms": 1000 * report["plaintext_profiles"][0]["p50_seconds"],
            "batch128_p50_ms": 1000 * report["plaintext_profiles"][1]["p50_seconds"],
            "batch128_peak_allocated_mib": report["plaintext_profiles"][1][
                "cuda_peak_allocated_bytes"
            ]
            / 2**20,
            "batch128_peak_reserved_mib": report["plaintext_profiles"][1][
                "cuda_peak_reserved_bytes"
            ]
            / 2**20,
            "parameter_count": report["parameter_count"],
        }
    )
    tail_rows.append(
        {
            "name": name,
            "family": family,
            "arm": arm,
            "seed": seed,
            "finite_images": int(finite.sum()),
            "invalid_images": int((~finite).sum()),
            "nll_median": float(np.median(losses)),
            "nll_p95": float(np.quantile(losses, 0.95)),
            "nll_p99": float(np.quantile(losses, 0.99)),
            "nll_maximum": float(losses.max()),
            "images_nll_gt_100": int((losses > 100).sum()),
            "images_nll_gt_1e6": int((losses > 1e6).sum()),
        }
    )
    if denominator_minimum:
        for layer in range(6):
            key = f"blocks.{layer}.denominator"
            row = census[key]
            denominator_rows.append(
                {
                    "name": name,
                    "arm": arm,
                    "seed": seed,
                    "layer": layer,
                    "minimum": row["minimum"],
                    "maximum": row["maximum"],
                    "q99_estimate": row["quantiles_estimated"]["q99"],
                }
            )

metrics = pd.DataFrame(records).sort_values(["family", "arm", "seed"])
tails = pd.DataFrame(tail_rows).sort_values(["family", "arm", "seed"])
denominators = pd.DataFrame(denominator_rows).sort_values(["arm", "seed", "layer"])
metrics.to_csv(OUT / "per_checkpoint_metrics.csv", index=False)
tails.to_csv(OUT / "per_checkpoint_tail_risk.csv", index=False)
denominators.to_csv(OUT / "power_denominator_ranges.csv", index=False)

factorial = metrics[(metrics.family == "factorial") & metrics.arm.isin(ARMS)].copy()
factorial_summary = (
    factorial.groupby("arm")
    .agg(
        n=("seed", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"),
        invalid_images_total=("invalid_images", "sum"),
        invalid_images_mean=("invalid_images", "mean"),
        conditional_nll_median_across_seeds=("conditional_nll", "median"),
        conditional_ece_mean=("conditional_ece", "mean"),
        conditional_brier_mean=("conditional_brier", "mean"),
        negative_attention_fraction_mean=("negative_attention_fraction", "mean"),
        maximum_block_mean_abs_row_mass_median=(
            "maximum_block_mean_abs_row_mass",
            "median",
        ),
        maximum_residual_rms_median=("maximum_residual_rms", "median"),
    )
    .reindex(ARMS)
    .reset_index()
)
factorial_summary.insert(1, "configuration", factorial_summary.arm.map(ARM_LABEL))
factorial_summary.to_csv(OUT / "factorial_summary.csv", index=False)

mapping = {
    "0": (0, 0, 0),
    "A": (1, 0, 0),
    "B": (0, 1, 0),
    "C": (0, 0, 1),
    "D": (1, 1, 0),
    "G": (1, 0, 1),
    "F": (0, 1, 1),
    "E": (1, 1, 1),
}
term_functions = {
    "GELU": lambda g, a, n: 1 if g else -1,
    "Attention": lambda g, a, n: 1 if a else -1,
    "Normalization": lambda g, a, n: 1 if n else -1,
    "GELU x Attention": lambda g, a, n: 1 if g == a else -1,
    "GELU x Normalization": lambda g, a, n: 1 if g == n else -1,
    "Attention x Normalization": lambda g, a, n: 1 if a == n else -1,
    "GELU x Attention x Normalization": lambda g, a, n: 1
    if (g + a + n) % 2 == 1
    else -1,
}
lookup = factorial.set_index(["seed", "arm"]).accuracy.to_dict()
effect_rows = []
for term, sign in term_functions.items():
    values = np.array(
        [
            sum(sign(*mapping[arm]) * lookup[(seed, arm)] for arm in mapping) / 4
            for seed in range(42, 47)
        ]
    )
    ci = stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=stats.sem(values))
    effect_rows.append(
        {
            "term": term,
            "mean_effect_pp": values.mean(),
            "sd": values.std(ddof=1),
            "ci95_low": ci[0],
            "ci95_high": ci[1],
            "one_sample_t_p": stats.ttest_1samp(values, 0).pvalue,
            **{f"seed_{seed}": value for seed, value in zip(range(42, 47), values)},
        }
    )
pd.DataFrame(effect_rows).to_csv(OUT / "factorial_effects.csv", index=False)

baseline = metrics[metrics.family == "baseline"]
baseline_summary = (
    baseline.groupby("arm")
    .agg(
        n=("seed", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        nll_mean=("conditional_nll", "mean"),
        ece_mean=("conditional_ece", "mean"),
        brier_mean=("conditional_brier", "mean"),
        batch1_p50_ms_mean=("batch1_p50_ms", "mean"),
        batch128_p50_ms_mean=("batch128_p50_ms", "mean"),
        peak_allocated_mib_mean=("batch128_peak_allocated_mib", "mean"),
    )
    .reset_index()
)
baseline_summary.to_csv(OUT / "baseline_summary.csv", index=False)

power = metrics[metrics.family == "power"]
power_summary = (
    power.groupby("arm")
    .agg(
        n=("seed", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        nll_mean=("conditional_nll", "mean"),
        invalid_images_total=("invalid_images", "sum"),
        batch1_p50_ms_mean=("batch1_p50_ms", "mean"),
        batch128_p50_ms_mean=("batch128_p50_ms", "mean"),
    )
    .reset_index()
)
power_summary.to_csv(OUT / "power_summary.csv", index=False)


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: factorial accuracy and interaction effects.
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
ax = axes[0]
for x, arm in enumerate(ARMS):
    values = factorial.loc[factorial.arm == arm, "accuracy"].to_numpy()
    jitter = np.linspace(-0.10, 0.10, len(values))
    ax.scatter(np.full(len(values), x) + jitter, values, color=ARM_COLORS[arm], s=25, alpha=0.85)
    ax.plot([x - 0.18, x + 0.18], [values.mean(), values.mean()], color="black", lw=1.4)
ax.axhline(10, color="#777777", ls="--", lw=0.8, label="Chance")
ax.set_xticks(range(len(ARMS)), ARMS)
ax.set_xlabel("Factorial arm")
ax.set_ylabel("Validation accuracy (%)")
ax.set_title("A. Composition creates seed-dependent failure")
ax.set_ylim(0, 90)

effects = pd.DataFrame(effect_rows)
order = [
    "GELU",
    "Attention",
    "Normalization",
    "GELU x Attention",
    "GELU x Normalization",
    "Attention x Normalization",
    "GELU x Attention x Normalization",
]
effects = effects.set_index("term").loc[order].reset_index()
y = np.arange(len(effects))
axes[1].errorbar(
    effects.mean_effect_pp,
    y,
    xerr=[effects.mean_effect_pp - effects.ci95_low, effects.ci95_high - effects.mean_effect_pp],
    fmt="o",
    color="#4C78A8",
    ecolor="#4C78A8",
    capsize=3,
)
axes[1].axvline(0, color="#555555", lw=0.8)
axes[1].set_yticks(y, [s.replace(" x ", " × ") for s in effects.term])
axes[1].invert_yaxis()
axes[1].set_xlabel("Factorial effect on validation accuracy (pp), 95% t interval")
axes[1].set_title("B. Large attention × normalization effect (5 seeds)")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure_factorial_interactions.{ext}", dpi=300)
plt.close(fig)

# Figure 2: mechanism plane. Means are annotated; faint points are seeds.
fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
for arm in ARMS:
    data = factorial[factorial.arm == arm]
    x = np.log10(data.maximum_block_mean_abs_row_mass)
    y = np.log10(data.maximum_residual_rms)
    ax.scatter(x, y, color=ARM_COLORS[arm], alpha=0.38, s=28)
    xm, ym = x.mean(), y.mean()
    ax.scatter([xm], [ym], color=ARM_COLORS[arm], edgecolor="black", linewidth=0.6, s=68)
    if arm not in {"0", "A", "C", "B", "D"}:
        ax.annotate(arm, (xm, ym), xytext=(4, 4), textcoords="offset points", weight="bold")
    elif arm == "C":
        ax.annotate("0, A, C", (xm, ym), xytext=(4.5, 5.0),
                    arrowprops={"arrowstyle": "-", "color": "#666666"})
    elif arm in {"B", "D"}:
        ax.annotate(arm, (xm, ym), xytext=(7.0, 1.0 if arm == "B" else 3.2),
                    arrowprops={"arrowstyle": "-", "color": "#666666"})
ax.set_xlabel("log10 maximum block mean absolute attention-row mass")
ax.set_ylabel("log10 maximum finite residual RMS")
ax.set_title("Attention-row mass and residual magnitude at evaluation")
ax.grid(alpha=0.2)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure_mechanism_plane.{ext}", dpi=300)
plt.close(fig)

# Figure 3: rare catastrophic tails hidden by accuracy and ECE.
factorial_tail = tails[(tails.family == "factorial") & tails.arm.isin(ARMS)]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
for x, arm in enumerate(ARMS):
    data = factorial_tail[factorial_tail.arm == arm]
    y = np.log10(data.nll_maximum.clip(lower=1e-6))
    axes[0].scatter(np.full(len(y), x) + np.linspace(-0.10, 0.10, len(y)), y,
                    color=ARM_COLORS[arm], s=25)
    axes[0].plot([x - 0.18, x + 0.18], [y.median(), y.median()], color="black", lw=1.4)
    inv = data.invalid_images.to_numpy()
    axes[1].scatter(np.full(len(inv), x) + np.linspace(-0.10, 0.10, len(inv)), inv,
                    color=ARM_COLORS[arm], s=25)
    axes[1].plot([x - 0.18, x + 0.18], [np.median(inv), np.median(inv)], color="black", lw=1.4)
axes[0].set_xticks(range(len(ARMS)), ARMS)
axes[1].set_xticks(range(len(ARMS)), ARMS)
axes[0].set_xlabel("Factorial arm")
axes[1].set_xlabel("Factorial arm")
axes[0].set_ylabel("log10 maximum finite per-image NLL")
axes[1].set_ylabel("Validation images with nonfinite logits")
axes[0].set_title("A. A few finite outputs have catastrophic loss")
axes[1].set_title("B. E and F are invalid in every seed")
axes[1].set_yscale("symlog", linthresh=1)
axes[1].set_ylim(-0.12, 140)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure_tail_reliability.{ext}", dpi=300)
plt.close(fig)

# Figure 4: denominator intervals for exact-division P/Q models.
fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
for i, row in denominators.reset_index(drop=True).iterrows():
    x = row.layer + (-0.12 if row.arm == "P" else 0.12)
    color = "#4C78A8" if row.arm == "P" else "#E45756"
    ax.plot([x, x], [row.minimum, row.maximum], color=color, alpha=0.35, lw=1)
    ax.scatter([x], [row.q99_estimate], color=color, s=12, alpha=0.7)
ax.axhline(65 * 0.01, color="black", ls="--", lw=0.9, label="Analytic lower bound: 65δ = 0.65")
ax.plot([], [], color="#4C78A8", marker="o", label="P: LayerNorm")
ax.plot([], [], color="#E45756", marker="o", label="Q: BatchNorm")
ax.set_yscale("log")
ax.set_xticks(range(6), [f"Block {i}" for i in range(6)])
ax.set_ylabel("Exact plaintext denominator")
ax.set_title("Positive floor bounds the denominator below, but not above")
ax.legend(frameon=False, fontsize=8)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure_power_denominator_ranges.{ext}", dpi=300)
plt.close(fig)

print(OUT)
