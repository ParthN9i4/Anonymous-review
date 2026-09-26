#!/usr/bin/env python3
"""Build the main factorial figure from archived CIFAR and audited medical rows.

The left panel uses CIFAR-10 validation accuracies from the archived five-seed
factorial.  The right panel uses locked medical test accuracy changes paired
within dataset and seed to arm 000.  The distinct scales and split labels are
intentional: the KD CIFAR and CE-only medical campaigns are not pooled.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

MASK_ORDER = ["000", "100", "010", "001", "110", "111", "011", "101"]
CIFAR_ARMS = list("0ABCDEFG")
X_LABELS = MASK_ORDER
COLORS = {
    "bloodmnist": "#0072B2",
    "pathmnist": "#D55E00",
    "dermamnist": "#009E73",
}
NAMES = {
    "bloodmnist": "BloodMNIST",
    "pathmnist": "PathMNIST",
    "dermamnist": "DermaMNIST",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

cifar = pd.read_csv(DATA / "per_checkpoint_metrics.csv")
cifar = cifar[(cifar["family"] == "factorial") & cifar["arm"].isin(CIFAR_ARMS)]
assert len(cifar) == 40, len(cifar)

medical = pd.read_csv(DATA / "medical_factorial_test.csv", dtype={"arm": str})
medical = medical[medical["arm"].isin(MASK_ORDER)].copy()
assert len(medical) == 72, len(medical)
failed = medical[medical["status"] != "MEASURED"]
assert len(failed) == 1
assert failed.iloc[0][["dataset", "seed", "arm"]].tolist() == ["pathmnist", 42, "111"]

baseline = medical[medical["arm"] == "000"][["dataset", "seed", "accuracy_percent"]]
baseline = baseline.rename(columns={"accuracy_percent": "baseline_accuracy"})
medical = medical.merge(baseline, on=["dataset", "seed"], how="left", validate="many_to_one")
medical["change_pp"] = medical["accuracy_percent"] - medical["baseline_accuracy"]

fig, (left, right) = plt.subplots(1, 2, figsize=(7.05, 2.85), constrained_layout=True)
xs = np.arange(len(MASK_ORDER))

# CIFAR-10: individual validation seeds, mean, and sample SD.
for x, arm in zip(xs, CIFAR_ARMS):
    values = cifar.loc[cifar["arm"] == arm].sort_values("seed")["accuracy"].to_numpy()
    offsets = np.linspace(-0.09, 0.09, len(values))
    left.scatter(
        x + offsets,
        values,
        s=17,
        facecolors="white",
        edgecolors="#6B7280",
        linewidths=0.8,
        zorder=2,
    )
    left.errorbar(
        x,
        values.mean(),
        yerr=values.std(ddof=1),
        fmt="o",
        color="#007C83",
        capsize=2.3,
        markersize=4.5,
        linewidth=1.1,
        zorder=3,
    )
left.axhline(
    cifar.loc[cifar["arm"] == "0", "accuracy"].mean(),
    color="#D1D5DB",
    linewidth=0.9,
    linestyle="--",
)
left.set_title("(a) CIFAR-10: validation accuracy")
left.set_ylabel("Accuracy (%)")
left.set_ylim(0, 90)
left.grid(axis="y", color="#ECEFF1", linewidth=0.55)

# Medical: changes paired to each dataset/seed's native 000 control.
dataset_offsets = {"bloodmnist": -0.22, "pathmnist": 0.0, "dermamnist": 0.22}
for dataset in ("bloodmnist", "pathmnist", "dermamnist"):
    color = COLORS[dataset]
    for x, mask in zip(xs, MASK_ORDER):
        rows = medical[(medical["dataset"] == dataset) & (medical["arm"] == mask)]
        measured = rows[rows["status"] == "MEASURED"].sort_values("seed")
        values = measured["change_pp"].to_numpy()
        center = x + dataset_offsets[dataset]
        seed_offsets = np.linspace(-0.035, 0.035, len(values)) if len(values) > 1 else np.array([0.0])
        right.scatter(
            center + seed_offsets,
            values,
            s=14,
            facecolors="white",
            edgecolors=color,
            linewidths=0.75,
            zorder=2,
        )
        # Do not summarize the incomplete PathMNIST 111 cell as a three-seed arm.
        if len(measured) == 3:
            right.errorbar(
                center,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="o",
                color=color,
                capsize=2.0,
                markersize=3.8,
                linewidth=1.0,
                zorder=3,
            )
    right.plot([], [], marker="o", linestyle="none", color=color, label=NAMES[dataset])

# Plot-floor marker records the failed training cell without assigning an accuracy.
failed_x = MASK_ORDER.index("111") + dataset_offsets["pathmnist"]
right.plot(
    failed_x,
    0.035,
    marker="x",
    color="#C83E4D",
    markersize=6,
    markeredgewidth=1.3,
    transform=right.get_xaxis_transform(),
    clip_on=False,
)
right.annotate(
    "seed 42 failed",
    xy=(failed_x, 0.035),
    xycoords=right.get_xaxis_transform(),
    xytext=(0, 7),
    textcoords="offset points",
    ha="center",
    va="bottom",
    color="#C83E4D",
    fontsize=6.5,
)

right.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--")
right.set_title("(b) MedMNIST: test-accuracy change")
right.set_ylabel("Change from arm 000 (pp)")
right.set_ylim(-82, 6)
right.grid(axis="y", color="#ECEFF1", linewidth=0.55)
right.legend(frameon=False, loc="lower left")

for ax in (left, right):
    ax.set_xticks(xs, X_LABELS)
    ax.set_xlabel("Arm (GELU, attention, normalization)")
    ax.set_xlim(-0.55, len(xs) - 0.45)

fig.savefig(OUT / "factorial_cross_dataset.pdf", bbox_inches="tight")
fig.savefig(OUT / "factorial_cross_dataset.png", dpi=240, bbox_inches="tight")
plt.close(fig)

print(OUT / "factorial_cross_dataset.pdf")
