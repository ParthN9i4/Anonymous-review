"""Plot audited H200 validation-only checkpoint means and sample SDs.

Values were transcribed from the user's closeout_audit_after_20260924
cifar_validation_means.csv and medical_validation_means.csv terminal output.
Reconcile against those CSVs before submission; no test result enters this plot.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cross_dataset_validation_summary.csv"
OUT = ROOT / "figures" / "cross_dataset_validation.pdf"
DATASETS = ["CIFAR-10", "BloodMNIST", "PathMNIST", "DermaMNIST"]
PANELS = [
    ("GELU-only conversion", "GELU", ["budget original", "degree 31"]),
    ("Attention-only conversion", "Attention", ["budget poly/poly", "higher-degree poly/poly"]),
]
COLORS = ["#D55E00", "#0072B2"]

with DATA.open(newline="") as stream:
    rows = {(row["dataset"], row["suite"], row["variant"]): row
            for row in csv.DictReader(stream)}

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.4), sharey=True,
                         constrained_layout=True)
xs = np.arange(len(DATASETS))
for ax, (title, suite, variants) in zip(axes, PANELS):
    for i, variant in enumerate(variants):
        vals = [rows[(dataset, suite, variant)] for dataset in DATASETS]
        means = [float(row["mean_change_pp"]) for row in vals]
        sds = [float(row["sample_sd_pp"]) for row in vals]
        ax.errorbar(xs + (i - .5) * .18, means, yerr=sds, fmt="o",
                    markersize=5, linewidth=1.1, capsize=2.3, capthick=1,
                    color=COLORS[i], label=variant)
    ax.axhline(0, color="#5B6573", linewidth=.8, linestyle="--")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(["CIFAR-10", "Blood", "Path", "Derma"], rotation=18)
    ax.set_xlim(-.43, 3.43)
    ax.set_ylim(-101, 8)
    ax.grid(axis="y", linewidth=.35, alpha=.45)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
axes[0].set_ylabel("Paired accuracy change (percentage points)")
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(OUT)
