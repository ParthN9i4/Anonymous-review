#!/usr/bin/env python3
"""Generate medical-conversion and perturbation figures from exported reports."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


DATASETS = ["bloodmnist", "pathmnist", "dermamnist"]
DATASET_LABELS = ["BloodMNIST", "PathMNIST", "DermaMNIST"]
MASKS = ["gelu", "attention", "norm", "all"]
MASK_LABELS = ["GELU", "Attention", "Norm", "All"]
COLORS = {
    "gelu": "#0072B2",
    "attention": "#D55E00",
    "norm": "#009E73",
    "all": "#7A5195",
    "activation": "#0072B2",
    "normalization": "#009E73",
}


def read(path):
    return json.loads(Path(path).read_text())


def save(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def medical_heatmap(summary, output):
    rows = [row for row in read(summary)["rows"] if row.get("status") == "MEASURED"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["profile"], row["mask"])].append(
            float(row["paired_change_pp"]))
    columns = [(profile, mask) for profile in ("accurate", "budget") for mask in MASKS]
    values = np.full((len(DATASETS), len(columns)), np.nan)
    for i, dataset in enumerate(DATASETS):
        for j, key in enumerate(columns):
            found = groups.get((dataset, key[0], key[1]), [])
            if found:
                values[i, j] = np.mean(found)

    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    image = ax.imshow(values, cmap="RdBu",
                      norm=TwoSlopeNorm(vmin=-100, vcenter=0, vmax=10),
                      aspect="auto")
    ax.set_yticks(range(len(DATASETS)), DATASET_LABELS)
    ax.set_xticks(
        range(len(columns)),
        [f"{'High' if profile == 'accurate' else 'Low'}\n"
         f"{MASK_LABELS[MASKS.index(mask)]}" for profile, mask in columns],
    )
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center",
                        fontsize=8, color="white" if values[i, j] < -45 else "black")
    ax.set_title("Paired accuracy change after frozen conversion (percentage points)")
    bar = fig.colorbar(image, ax=ax, pad=0.015)
    bar.set_label("Converted minus native accuracy (pp)")
    fig.tight_layout()
    save(fig, output)


def noise_curves(summary, output):
    records = read(summary)["noise_onsets"]
    grouped = defaultdict(list)
    for record in records:
        job = record["job"]
        for point in record["curve"]:
            grouped[(job["dataset"], job["site"], float(point["epsilon"]))].append(
                float(point["mean_accuracy_drop_pp"]))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
    for ax, dataset, label in zip(axes, DATASETS, DATASET_LABELS):
        for site in ("activation", "attention", "normalization"):
            points = sorted(
                (epsilon, np.mean(values))
                for (name, candidate, epsilon), values in grouped.items()
                if name == dataset and candidate == site
            )
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], marker="o",
                        linewidth=1.6, markersize=3.5, color=COLORS[site],
                        label=site.capitalize())
        ax.axhline(10, color="#5B6573", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(label)
        ax.set_xlabel("Relative output-noise RMS")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean accuracy drop (pp)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Validation sensitivity to operator-output perturbations", y=1.02)
    fig.tight_layout()
    save(fig, output)


def interpolation_curves(summary, output, profile="budget"):
    records = read(summary)["interpolation_onsets"]
    grouped = defaultdict(list)
    for record in records:
        job = record["job"]
        if job["profile"] != profile:
            continue
        for point in record["curve"]:
            grouped[(job["dataset"], job["mask"], float(point["lambda"]))].append(
                -float(point["accuracy_change_pp"]))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
    for ax, dataset, label in zip(axes, DATASETS, DATASET_LABELS):
        for mask, mask_label in zip(MASKS, MASK_LABELS):
            points = sorted(
                (weight, np.mean(values))
                for (name, candidate, weight), values in grouped.items()
                if name == dataset and candidate == mask
            )
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], marker="o",
                        linewidth=1.6, markersize=3.5, color=COLORS[mask],
                        label=mask_label)
        ax.axhline(10, color="#5B6573", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Conversion interpolation weight")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean accuracy drop (pp)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Validation path from native to {profile} polynomial operators", y=1.02)
    fig.tight_layout()
    save(fig, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medical-summary")
    parser.add_argument("--perturbation-summary")
    parser.add_argument("--out", default="paper/figures")
    args = parser.parse_args()
    out = Path(args.out)
    generated = []
    if args.medical_summary:
        medical_heatmap(args.medical_summary, out / "medical_conversion_heatmap")
        generated.append("medical_conversion_heatmap")
    if args.perturbation_summary:
        noise_curves(args.perturbation_summary, out / "perturbation_noise_curves")
        interpolation_curves(
            args.perturbation_summary, out / "conversion_interpolation_budget", "budget")
        generated += ["perturbation_noise_curves", "conversion_interpolation_budget"]
    if not generated:
        raise ValueError("Provide at least one summary")
    print(json.dumps({"generated": generated, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
