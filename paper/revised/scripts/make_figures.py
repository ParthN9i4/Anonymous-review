#!/usr/bin/env python3
"""Create paper figures from the archived, recomputed evidence tables."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"
OUT = PACKAGE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#007C83"
ORANGE = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
PURPLE = "#7A5195"
GRAY = "#6B7280"
LIGHT = "#D1D5DB"
RED = "#C83E4D"


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def factorial_plot():
    df = pd.read_csv(DATA / "per_checkpoint_metrics.csv")
    df = df[df.family.eq("factorial") & ~df.arm.eq("teacher")].copy()
    order = list("0ABCDEFG")
    labels = ["000", "100", "010", "001", "110", "111", "011", "101"]
    fig, ax = plt.subplots(figsize=(7.05, 2.55))
    rng = np.random.default_rng(17)
    for i, arm in enumerate(order):
        vals = df.loc[df.arm.eq(arm), "accuracy"].to_numpy()
        jitter = rng.uniform(-0.10, 0.10, len(vals))
        ax.scatter(i + jitter, vals, s=18, facecolors="white", edgecolors=GRAY, linewidths=0.8, zorder=2)
        ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color=TEAL, ecolor=TEAL,
                    capsize=2.5, markersize=4.5, linewidth=1.2, zorder=3)
    ax.axhline(df.loc[df.arm.eq("0"), "accuracy"].mean(), color=LIGHT, lw=1, ls="--", zorder=0)
    ax.set_xticks(range(len(order)), [f"{a}\n{b}" for a, b in zip(order, labels)])
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xlabel("Arm and substitution mask (GELU, attention, normalization)")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.6)
    ax.text(0.01, 0.04, "Open circles: seeds 42–46   Filled circles: mean   Bars: sample SD",
            transform=ax.transAxes, color=GRAY, fontsize=7)
    save(fig, "factorial_seed_results")


def recipe_plot():
    df = pd.read_csv(DATA / "per_checkpoint_metrics.csv")
    df = df[df.family.eq("baseline")].copy()
    fig, ax = plt.subplots(figsize=(3.42, 2.55))
    xpos = {"legacy": 0, "regularized": 1}
    colors = {"ln": BLUE, "bn": ORANGE}
    markers = {"ln": "o", "bn": "s"}
    for norm in ["ln", "bn"]:
        means = []
        sds = []
        for recipe in ["legacy", "regularized"]:
            arm = f"{recipe}_{norm}"
            vals = df.loc[df.arm.eq(arm), "accuracy"].to_numpy()
            means.append(vals.mean())
            sds.append(vals.std(ddof=1))
            offsets = np.linspace(-0.035, 0.035, len(vals))
            ax.scatter(np.full_like(vals, xpos[recipe], dtype=float) + offsets, vals,
                       s=16, facecolors="white", edgecolors=colors[norm], linewidths=0.8, zorder=3)
        ax.errorbar([0, 1], means, yerr=sds, color=colors[norm], marker=markers[norm],
                    capsize=2.5, markersize=4.5, linewidth=1.2,
                    label={"ln": "LayerNorm", "bn": "BatchNorm"}[norm])
    ax.set_xticks([0, 1], ["Base recipe", "Stronger recipe"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(74, 89)
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.08, 79.1, "gap\n4.61 pp", ha="right", va="center", color=GRAY, fontsize=7)
    ax.text(1.08, 87.15, "gap\n0.85 pp", ha="left", va="center", color=GRAY, fontsize=7)
    save(fig, "recipe_normalization_interaction")


def conversion_plot():
    df = pd.read_csv(DATA / "conversion_verified.csv")
    df = df[df.split.eq("clean") & df.fit_n.eq(2048)].copy()
    df = df[df["mask"].isin(["gelu", "norm", "attention", "all"])]
    mask_order = ["gelu", "norm", "attention", "all"]
    labels = ["GELU", "LayerNorm", "Attention", "All"]
    fig, ax = plt.subplots(figsize=(7.05, 2.75))
    offsets = {"budget": -0.17, "accurate": 0.17}
    styles = {"budget": (ORANGE, "o", "Budget"), "accurate": (BLUE, "s", "Higher degree")}
    for profile, (color, marker, label) in styles.items():
        for i, mask in enumerate(mask_order):
            vals = df[(df.profile.eq(profile)) & (df["mask"].eq(mask))]["converted_accuracy"].to_numpy()
            x = i + offsets[profile]
            jitter = np.linspace(-0.035, 0.035, len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=17, facecolors="white",
                       edgecolors=color, marker=marker, linewidths=0.8, zorder=2)
            ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt=marker, color=color,
                        capsize=2.5, markersize=4.5, linewidth=1.1, zorder=3)
        ax.plot([], [], marker=marker, color=color, linestyle="none", label=label)
    ref = df.groupby("seed").reference_accuracy.first().mean()
    ax.axhline(ref, color=GRAY, ls="--", lw=1, label=f"Reference mean ({ref:.2f}%)")
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Clean-test accuracy (%)")
    ax.set_xlabel("Frozen operation mask")
    ax.set_ylim(0, 92)
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    save(fig, "frozen_conversion_accuracy")


def detector_plot():
    df = pd.read_csv(DATA / "conversion_verified.csv")
    df = df[df.split.eq("clean")].drop_duplicates("configuration").copy()
    df = df.sort_values(["profile", "mask", "seed", "fit_n"]).reset_index(drop=True)
    colors = df.profile.map({"budget": ORANGE, "accurate": BLUE}).to_numpy()
    markers = df["mask"].map({"gelu": "o", "norm": "s", "attention": "^", "all": "D"}).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), gridspec_kw={"wspace": 0.33})
    for i, row in df.iterrows():
        axes[0].scatter(i + 1, 100 * row.calibration_failure_rate, color=colors[i], marker=markers[i], s=18)
        axes[1].scatter(i + 1, row.precision_extra_true_positive, color=colors[i], marker=markers[i], s=18)
    axes[0].axhline(1, color=RED, lw=1, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Calibration failures (%)")
    axes[0].set_title("Calibration failures per configuration")
    axes[0].text(26, 1.25, "1% threshold", ha="right", va="bottom", color=RED, fontsize=7)
    axes[1].set_ylabel("Extra failures found by FP64")
    axes[1].set_title("Extra failures found by the FP64 check")
    axes[1].set_ylim(-0.25, max(4.5, df.precision_extra_true_positive.max() + 0.5))
    for ax in axes:
        ax.set_xlabel("Conversion configuration")
        ax.set_xlim(0, len(df) + 1)
        ax.set_xticks([1, 5, 10, 15, 20, 26])
        ax.grid(axis="y", color="#ECEFF1", linewidth=0.6)
    handles = [
        plt.Line2D([], [], color=ORANGE, marker="s", markersize=6, linestyle="none", label="Low-cost profile"),
        plt.Line2D([], [], color=BLUE, marker="s", markersize=6, linestyle="none", label="Higher-degree profile"),
    ] + [plt.Line2D([], [], color=GRAY, marker=m, linestyle="none", label=f"Mask: {name}")
         for m, name in (("o", "GELU"), ("s", "normalization"), ("^", "attention"), ("D", "all operations"))]
    axes[1].legend(handles=handles, frameon=False, loc="upper right")
    save(fig, "screening_results")


def error_decomposition_plot():
    df = pd.read_csv(DATA / "ckks.csv")
    df = df[df.status.eq("MEASURED")].copy()
    op_order = ["softmax_row", "gelu", "ln_inverse_sqrt"]
    labels = ["Attention row", "GELU", "LN inverse sqrt"]
    approx = [df.loc[df.operation.eq(op), "approximation_max_abs"].max() for op in op_order]
    ckks = [df.loc[df.operation.eq(op), "ckks_max_abs"].max() for op in op_order]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(3.42, 2.65))
    width = 0.34
    ax.scatter(x - .08, approx, color=BLUE, marker="o", label="Approximation to target", zorder=3)
    ax.scatter(x + .08, ckks, color=ORANGE, marker="s", label="Additional CKKS error", zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Maximum absolute error")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.6, which="both")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), fontsize=7)
    save(fig, "approximation_vs_ckks_error")


def main():
    if not DATA.exists():
        sys.exit(f"Bundled evidence directory not found: {DATA}")
    style()
    factorial_plot()
    recipe_plot()
    conversion_plot()
    detector_plot()
    error_decomposition_plot()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
