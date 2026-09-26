#!/usr/bin/env python3
"""CIFAR-10 fixed-weight test conversion: paired accuracy change per seed, by mask and profile.

    python paper/revised/scripts/make_paired_conversion_figure.py

Each open marker is one seed (converted minus native test accuracy on the same checkpoint); filled
markers are means with sample SD. These are the per-seed values behind the CIFAR-10 rows of Table II.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import BLUE, DATA, GRAY, ORANGE, save, style

MASKS = [("gelu", "GELU"), ("norm", "Normalization"), ("attention", "Attention"), ("all", "All operations")]
PROFILES = [("budget", "Low cost", ORANGE, "o", -0.17), ("accurate", "Higher degree", BLUE, "s", 0.17)]
TABLE_II = {("budget", "gelu"): -61.21, ("budget", "attention"): -54.39, ("budget", "norm"): -0.34,
            ("budget", "all"): -76.06, ("accurate", "gelu"): -10.41, ("accurate", "attention"): -6.54,
            ("accurate", "norm"): -0.15, ("accurate", "all"): -16.73}


def main():
    style()
    df = pd.read_csv(DATA / "conversion_verified.csv")
    df = df[df["split"].eq("clean") & df["fit_n"].eq(2048)].copy()
    df["change"] = df["converted_accuracy"] - df["reference_accuracy"]
    fig, ax = plt.subplots(figsize=(7.05, 2.6))
    for profile, label, color, marker, offset in PROFILES:
        for i, (mask, _) in enumerate(MASKS):
            vals = df[df["profile"].eq(profile) & df["mask"].eq(mask)].sort_values("seed")["change"].to_numpy()
            assert len(vals) == 3 and abs(vals.mean() - TABLE_II[profile, mask]) < 0.005, (profile, mask)
            x = i + offset
            ax.scatter(x + np.linspace(-0.04, 0.04, 3), vals, s=17, facecolors="white", edgecolors=color,
                       marker="o", linewidths=0.8, zorder=2)
            ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt=marker, color=color, capsize=2.5,
                        markersize=4.5, linewidth=1.1, zorder=3)
        ax.plot([], [], marker=marker, color=color, linestyle="none", label=label)
    ax.axhline(0, color=GRAY, ls="--", lw=0.9)
    ax.set_xticks(range(len(MASKS)), [name for _, name in MASKS])
    ax.set_xlabel("Converted operations")
    ax.set_ylabel("Test-accuracy change (pp)")
    ax.set_ylim(-85, 5)
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.6)
    ax.legend(frameon=False, loc="lower center", ncol=2)
    save(fig, "paired_conversion")


if __name__ == "__main__":
    main()
