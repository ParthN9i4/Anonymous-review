#!/usr/bin/env python3
"""Approximation error (P-O) versus additional CKKS error (C-P) for the 63 finite primitive cases.

    python paper/revised/scripts/make_primitive_error_figure.py

Each point is one case (seed, profile, block, path, operation); both errors are the median of its two
measured repeats.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import BLUE, DATA, GRAY, GREEN, ORANGE, save, style

OPERATIONS = [("softmax_row", "Attention row"), ("gelu", "GELU"), ("ln_inverse_sqrt", "Inverse square root")]
CASE = ["configuration", "profile", "block", "path", "operation"]


def main():
    style()
    df = pd.read_csv(DATA / "ckks.csv")
    df = df[df["status"] == "MEASURED"]
    assert len(df) == 126, len(df)
    cases = df.groupby(CASE)[["approximation_max_abs", "ckks_max_abs"]].median().reset_index()
    assert len(cases) == 63 and (df.groupby(CASE).size() == 2).all()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    lo, hi = 1e-6, 1.0
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=0.9, ls="--", zorder=1)
    ax.text(2e-4, 5e-4, "equal errors", rotation=45, rotation_mode="anchor", color=GRAY, fontsize=6.5)
    for (op, title), color, marker in zip(OPERATIONS, (BLUE, ORANGE, GREEN), ("o", "s", "^")):
        rows = cases[cases["operation"] == op]
        ax.scatter(rows["approximation_max_abs"], rows["ckks_max_abs"], s=14, color=color, marker=marker,
                   label=f"{title} ($n={len(rows)}$)", zorder=3)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal")
    ax.set_xlabel("Approximation error $|P-O|$")
    ax.set_ylabel("CKKS error $|C-P|$")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    save(fig, "paired_primitive_errors")


if __name__ == "__main__":
    main()
