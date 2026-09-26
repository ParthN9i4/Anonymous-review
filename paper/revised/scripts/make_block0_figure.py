#!/usr/bin/env python3
"""Block-0 GELU: exact function, deployed Chebyshev-node fit, and activation-sample refit (seed 42).

    python paper/revised/scripts/make_block0_figure.py
"""
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from make_figures import BLUE, DATA, GRAY, ORANGE, save, style

SEED = "42"


def poly(p, x):
    return np.polynomial.polynomial.polyval((x - p["center"]) / p["radius"], p["coefficients"])


def main():
    style()
    closeout = DATA / "closeout_2026-09-24"
    deployed = json.loads((closeout / "deployed_block0_gelu.json").read_text())["polys"][SEED]
    run = json.loads((closeout / f"block0_s{SEED}.json").read_text())
    refit = run["empirical_polynomial"]
    assert (refit["lo"], refit["hi"]) == (deployed["lo"], deployed["hi"])
    q01, q99 = run["gate"]["input_quantiles"][1], run["gate"]["input_quantiles"][5]

    x = np.linspace(deployed["lo"], deployed["hi"], 4001)
    exact = 0.5 * x * (1 + np.array([math.erf(v / math.sqrt(2)) for v in x]))
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.3), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax in axes:
        ax.axvspan(q01, q99, color="#E5E7EB", zorder=0, lw=0)
    axes[0].plot(x, exact, color="black", lw=1.1, label="GELU")
    axes[0].plot(x, poly(deployed, x), color=ORANGE, lw=1.1, label="Chebyshev-node fit (deployed)")
    axes[0].plot(x, poly(refit, x), color=BLUE, lw=1.1, ls="--", label="Activation-sample fit")
    axes[0].set_ylim(-3, 17)
    axes[0].set_ylabel("Output")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    axes[1].semilogy(x, np.abs(poly(deployed, x) - exact), color=ORANGE, lw=1.1)
    axes[1].semilogy(x, np.abs(poly(refit, x) - exact), color=BLUE, lw=1.1, ls="--")
    axes[1].set_ylim(1e-3, 30)
    axes[1].set_ylabel("Absolute error")
    axes[1].set_xlabel("Block-0 GELU input (seed 42 fitted interval)")
    axes[1].text((q01 + q99) / 2, 12, "98% of inputs", ha="center", fontsize=6.5, color=GRAY)
    save(fig, "block0_gelu_fits")


if __name__ == "__main__":
    main()
