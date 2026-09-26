"""Generate generated_tables_*.tex from data/ so no table value is hand-typed.

    python paper/revised/scripts/make_tables.py

Caption convention for every table: first sentence states the finding; then the data,
split and units; then how to read special entries.
"""
import csv
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLOSEOUT = DATA / "closeout_2026-09-24"
REFIT = DATA / "closeout_2026-09-25"

# Largest-class count / test-split size (MedMNIST v2 test splits). A cell whose accuracy
# equals this share exactly is consistent with predicting that single class for every image.
MAJORITY = {"bloodmnist": (666, 3421), "dermamnist": (1341, 2005), "pathmnist": (1338, 7180)}
NAMES = {"bloodmnist": "BloodMNIST", "pathmnist": "PathMNIST", "dermamnist": "DermaMNIST"}
SEEDS = ["42", "43", "44"]


def rows(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def table(label, caption, spec, header, body, star=False, colsep="4pt"):
    env = "table*" if star else "table"
    return "\n".join([
        rf"\begin{{{env}}}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\setlength{{\tabcolsep}}{{{colsep}}}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header,
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"])


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def medical_interaction():
    acc = {(r["dataset"], r["seed"], r["arm"]): float(r["accuracy_percent"])
           for r in rows("medical_factorial_test.csv") if r["status"] == "MEASURED"}
    body = []
    for d in ["bloodmnist", "pathmnist", "dermamnist"]:
        effects = []
        for s in SEEDS:
            a = {k: acc.get((d, s, k)) for k in ["000", "100", "010", "001", "110", "101", "011", "111"]}
            if None in a.values():
                effects.append(None)
                continue
            effects.append((a["011"] - a["010"] - a["001"] + a["000"]
                            + a["111"] - a["110"] - a["101"] + a["100"]) / 4)
        count, n = MAJORITY[d]
        floor = 100 * count / n
        joint = [acc[(d, s, k)] for s in SEEDS for k in ["011", "111"] if (d, s, k) in acc]
        collapsed = sum(abs(v - floor) < 1e-9 for v in joint)
        cells = [f"${e:.2f}$" if e is not None else "failed" for e in effects]
        mean = f"${st.mean(effects):.2f}$" if None not in effects else "---"
        body.append(f"{NAMES[d]} & {' & '.join(cells)} & {mean} & {collapsed}/{len(joint)} \\\\")
    return table(
        "tab:medical-factorial",
        "Medical attention--normalization interaction effects (pp, Eq.~\\ref{eq:interaction}) by seed."
        " \\emph{Collapsed}: runs of arms 011 and 111 whose test accuracy equals the largest-class share,"
        " consistent with predicting one class for every image.",
        "@{}lrrrrr@{}", r"Dataset & Seed 42 & Seed 43 & Seed 44 & Mean & Collapsed \\", body)


def block0_refit():
    runs = [json.loads((CLOSEOUT / f"block0_s{s}.json").read_text()) for s in SEEDS]
    assert all(r["degree"] == 7 and r["gate_block0"]["order"] ==
               ["native", "block0_frozen_fit", "block0_empirical_fit"] for r in runs)
    assert all(r["gate"]["outside_frozen_domain_fraction"] == 0 for r in runs)
    assert all(sum(r["gate_block0"]["invalid_images"]) == 0 for r in runs)

    def line(name, values, fmt):
        cells = [fmt.format(v) for v in values] + [fmt.format(st.mean(values))]
        return f"{name} & " + " & ".join(cells) + r" \\"

    acc = lambda i: [r["gate_block0"]["accuracy_percent"][i] for r in runs]
    err = lambda fit, m: [r["gate"][fit][m] for r in runs]
    body = [
        r"\multicolumn{5}{@{}l}{\emph{Gate-image accuracy, only block 0 converted (\%)}} \\",
        line("Native model", acc(0), "{:.2f}"),
        line("Chebyshev-node fit", acc(1), "{:.2f}"),
        line("Activation-sample fit", acc(2), "{:.2f}"),
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Mean GELU error on gate activations}} \\",
        line("Chebyshev-node fit", err("frozen", "mean_abs"), "{:.3f}"),
        line("Activation-sample fit", err("empirical", "mean_abs"), "{:.3f}"),
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Maximum GELU error on gate activations}} \\",
        line("Chebyshev-node fit", err("frozen", "max_abs"), "{:.3f}"),
        line("Activation-sample fit", err("empirical", "max_abs"), "{:.3f}"),
    ]
    return table(
        "tab:block0-refit",
        "Earlier block-0 diagnostic: the deployed Chebyshev-node fit versus the activation-sample fit"
        " (degree 7, same interval), per seed.",
        "@{}lrrrr@{}", r" & Seed 42 & Seed 43 & Seed 44 & Mean \\", body, colsep="3.5pt")


def gelu_refit():
    runs = [json.loads((REFIT / f"block0_refit_eval_s{s}.json").read_text()) for s in SEEDS]
    assert all(r["fit_gate_overlap"] == 0 and r["degree"] == 7 for r in runs)
    recorded = {"42": 26.93, "43": 25.55, "44": 22.72}
    assert all(abs(r["baseline"]["frozen_test"]["accuracy_percent"] - recorded[s]) < 1e-9
               for s, r in zip(SEEDS, runs))
    assert all(v["invalid_images"] == 0 for r in runs for sec in ("baseline", "refit") for v in r[sec].values())

    def cell(section, key):
        v = [r[section][key]["accuracy_percent"] for r in runs]
        return f"${st.mean(v):.2f}\\pm{st.stdev(v):.2f}$"
    body = [
        f"Native model & {cell('baseline', 'reference_gate')} & {cell('baseline', 'reference_test')} \\\\",
        f"Deployed fits & {cell('baseline', 'frozen_gate')} & {cell('baseline', 'frozen_test')} \\\\",
        f"Block 0 refitted & {cell('refit', 'block0_gate')} & {cell('refit', 'block0_test')} \\\\",
        f"All six refitted & {cell('refit', 'all_blocks_gate')} & {cell('refit', 'all_blocks_test')} \\\\",
    ]
    return table(
        "tab:gelu_refit",
        "GELU-only conversion (low cost, degree 7) with deployed and refitted fits on the same intervals;"
        " accuracy (\\%), mean $\\pm$ SD over three seeds.",
        "@{}lrr@{}", r"Configuration & Held-out train & Test \\", body, colsep="4pt")


def degree31_probe():
    runs = [json.loads((CLOSEOUT / f"degree31_s{s}_block0.json").read_text()) for s in SEEDS]
    trials = [t for r in runs for t in r["trials"]]
    assert all(r["degree"] == 31 and r["parameters"]["scale_bits"] == 35 for r in runs)
    assert all(t["nonfinite"] == 0 for t in trials)
    levels = {t["cost"]["levels_consumed"] for t in trials}
    assert len(levels) == 1
    return dict(n=len(trials), po=max(t["approximation_max_abs"] for t in trials),
                cp=max(t["ckks_max_abs"] for t in trials), cp_min=min(t["ckks_max_abs"] for t in trials),
                levels=levels.pop(),
                time=st.median(t["cost"]["server_evaluation_seconds"] for t in trials),
                domains=[r["fit_domain"] for r in runs])


def gelu(x):
    return 0.5 * x * (1 + np.array([math.erf(v / math.sqrt(2)) for v in x]))


def power_basis_coefficients(lo, hi, degree):
    """Replicates fit_poly in experiments/active/closeout_2026-09-24/degree31_ckks_probe.py."""
    center, radius = (lo + hi) / 2, (hi - lo) / 2
    nodes = np.cos(np.pi * (np.arange(1024) + .5) / 1024)
    cheb = np.polynomial.chebyshev.chebfit(nodes, gelu(center + radius * nodes), degree)
    return cheb, np.polynomial.chebyshev.cheb2poly(cheb)


def coefficient_growth():
    lo, hi = degree31_probe()["domains"][0]
    out = {}
    for d in (7, 15, 31):
        cheb, power = power_basis_coefficients(lo, hi, d)
        out[d] = (float(np.abs(cheb).max()), float(np.abs(power).max()))
    return out


def opc_by_operation():
    ms = [r for r in rows("ckks.csv") if r["status"] == "MEASURED"]
    label = {"softmax_row": "Attn.\\ row", "gelu": "GELU", "ln_inverse_sqrt": "LN $x^{-1/2}$"}
    prof = {"budget": "Low", "accurate": "High"}
    body = []
    for op in ["softmax_row", "gelu", "ln_inverse_sqrt"]:
        for p in ["budget", "accurate"]:
            s = [r for r in ms if r["operation"] == op and r["profile"] == p]
            po = max(float(r["approximation_max_abs"]) for r in s)
            cp = max(float(r["ckks_max_abs"]) for r in s)
            t = st.median(float(r["server_evaluation_seconds"]) for r in s)
            lv = sorted({r["levels_consumed"] for r in s})
            assert len(lv) == 1, (op, p, lv)
            body.append(f"{label[op]} & {prof[p]} & {len(s)} & {po:.4f} & {sci(cp)} & {lv[0]} & {t:.2f} \\\\")
    assert len(ms) == 126 and all(float(r["approximation_max_abs"]) > float(r["ckks_max_abs"]) for r in ms)
    d = degree31_probe()
    body += [r"\midrule",
             f"GELU-31 & Low & {d['n']} & {d['po']:.4f} & {sci(d['cp'])} & {d['levels']} & {d['time']:.2f} \\\\"]
    return table(
        "tab:opc",
        "CKKS adds little error to the deployed polynomials, but not to every more accurate one. Each row gives"
        " the largest per-record maximum absolute error: $|P-O|$ compares the plaintext polynomial with the"
        " original operation, and $|C-P|$ compares the decrypted CKKS result with the plaintext polynomial."
        " Rows above the rule cover the 126 deployed-polynomial records (three seeds, blocks 0 and 5, source"
        " and converted inputs, two repeats; Low and High are the low-cost and higher-degree profiles)."
        " GELU-31 is the degree-31 block-0 refit, run twice per seed on the same fixture and low-cost"
        " parameters. Lvl.\\ is consumed CKKS levels; time is the median server time per job in seconds.",
        "@{}llrrrrr@{}", r"Op. & Profile & $n$ & $|P-O|$ & $|C-P|$ & Lvl. & Time \\", body, colsep="3pt")


def main():
    header = "% Generated by scripts/make_tables.py; do not edit by hand.\n"
    for name, body in [("generated_tables_factorial.tex", medical_interaction()),
                       ("generated_tables_block0.tex", block0_refit()),
                       ("generated_tables_refit.tex", gelu_refit())]:
        (ROOT / name).write_text(header + body + "\n")
        print(ROOT / name)
    print("coefficient growth (max |Chebyshev|, max |power basis|):", coefficient_growth())
    print("degree-31 probe:", {k: v for k, v in degree31_probe().items() if k != "domains"})


if __name__ == "__main__":
    main()
