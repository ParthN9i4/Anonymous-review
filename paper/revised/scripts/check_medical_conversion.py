#!/usr/bin/env python3
"""Recover the medical rows of Table II from the H200 run directory and compare them with the paper.

Run on the machine that holds the medical replication run (standard library only):

    python check_medical_conversion.py --run "$PVIT_RUN" --out medical_conversion_test.csv

It reads, for each of the nine native checkpoints (arm 000),
    <run>/test/<id>/report.json                               native test accuracy
    <run>/conversion_test/<id>/<profile>_<mask>/report.json   converted test metrics
and prints per-cell values, the Table II means, and any mismatch with the numbers now in the paper.
Copy the CSV back to paper/revised/data/ so the paper's medical numbers have a committed source.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

DATASETS = ["bloodmnist", "pathmnist", "dermamnist"]
SEEDS = [42, 43, 44]
PROFILES = ["budget", "accurate"]
MASKS = ["gelu", "attention", "norm", "all"]

# Values currently printed in Table II: (mean change pp, SD pp) per mask, then all-operation invalid count.
PAPER = {
    ("bloodmnist", "budget"): ([(-2.04, 1.40), (-74.11, 8.50), (-0.99, 0.94), (-77.48, 16.32)], 2726),
    ("bloodmnist", "accurate"): ([(-0.03, 0.21), (-68.14, 21.19), (0.01, 0.06), (-88.06, 7.11)], 6410),
    ("pathmnist", "budget"): ([(-38.36, 5.21), (-44.25, 12.91), (1.66, 0.35), (-51.60, 5.82)], 310),
    ("pathmnist", "accurate"): ([(-3.22, 1.45), (-29.66, 19.62), (-0.84, 0.79), (-33.43, 14.47)], 577),
    ("dermamnist", "budget"): ([(-0.18, 0.75), (-21.35, 3.86), (0.20, 0.52), (-12.57, 1.23)], 7),
    ("dermamnist", "accurate"): ([(0.13, 0.34), (-22.21, 2.94), (-0.07, 0.10), (-21.70, 2.47)], 7),
}


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("medical_conversion_test.csv"))
    a = p.parse_args()

    rows, missing = [], []
    for d in DATASETS:
        for s in SEEDS:
            job = f"{d}_s{s}_000"
            native = a.run / "test" / job / "report.json"
            if not native.exists():
                missing.append(str(native))
                continue
            ref = read(native)["metrics"]
            for profile in PROFILES:
                for mask in MASKS:
                    f = a.run / "conversion_test" / job / f"{profile}_{mask}" / "report.json"
                    if not f.exists():
                        missing.append(str(f))
                        continue
                    r = read(f)
                    m = r["metrics"]
                    rows.append(dict(dataset=d, seed=s, profile=profile, mask=mask, status=r["status"],
                                     n=m["n"], native_accuracy=ref["accuracy_percent"],
                                     converted_accuracy=m["accuracy_percent"],
                                     change_pp=m["accuracy_percent"] - ref["accuracy_percent"],
                                     invalid_outputs=m["invalid_outputs"], gate=r.get("gate")))

    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out} (expected 72); missing files: {len(missing)}")
    for m in missing:
        print("  MISSING", m)

    problems = 0
    for (d, profile), (cells, paper_invalid) in PAPER.items():
        sub = [r for r in rows if r["dataset"] == d and r["profile"] == profile]
        print(f"\n{d} {profile}")
        for mask, (pm, psd) in zip(MASKS, cells):
            ch = [r["change_pp"] for r in sub if r["mask"] == mask]
            if len(ch) != 3:
                print(f"  {mask:9s} only {len(ch)} seeds")
                problems += 1
                continue
            mean, sd = st.mean(ch), st.stdev(ch)
            ok = abs(mean - pm) < 0.006 and abs(sd - psd) < 0.006
            problems += not ok
            print(f"  {mask:9s} {mean:+.2f} +- {sd:.2f}   paper {pm:+.2f} +- {psd:.2f}   {'ok' if ok else 'MISMATCH'}"
                  f"   invalid {sum(r['invalid_outputs'] for r in sub if r['mask'] == mask)}")
        alls = [r for r in sub if r["mask"] == "all"]
        inv, n = sum(r["invalid_outputs"] for r in alls), sum(r["n"] for r in alls)
        ok = inv == paper_invalid
        problems += not ok
        print(f"  all-operation invalid: {inv} of {n} ({100 * inv / n:.2f}%)   paper {paper_invalid}"
              f"   {'ok' if ok else 'MISMATCH'}")
    print("\nALL MATCH" if problems == 0 and not missing else f"\n{problems} mismatch(es), {len(missing)} missing")


if __name__ == "__main__":
    main()
