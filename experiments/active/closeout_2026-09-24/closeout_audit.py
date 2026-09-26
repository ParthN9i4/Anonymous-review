#!/usr/bin/env python3
"""Read-only checks of the September 2026 Private ViT evidence bundle.

Usage: python closeout_audit.py /path/to/extracted/bundle [--predictions DIR]
Optional prediction files: DIR/<dataset>_s<seed>_<arm>.npz with labels and logits.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def audit(root, predictions):
    base = root / "pvit_results" / "closeout_audit_after_20260924"
    paper = root / "pvit_closeout_20260923" / "paper" / "revised" / "paper.tex"
    campaign = root / "pvit_closeout_20260923" / "evidence" / "campaign"
    rows = read_csv(base / "medical_test.csv")
    measured = [r for r in rows if r["status"] == "MEASURED"]
    failed = [r for r in rows if r["status"] != "MEASURED"]
    print("MEDICAL FACTORIAL: declared", len(rows), "measured", len(measured),
          "other", [(r["id"], r["status"]) for r in failed])
    majority = {"bloodmnist": 666 / 3421 * 100,
                "dermamnist": 1341 / 2005 * 100}
    candidate = [r for r in measured if r["dataset"] in majority and
                 abs(float(r["accuracy_percent"]) - majority[r["dataset"]]) < 1e-9]
    print("MAJORITY-ACCURACY CANDIDATES (requires predictions to prove collapse):")
    print(json.dumps([(r["id"], r["accuracy_percent"]) for r in candidate]))
    if predictions:
        import numpy as np
        for r in candidate:
            path = predictions / (r["id"] + ".npz")
            if not path.exists():
                print("MISSING RAW PREDICTIONS:", path)
                continue
            with np.load(path) as data:
                y, logits = data["labels"], data["logits"]
                if logits.ndim != 2 or len(y) != len(logits):
                    raise ValueError(f"Invalid labels/logits shape in {path}")
                finite = np.isfinite(logits).all(axis=1)
                pred = logits[finite].argmax(axis=1)
                counts = np.bincount(pred, minlength=logits.shape[1])
                acc = 100 * np.mean(pred == y[finite]) if finite.any() else float("nan")
                print("RAW:", r["id"], "finite", int(finite.sum()), "of", len(y),
                      "predicted-class counts", counts.tolist(), "finite accuracy", acc)
                if finite.all() and abs(acc - float(r["accuracy_percent"])) > 1e-7:
                    raise ValueError(f"Accuracy mismatch: {r['id']}")

    conv = read_csv(base / "medical_conversions.csv")
    print("MEDICAL CONVERSIONS:", len(conv), "records; invalid outputs by dataset/profile/mask")
    print(json.dumps([(r["dataset"], r["profile"], r["mask"], r["seed"],
                       int(r["invalid_outputs"])) for r in conv if int(r["invalid_outputs"]) > 0]))
    val = read_csv(base / "medical_validation_variants.csv")
    keys = Counter((r["dataset"], r["seed"], r["profile"], r["suite"], r["variant"])
                   for r in val)
    if max(keys.values()) != 1:
        raise ValueError("Duplicate medical validation intervention keys")
    print("MEDICAL VALIDATION:", len(val), "rows;", len(keys), "distinct intervention keys")
    print("INTERVENTION VARIANTS:", json.dumps(sorted(set(r["variant"] for r in val))))

    frozen = list((campaign / "conversion").glob("*/frozen.json"))
    intersections = []
    for path in frozen:
        obj = json.loads(path.read_text())
        a, b = set(obj.get("fit_image_ids", [])), set(obj.get("gate_image_ids", []))
        if a & b:
            intersections.append((str(path), len(a & b)))
    print("FIT/GATE DISJOINT:", len(frozen), "metadata files checked; overlaps:", intersections)
    print("Note: fit vs *other validation* membership requires the evaluation ID lists;")
    print("      fit/gate disjointness alone does not establish that broader claim.")

    ckks = read_csv(campaign / "summary" / "ckks.csv")
    statuses = Counter(r["status"] for r in ckks)
    print("CKKS:", len(ckks), "records; statuses:", dict(statuses))
    for r in ckks:
        if r["status"] == "MEASURED" and not (
            float(r["approximation_max_abs"]) > float(r["ckks_max_abs"])):
            raise ValueError("Approximation dominance is not universal among measured records")

    source = paper.read_text()
    checks = {
        "unsupported CIFAR-10.1 mention": "CIFAR-10.1" in source,
        "missing tc128 statement in manuscript": "tc128" not in source,
        "LLM editorial-only and script drafting together":
            "editorial purposes" in source and "drafting of analysis scripts" in source,
    }
    print("MANUSCRIPT FLAGS:", json.dumps(checks))
    print("UNVERIFIED FROM THIS BUNDLE: nonfinite scoring rule at image level,"
          " raw medical class collapse, PathMNIST training traceback,"
          " validation membership, activation-weighted fit, degree-31 CKKS.")
    return int(bool(intersections))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="directory containing pvit_results and pvit_closeout_20260923")
    p.add_argument("--predictions", type=Path)
    a = p.parse_args()
    raise SystemExit(audit(a.root, a.predictions))
