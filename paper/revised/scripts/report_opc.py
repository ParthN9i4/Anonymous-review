#!/usr/bin/env python3
"""Summarize original -> polynomial -> CKKS primitive measurements.

The input CSV already contains paired local errors on identical captured values:
``approximation_max_abs`` is P-O, ``ckks_max_abs`` is C-P, and
``total_max_abs`` is C-O.  This script makes that correspondence explicit and
refuses to turn repeated ciphertext evaluations into independent model runs.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "median": None, "maximum": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="paper/data/ckks.csv")
    parser.add_argument("--json", default="paper/data/opc_summary.json")
    parser.add_argument("--csv", default="paper/data/opc_summary.csv")
    args = parser.parse_args()

    with Path(args.input).open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    measured = []
    excluded = []
    for row in rows:
        if row.get("status") != "MEASURED":
            excluded.append({
                "configuration": row.get("configuration"),
                "operation": row.get("operation"),
                "status": row.get("status"),
                "error": row.get("error"),
            })
            continue
        item = dict(row)
        item["p_minus_o_max_abs"] = finite(row.get("approximation_max_abs"))
        item["c_minus_p_max_abs"] = finite(row.get("ckks_max_abs"))
        item["c_minus_o_max_abs"] = finite(row.get("total_max_abs"))
        values = [item[k] for k in (
            "p_minus_o_max_abs", "c_minus_p_max_abs", "c_minus_o_max_abs")]
        if any(value is None for value in values):
            excluded.append({
                "configuration": row.get("configuration"),
                "operation": row.get("operation"),
                "status": "NONFINITE_OR_MISSING_ERROR",
            })
            continue
        # The infinity norm obeys the triangle inequality.  A small tolerance
        # accommodates decimal serialization of independently computed arrays.
        item["triangle_inequality_holds"] = (
            item["c_minus_o_max_abs"]
            <= item["p_minus_o_max_abs"] + item["c_minus_p_max_abs"] + 1e-9
        )
        item["approximation_dominates_ckks"] = (
            item["p_minus_o_max_abs"] > item["c_minus_p_max_abs"]
        )
        measured.append(item)

    groups = defaultdict(list)
    for row in measured:
        groups[(row["operation"], row["profile"], row["path"])].append(row)

    summaries = []
    for (operation, profile, path), group in sorted(groups.items()):
        summaries.append({
            "operation": operation,
            "profile": profile,
            "path": path,
            "trials": len(group),
            "distinct_configurations": len({x["configuration"] for x in group}),
            "p_minus_o_max_abs": summarize([x["p_minus_o_max_abs"] for x in group]),
            "c_minus_p_max_abs": summarize([x["c_minus_p_max_abs"] for x in group]),
            "c_minus_o_max_abs": summarize([x["c_minus_o_max_abs"] for x in group]),
            "approximation_dominates_fraction": sum(
                x["approximation_dominates_ckks"] for x in group) / len(group),
            "triangle_inequality_failures": sum(
                not x["triangle_inequality_holds"] for x in group),
            "median_server_seconds": statistics.median(
                float(x["server_evaluation_seconds"]) for x in group
            ),
        })

    result = {
        "notation": {
            "O": "original local operation on the captured plaintext value",
            "P": "deployed plaintext polynomial on the identical value",
            "C": "decrypted CKKS evaluation of that identical polynomial",
        },
        "scope": (
            "Primitive-level paired measurements. These are not a chained "
            "encrypted block and not encrypted classifier accuracy."
        ),
        "input_rows": len(rows),
        "measured_rows": len(measured),
        "excluded_rows": excluded,
        "all_measured": {
            "p_minus_o_max_abs": summarize([x["p_minus_o_max_abs"] for x in measured]),
            "c_minus_p_max_abs": summarize([x["c_minus_p_max_abs"] for x in measured]),
            "c_minus_o_max_abs": summarize([x["c_minus_o_max_abs"] for x in measured]),
            "approximation_dominates_fraction": (
                sum(x["approximation_dominates_ckks"] for x in measured) / len(measured)
                if measured else None
            ),
            "triangle_inequality_failures": sum(
                not x["triangle_inequality_holds"] for x in measured),
        },
        "groups": summaries,
    }

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "operation", "profile", "path", "trials", "distinct_configurations",
        "p_minus_o_median", "p_minus_o_maximum", "c_minus_p_median",
        "c_minus_p_maximum", "c_minus_o_median", "c_minus_o_maximum",
        "approximation_dominates_fraction", "triangle_inequality_failures",
        "median_server_seconds",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "operation": row["operation"],
                "profile": row["profile"],
                "path": row["path"],
                "trials": row["trials"],
                "distinct_configurations": row["distinct_configurations"],
                "p_minus_o_median": row["p_minus_o_max_abs"]["median"],
                "p_minus_o_maximum": row["p_minus_o_max_abs"]["maximum"],
                "c_minus_p_median": row["c_minus_p_max_abs"]["median"],
                "c_minus_p_maximum": row["c_minus_p_max_abs"]["maximum"],
                "c_minus_o_median": row["c_minus_o_max_abs"]["median"],
                "c_minus_o_maximum": row["c_minus_o_max_abs"]["maximum"],
                "approximation_dominates_fraction": row["approximation_dominates_fraction"],
                "triangle_inequality_failures": row["triangle_inequality_failures"],
                "median_server_seconds": row["median_server_seconds"],
            })
    print(json.dumps({"json": str(json_path), "csv": str(csv_path),
                      "measured": len(measured), "excluded": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
