#!/usr/bin/env python
"""Analyze the frozen π0.5 prefill-time instruction mediation experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


CONDITIONS = (
    "prefill_instr_src",
    "prefill_restore_instr_dst",
    "prefill_restore_noninstr_dst",
    "prefill_restore_img_dst",
    "prefill_restore_randimg_dst",
    "postcache_instr_src",
)


def med(rows: list[dict], key: str) -> float:
    return float(median(float(row[key]) for row in rows))


def summarize(rows: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_cell: dict[tuple[str, int, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_cell[(row["pair"], int(row["init"]), row["direction"])][row["condition"]] = row
    missing = [condition for condition in CONDITIONS if not by_condition[condition]]
    if missing:
        raise AssertionError(f"missing conditions: {missing}")
    paired_deltas = []
    for key, conditions in by_cell.items():
        if "prefill_restore_img_dst" not in conditions or "prefill_restore_randimg_dst" not in conditions:
            raise AssertionError(f"incomplete image-control pair: {key}")
        paired_deltas.append(
            float(conditions["prefill_restore_randimg_dst"]["D_dst_10"])
            - float(conditions["prefill_restore_img_dst"]["D_dst_10"])
        )
    result = {
        "n_cells": len(by_cell),
        "medians": {
            condition: {
                "D_dst_10": med(by_condition[condition], "D_dst_10"),
                "D_src_10": med(by_condition[condition], "D_src_10"),
                "D_dst_50": med(by_condition[condition], "D_dst_50"),
                "D_src_50": med(by_condition[condition], "D_src_50"),
            }
            for condition in CONDITIONS
        },
        "median_paired_rand_minus_img_D_dst_10": float(median(paired_deltas)),
        "paired_rand_minus_img_positive_fraction": float(sum(value > 0 for value in paired_deltas) / len(paired_deltas)),
    }
    m = result["medians"]
    result["gates"] = {
        "prefill_reproduces_src": m["prefill_instr_src"]["D_src_10"] <= 0.05,
        "restore_instr_stays_src": (
            m["prefill_restore_instr_dst"]["D_src_10"] <= 0.25
            and m["prefill_restore_instr_dst"]["D_dst_10"] >= 0.75
        ),
        "restore_noninstr_returns_dst": (
            m["prefill_restore_noninstr_dst"]["D_dst_10"] <= 0.25
            and m["prefill_restore_noninstr_dst"]["D_src_10"] >= 0.75
        ),
        "restore_img_beats_random_by_0_40": result["median_paired_rand_minus_img_D_dst_10"] >= 0.40,
        "postcache_instr_stays_dst": m["postcache_instr_src"]["D_dst_10"] <= 0.25,
    }
    result["all_gates_pass"] = all(result["gates"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    identities = [row for row in rows if row["condition"] == "identity_dst"]
    if not identities:
        raise AssertionError("identity rows missing")

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata[(row["pair"], row["direction"])].append(row)
    report = {
        "schema_version": 1,
        "inputs": [str(path) for path in args.inputs],
        "n_rows": len(rows),
        "n_identity_rows": len(identities),
        "pooled": summarize(rows),
        "strata": [
            {"pair": pair, "direction": direction, **summarize(group)}
            for (pair, direction), group in sorted(strata.items())
        ],
    }
    report["overall_pass"] = report["pooled"]["all_gates_pass"] and all(
        stratum["all_gates_pass"] for stratum in report["strata"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

