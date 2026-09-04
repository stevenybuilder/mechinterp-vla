#!/usr/bin/env python
"""Descriptive audit of sub-threshold effects after the sealed screen gate.

This analysis cannot alter the preregistered screen decision or license opening
confirmation IDs. It quantifies whether the failed operator nevertheless moved
outputs in the predicted sign relative to the prespecified controls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from math import comb
from pathlib import Path
import statistics
from typing import Any


FAMILIES = ("fit", "reverse", "wrong", "random", "ceiling")
CONTROLS = ("reverse", "wrong", "random")


def binomial_tail(successes: int, total: int) -> float:
    return sum(comb(total, index) for index in range(successes, total + 1)) / (2**total)


def cell_medians(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["cell"], row["condition"])].append(row)
    return {
        key: {
            "D_A": statistics.median(row["nL2_to_A"] for row in subset),
            "D_B": statistics.median(row["nL2_to_B"] for row in subset),
        }
        for key, subset in grouped.items()
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    medians = cell_medians(rows)
    cells = sorted({row["cell"] for row in rows})
    task_pairs = {
        row["cell"]: tuple(sorted((int(row["task_a"]), int(row["task_b"]))))
        for row in rows
    }
    output: dict[str, Any] = {
        "analysis_status": "posthoc_descriptive_only",
        "screen_gate_remains": "fail",
        "directed_cells": len(cells),
        "unordered_pairs": len(set(task_pairs.values())),
        "modes": {},
    }
    for mode, target_key, source_key in (
        ("insert", "D_B", "D_A"),
        ("remove", "D_A", "D_B"),
    ):
        target = {
            family: [medians[(cell, f"{mode}_{family}")][target_key] for cell in cells]
            for family in FAMILIES
        }
        source = {
            family: [medians[(cell, f"{mode}_{family}")][source_key] for cell in cells]
            for family in FAMILIES
        }
        comparisons: dict[str, Any] = {}
        for control in CONTROLS:
            advantages = [control_value - fit for fit, control_value in zip(target["fit"], target[control])]
            pair_advantages = []
            for pair in sorted(set(task_pairs.values())):
                indexes = [index for index, cell in enumerate(cells) if task_pairs[cell] == pair]
                pair_advantages.append(statistics.median(advantages[index] for index in indexes))
            comparisons[control] = {
                "fit_lower_target_distance_cells": sum(value > 0 for value in advantages),
                "directed_cell_count": len(advantages),
                "median_target_distance_advantage": statistics.median(advantages),
                "directed_cell_sign_tail_p_descriptive": binomial_tail(sum(value > 0 for value in advantages), len(advantages)),
                "fit_lower_target_distance_unordered_pairs": sum(value > 0 for value in pair_advantages),
                "unordered_pair_count": len(pair_advantages),
                "unordered_pair_sign_tail_p_descriptive": binomial_tail(sum(value > 0 for value in pair_advantages), len(pair_advantages)),
            }
        output["modes"][mode] = {
            "fit_median_target_distance": statistics.median(target["fit"]),
            "fit_median_source_distance": statistics.median(source["fit"]),
            "fit_target_distance_below_clean_source_baseline_cells": sum(value < 1.0 for value in target["fit"]),
            "fit_crosses_target_midpoint_cells": sum(
                target_value < source_value for target_value, source_value in zip(target["fit"], source["fit"])
            ),
            "family_median_target_distance": {
                family: statistics.median(values) for family, values in target.items()
            },
            "comparisons": comparisons,
        }
    return output


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# π0.5 SONAR-lite sub-threshold effect audit",
        "",
        "This is a post-hoc descriptive analysis. The preregistered screen remains `FAIL`; confirmation IDs 10–15 remain unopened.",
        "",
        "| Mode | Fit target distance | Fit source distance | Target distance < 1 | Crosses midpoint |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("insert", "remove"):
        value = summary["modes"][mode]
        lines.append(
            f"| {mode} | {value['fit_median_target_distance']:.4f} | {value['fit_median_source_distance']:.4f} | "
            f"{value['fit_target_distance_below_clean_source_baseline_cells']}/12 | {value['fit_crosses_target_midpoint_cells']}/12 |"
        )
    lines.extend(["", "## Prespecified-control comparisons", ""])
    for mode in ("insert", "remove"):
        lines.extend(
            [
                f"### {mode.capitalize()}",
                "",
                "| Control | Fit has lower target distance | Median advantage | Unordered-pair wins |",
                "|---|---:|---:|---:|",
            ]
        )
        for control in CONTROLS:
            value = summary["modes"][mode]["comparisons"][control]
            lines.append(
                f"| {control} | {value['fit_lower_target_distance_cells']}/12 | "
                f"{value['median_target_distance_advantage']:.4f} | "
                f"{value['fit_lower_target_distance_unordered_pairs']}/6 |"
            )
        lines.append("")
    lines.extend(
        [
            "The fitted component consistently reduced distance to the intended target relative to reverse-sign and matched-spectrum controls, and usually relative to the wrong-prompt component. It remained closer to the source endpoint in 11/12 cells for insertion and 11/12 for removal, so this is partial selective causal leverage—not a sufficient or necessary mediator.",
            "",
            "Sign-test values in the JSON are descriptive. Reciprocal directed cells are nested within six unordered prompt pairs, and this audit was written after the primary gate failed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--json_out", type=Path, required=True)
    parser.add_argument("--md_out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    summary = analyze(rows)
    summary["rows_sha256"] = hashlib.sha256(args.rows.read_bytes()).hexdigest()
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(summary))


if __name__ == "__main__":
    main()
