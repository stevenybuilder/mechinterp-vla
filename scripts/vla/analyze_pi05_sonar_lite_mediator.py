#!/usr/bin/env python
"""Gate and report the sealed pi0.5 SONAR-lite causal stages."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Any

from sonar_lite_geometry import median, parse_id_spec


CONDITIONS = (
    "clean_A",
    "clean_B",
    "insert_fit",
    "remove_fit",
    "insert_reverse",
    "remove_reverse",
    "insert_wrong",
    "remove_wrong",
    "insert_random",
    "remove_random",
    "insert_ceiling",
    "remove_ceiling",
)
FAMILIES = ("fit", "reverse", "wrong", "random", "ceiling")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def binomial_tail(successes: int, total: int) -> float:
    return sum(comb(total, index) for index in range(successes, total + 1)) / (2**total)


def summarize_cell(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    first = rows[0]
    output: dict[str, Any] = {
        "cell": first["cell"],
        "task_a": int(first["task_a"]),
        "task_b": int(first["task_b"]),
        "unordered_pair": sorted((int(first["task_a"]), int(first["task_b"]))),
        "n_initializations": len({int(row["init"]) for row in rows}),
        "conditions": {},
        "families": {},
    }
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        output["conditions"][condition] = {
            "median_D_A": median(row["nL2_to_A"] for row in subset),
            "median_D_B": median(row["nL2_to_B"] for row in subset),
            "median_R": median(row["R"] for row in subset),
        }
    for family in FAMILIES:
        insertion = output["conditions"][f"insert_{family}"]
        removal = output["conditions"][f"remove_{family}"]
        insertion_lands = insertion["median_D_B"] < insertion["median_D_A"] and insertion["median_D_B"] <= threshold
        removal_lands = removal["median_D_A"] < removal["median_D_B"] and removal["median_D_A"] <= threshold
        output["families"][family] = {
            "insertion_lands": insertion_lands,
            "removal_lands": removal_lands,
            "joint_lands": insertion_lands and removal_lands,
        }
    return output


def analyze(rows: list[dict[str, Any]], stage: str, expected_ids: list[int]) -> dict[str, Any]:
    threshold = 0.75 if stage == "screen" else 0.50
    counts = Counter((row["cell"], int(row["init"]), row["condition"]) for row in rows)
    duplicates = [list(key) for key, count in counts.items() if count != 1]
    cells = sorted({row["cell"] for row in rows})
    missing = [
        [cell, init_id, condition]
        for cell in cells
        for init_id in expected_ids
        for condition in CONDITIONS
        if counts[(cell, init_id, condition)] != 1
    ]
    complete = len(cells) == 12 and not duplicates and not missing
    cell_rows = [summarize_cell([row for row in rows if row["cell"] == cell], threshold) for cell in cells]
    family_counts = {
        family: sum(cell["families"][family]["joint_lands"] for cell in cell_rows)
        for family in FAMILIES
    }
    successful_pairs = {
        tuple(cell["unordered_pair"])
        for cell in cell_rows
        if cell["families"]["fit"]["joint_lands"]
    }
    fit_insert_target = median(
        cell["conditions"]["insert_fit"]["median_D_B"] for cell in cell_rows
    )
    fit_remove_target = median(
        cell["conditions"]["remove_fit"]["median_D_A"] for cell in cell_rows
    )
    if stage == "screen":
        decisions = {
            "complete_12_cells": complete,
            "fit_joint_cells_at_least_8": family_counts["fit"] >= 8,
            "fit_covers_all_6_unordered_pairs": len(successful_pairs) == 6,
            "fit_insert_median_target_at_most_0_75": fit_insert_target <= 0.75,
            "fit_remove_median_target_at_most_0_75": fit_remove_target <= 0.75,
            "wrong_joint_cells_at_most_2": family_counts["wrong"] <= 2,
            "random_joint_cells_at_most_2": family_counts["random"] <= 2,
            "ceiling_joint_cells_at_least_10": family_counts["ceiling"] >= 10,
        }
    else:
        decisions = {
            "complete_12_cells": complete,
            "fit_joint_cells_at_least_10": family_counts["fit"] >= 10,
            "fit_covers_all_6_unordered_pairs": len(successful_pairs) == 6,
            "fit_insert_median_target_at_most_0_50": fit_insert_target <= 0.50,
            "fit_remove_median_target_at_most_0_50": fit_remove_target <= 0.50,
            "wrong_joint_cells_at_most_2": family_counts["wrong"] <= 2,
            "random_joint_cells_at_most_2": family_counts["random"] <= 2,
        }
    return {
        "status": "pass" if all(decisions.values()) else "fail",
        "stage": stage,
        "target_distance_threshold": threshold,
        "expected_ids": expected_ids,
        "row_count": len(rows),
        "cell_count": len(cells),
        "duplicates": duplicates,
        "missing_count": len(missing),
        "missing_first_20": missing[:20],
        "family_joint_landing_cells": family_counts,
        "fit_successful_unordered_pairs": [list(pair) for pair in sorted(successful_pairs)],
        "fit_successful_unordered_pair_count": len(successful_pairs),
        "fit_median_insert_D_B": fit_insert_target,
        "fit_median_remove_D_A": fit_remove_target,
        "fit_cell_sign_tail_p_descriptive": binomial_tail(family_counts["fit"], len(cell_rows)) if cell_rows else 1.0,
        "decisions": decisions,
        "cells": cell_rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# π0.5 SONAR-lite {summary['stage']} result",
        "",
        f"**Gate:** `{summary['status'].upper()}`  ",
        f"**Rows:** {summary['row_count']} across {summary['cell_count']} directed cells  ",
        f"**Fitted joint landings:** {summary['family_joint_landing_cells']['fit']}/12  ",
        f"**Unordered-pair coverage:** {summary['fit_successful_unordered_pair_count']}/6  ",
        f"**Median target distances:** insertion `D_B={summary['fit_median_insert_D_B']:.4f}`, removal `D_A={summary['fit_median_remove_D_A']:.4f}`",
        "",
        "| Family | Joint target landings / 12 |",
        "|---|---:|",
    ]
    for family in FAMILIES:
        lines.append(f"| {family} | {summary['family_joint_landing_cells'][family]} |")
    lines.extend(["", "## Frozen gate", ""])
    for name, passed in summary["decisions"].items():
        lines.append(f"- `{'PASS' if passed else 'FAIL'}` — {name}")
    lines.extend(
        [
            "",
            "## Cell-level results",
            "",
            "| Cell | Fit joint | Insert D_B | Remove D_A | Wrong joint | Random joint | Ceiling joint |",
            "|---|:---:|---:|---:|:---:|:---:|:---:|",
        ]
    )
    for cell in summary["cells"]:
        lines.append(
            "| {cell} | {fit} | {insert:.3f} | {remove:.3f} | {wrong} | {random} | {ceiling} |".format(
                cell=cell["cell"],
                fit="yes" if cell["families"]["fit"]["joint_lands"] else "no",
                insert=cell["conditions"]["insert_fit"]["median_D_B"],
                remove=cell["conditions"]["remove_fit"]["median_D_A"],
                wrong="yes" if cell["families"]["wrong"]["joint_lands"] else "no",
                random="yes" if cell["families"]["random"]["joint_lands"] else "no",
                ceiling="yes" if cell["families"]["ceiling"]["joint_lands"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "The binomial tail shown in the JSON is descriptive only: reciprocal directed cells share task-pair structure and are not treated as 12 independent experiments.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "confirm"), required=True)
    parser.add_argument("--screen_ids", default="8-9")
    parser.add_argument("--confirm_ids", default="10-15")
    args = parser.parse_args()
    rows_path = args.out / f"{args.stage}_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    expected_ids = parse_id_spec(args.screen_ids if args.stage == "screen" else args.confirm_ids)
    summary = analyze(rows, args.stage, expected_ids)
    summary["rows_sha256"] = sha256(rows_path)
    write_json(args.out / f"{args.stage}_summary.json", summary)
    (args.out / f"{args.stage}_report.md").write_text(render_markdown(summary))
    print(json.dumps({key: value for key, value in summary.items() if key != "cells"}, indent=2))


if __name__ == "__main__":
    main()
