#!/usr/bin/env python
"""Aggregate the static and state-conditioned π0.5 instruction-repair studies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


TASKS = (1, 2)


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], int(row["task_id"]), row["condition"], float(row["alpha"]))].append(row)
    result = []
    for (phase, task_id, condition, alpha), group in sorted(groups.items()):
        result.append(
            {
                "phase": phase,
                "task_id": task_id,
                "condition": condition,
                "alpha": alpha,
                "n": len(group),
                "success": sum(bool(row["success"]) for row in group),
                "correct_target_first_touched": sum(
                    bool(row["correct_target_first_touched"]) for row in group
                ),
                "conflict_target_first_touched": sum(
                    bool(row["conflict_target_first_touched"]) for row in group
                ),
                "touched_anything": sum(bool(row["touched_any"]) for row in group),
            }
        )
    return result


def lookup(summary: list[dict], phase: str, task_id: int, condition: str) -> dict:
    matches = [
        row
        for row in summary
        if row["phase"] == phase and row["task_id"] == task_id and row["condition"] == condition
    ]
    if len(matches) != 1:
        raise AssertionError((phase, task_id, condition, matches))
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.inputs)
    summary = aggregate(rows)
    state_confirm_present = any(row["phase"] == "state_confirm" for row in rows)
    confirmation = []
    if state_confirm_present:
        for task_id in TASKS:
            conflict = lookup(summary, "state_confirm", task_id, "conflict")
            correct = lookup(summary, "state_confirm", task_id, "correct")
            live = lookup(summary, "state_confirm", task_id, "state_live")
            early = lookup(summary, "state_confirm", task_id, "state_early")
            gates = {
                "correct_ge_8_and_conflict_le_2": correct["success"] >= 8 and conflict["success"] <= 2,
                "live_success_ge_7_and_correct_first_ge_8": (
                    live["success"] >= 7 and live["correct_target_first_touched"] >= 8
                ),
                "early_success_le_2_and_correct_first_le_2": (
                    early["success"] <= 2 and early["correct_target_first_touched"] <= 2
                ),
                "live_minus_early_success_ge_5": live["success"] - early["success"] >= 5,
            }
            confirmation.append(
                {
                    "task_id": task_id,
                    "conflict": conflict,
                    "correct": correct,
                    "live": live,
                    "early": early,
                    "gates": gates,
                    "pass": all(gates.values()),
                }
            )
    report = {
        "schema_version": 1,
        "inputs": [str(path) for path in args.inputs],
        "n_rows": len(rows),
        "summary": summary,
        "state_confirmation": confirmation,
        "state_confirmation_pass": bool(confirmation) and all(row["pass"] for row in confirmation),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

