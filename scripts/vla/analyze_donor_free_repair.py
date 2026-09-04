#!/usr/bin/env python
"""Audit and score the preregistered donor-free repair confirmation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


CONDITIONS = (
    "conflict",
    "correct",
    "repair",
    "random_matched",
    "orthogonal_matched",
    "wrong_instruction",
    "early_matched",
    "preserve_correct",
)
MATCHED = ("random_matched", "orthogonal_matched", "wrong_instruction", "early_matched")
NO_CORRECT_FORWARD = ("conflict", "repair") + MATCHED


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return [center - radius, center + radius]


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict:
    if len(left) != len(right):
        raise AssertionError("paired vectors have different lengths")
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "left_only": left_only,
        "right_only": right_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    duplicate_keys = set()
    seen = set()
    for row in rows:
        key = (row["phase"], row["pair_id"], int(row["init_id"]), row["condition"])
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)
    if duplicate_keys:
        raise AssertionError(f"duplicate episode cells: {sorted(duplicate_keys)}")
    return rows


def summarize_group(group: list[dict]) -> dict:
    successes = sum(bool(row["success"]) for row in group)
    correct_first = sum(bool(row["correct_target_first_touched"]) for row in group)
    source_first = sum(bool(row["source_target_first_touched"]) for row in group)
    return {
        "n": len(group),
        "success": successes,
        "success_rate": successes / len(group) if group else None,
        "success_wilson_95": wilson(successes, len(group)),
        "correct_target_first_touched": correct_first,
        "source_target_first_touched": source_first,
    }


def maximum_norm_error(rows: list[dict]) -> float:
    errors = []
    for row in rows:
        for replan in row["replans"]:
            diagnostics = replan["edit_diagnostics"]
            if diagnostics is not None:
                errors.append(float(diagnostics["max_relative_norm_error"]))
    return max(errors, default=0.0)


def action_hashes(row: dict) -> list[str]:
    return [replan["action_sha256"] for replan in row["replans"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in load_rows(args.episodes) if row["phase"] == "confirm"]
    manifest = json.loads(args.manifest.read_text())
    model_hashes = {row["model_sha256"] for row in rows}
    leakage_rows = [
        {
            "pair_id": row["pair_id"],
            "init_id": row["init_id"],
            "condition": row["condition"],
            "correct_prompt_forwards": row["correct_prompt_forwards"],
        }
        for row in rows
        if row["condition"] in NO_CORRECT_FORWARD and int(row["correct_prompt_forwards"]) != 0
    ]
    integrity = {
        "manifest_expected_rows": int(manifest["n_expected"]),
        "observed_rows": len(rows),
        "complete": len(rows) == int(manifest["n_expected"]),
        "single_model_hash": len(model_hashes) == 1,
        "model_hash_matches_manifest": model_hashes == {manifest["model_sha256"]},
        "repair_path_leakage_rows": leakage_rows,
        "no_repair_path_correct_prompt_forwards": not leakage_rows,
        "lambda_zero_identity": bool(manifest["identity"]["normalized_bitwise_equal"])
        and bool(manifest["identity"]["environment_bitwise_equal"]),
        "heldout_disjoint_from_training": not (
            set(manifest["heldout_edges"]) & set(manifest["training_edges"])
        ),
    }

    by_pair_condition: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_pair_condition[(row["pair_id"], row["condition"])].append(row)
    pair_reports = []
    for pair_id in sorted(set(row["pair_id"] for row in rows)):
        groups = {
            condition: sorted(by_pair_condition[(pair_id, condition)], key=lambda row: int(row["init_id"]))
            for condition in CONDITIONS
        }
        summaries = {condition: summarize_group(group) for condition, group in groups.items()}
        repair_success = summaries["repair"]["success"]
        preservation_exact = True
        preservation_mismatches = []
        correct_by_init = {int(row["init_id"]): row for row in groups["correct"]}
        preserve_by_init = {int(row["init_id"]): row for row in groups["preserve_correct"]}
        for init_id in sorted(set(correct_by_init) | set(preserve_by_init)):
            correct = correct_by_init.get(init_id)
            preserve = preserve_by_init.get(init_id)
            exact = bool(
                correct
                and preserve
                and bool(correct["success"]) == bool(preserve["success"])
                and int(correct["n_steps"]) == int(preserve["n_steps"])
                and action_hashes(correct) == action_hashes(preserve)
            )
            preservation_exact = preservation_exact and exact
            if not exact:
                preservation_mismatches.append(init_id)
        norm_errors = {condition: maximum_norm_error(groups[condition]) for condition in MATCHED}
        gates = {
            "all_arms_n10": all(summaries[condition]["n"] == 10 for condition in CONDITIONS),
            "correct_ge8_conflict_le2": (
                summaries["correct"]["success"] >= 8 and summaries["conflict"]["success"] <= 2
            ),
            "repair_success_ge7_correct_first_ge8": (
                repair_success >= 7 and summaries["repair"]["correct_target_first_touched"] >= 8
            ),
            "repair_minus_each_control_ge5": all(
                repair_success - summaries[condition]["success"] >= 5 for condition in MATCHED
            ),
            "matched_norm_error_le1e-5": all(error <= 1e-5 for error in norm_errors.values()),
            "preservation_action_exact": preservation_exact,
        }
        paired_tests = {}
        repair_by_init = {int(row["init_id"]): bool(row["success"]) for row in groups["repair"]}
        for condition in ("conflict",) + MATCHED:
            control_by_init = {int(row["init_id"]): bool(row["success"]) for row in groups[condition]}
            common = sorted(set(repair_by_init) & set(control_by_init))
            paired_tests[f"repair_vs_{condition}"] = exact_mcnemar(
                [repair_by_init[init_id] for init_id in common],
                [control_by_init[init_id] for init_id in common],
            )
        pair_reports.append(
            {
                "pair_id": pair_id,
                "conditions": summaries,
                "maximum_relative_norm_error": norm_errors,
                "preservation_mismatch_init_ids": preservation_mismatches,
                "paired_success_tests": paired_tests,
                "gates": gates,
                "pass": all(gates.values()),
            }
        )

    pooled = {
        condition: summarize_group([row for row in rows if row["condition"] == condition])
        for condition in CONDITIONS
    }
    overall_integrity = not leakage_rows and all(
        bool(value) for key, value in integrity.items() if key != "repair_path_leakage_rows"
    )
    report = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "inputs": {"episodes": str(args.episodes), "manifest": str(args.manifest)},
        "integrity": integrity,
        "pair_reports": pair_reports,
        "pooled": pooled,
        "confirmation_pass": bool(pair_reports)
        and len(pair_reports) >= 5
        and overall_integrity
        and all(report["pass"] for report in pair_reports),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
