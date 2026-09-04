#!/usr/bin/env python3
"""Frozen task-level analysis for ``oft_downstream_kv.py``.

Init and direction are paired repeated measures.  Every normalized summary first takes the median within task;
tasks, not rows, are the inferential unit.  The cross-scene trajectory-copying condition is reported separately and
cannot contribute to the primary success gates.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PRIMARY = "kv_img_8_31"
SINGLE = "single_resid_l8"
RANDOM = "kv_img_8_31_random"
RESAMPLES = tuple(f"kv_img_8_31_resample_c{i}" for i in range(1, 4))
SELF = "kv_img_8_31_self"
XSCENE = "kv_img_8_31_xscene"
PARAPHRASE = "kv_img_8_31_paraphrase"
MIN_TASK_CLEAN_FRACTION = 0.80
PRIMARY_CONDITIONS = (
    "single_resid_l8",
    "kv_img_8_31",
    "kv_img_0_7",
    "kv_img_8_31_self",
    "kv_img_8_31_random",
    *RESAMPLES,
    "kv_img_8_31_xscene",
)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def exact_sign_flip(differences):
    """Enumerated task-level randomization test using mean signed difference as the statistic."""
    d = np.asarray(differences, dtype=float)
    if not len(d):
        return {"n": 0, "mean": None, "p_greater": None, "p_two_sided": None}
    observed = float(d.mean())
    stats = np.array([
        float(np.mean(d * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(d))
    ])
    tol = 1e-12
    return {
        "n": len(d),
        "mean": observed,
        "p_greater": float(np.mean(stats >= observed - tol)),
        "p_two_sided": float(np.mean(np.abs(stats) >= abs(observed) - tol)),
        "enumerations": len(stats),
    }


def median_or_none(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return None if not vals else float(np.median(vals))


def run_smoke():
    unanimous = exact_sign_flip([1.0] * 10)
    assert unanimous["enumerations"] == 1024
    assert abs(unanimous["p_two_sided"] - 2 / 1024) < 1e-12
    assert abs(unanimous["p_greater"] - 1 / 1024) < 1e-12
    assert median_or_none([None, 1, 3]) == 2.0
    assert median_or_none([None]) is None
    print("OFT downstream-K/V analysis smoke: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
        return
    if not args.dir:
        parser.error("--dir is required unless --smoke is used")
    directory = Path(args.dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "oft_downstream_kv_v2":
        raise ValueError(f"expected amended oft_downstream_kv_v2, got {manifest.get('schema')!r}")
    if float(manifest.get("minimum_task_clean_separation_fraction", -1)) != MIN_TASK_CLEAN_FRACTION:
        raise ValueError("manifest clean-separation gate does not match the frozen analyzer")
    rows = [json.loads(line) for line in (directory / "rows.jsonl").open() if line.strip()]
    init_ids = manifest["init_ids"]

    expected_conditions = {
        "legacy": set(PRIMARY_CONDITIONS),
        "equal": set(PRIMARY_CONDITIONS) | {"kv_instr_8_31", "kv_both_8_31"},
    }
    seen = Counter((r["panel"], r["task"], r["init"], r["direction"], r["condition"]) for r in rows)
    duplicates = [list(k) for k, count in seen.items() if count != 1]
    expected_keys = set()
    panel_maps = {
        "legacy": {int(k): int(v) for k, v in manifest["legacy_pair_map"].items()},
        "equal": {int(k): int(v) for k, v in manifest["equal_token_pair_map"].items()},
    }
    for panel in manifest["panels"]:
        for task in panel_maps[panel]:
            for init_id in init_ids:
                for direction in ("B->A", "A->B"):
                    conditions = set(expected_conditions[panel])
                    dst = direction.split("->", 1)[1]
                    para_meta = manifest.get("pair_metadata", {}).get(f"{panel}|{task}", {}).get("paraphrase", {})
                    if panel == "equal" and para_meta.get(dst, {}).get("eligible"):
                        conditions.add(PARAPHRASE)
                    for condition in conditions:
                        expected_keys.add((panel, task, init_id, direction, condition))
    missing = [list(k) for k in sorted(expected_keys - set(seen))]
    unexpected = [list(k) for k in sorted(set(seen) - expected_keys)]

    self_rows = [r for r in rows if r["condition"] == SELF]
    self_failures = [
        [r["panel"], r["task"], r["init"], r["direction"], r["action_max_abs_to_dst"]]
        for r in self_rows if r["action_max_abs_to_dst"] > 1e-6
    ]
    write_failures = []
    for r in rows:
        if r["condition"] == SELF:
            continue
        audit = r["write_audit"]
        if audit["expected_nonzero_calls"] and not audit["all_expected_nonzero_writes_occurred"]:
            write_failures.append([
                r["panel"], r["task"], r["init"], r["direction"], r["condition"], audit
            ])
    determinism_failures = [
        [r["panel"], r["task"], r["init"], r["direction"], r["clean_repeat_max_abs"]]
        for r in rows if r["clean_repeat_max_abs"] != 0.0
    ]
    wrong_observation_contract = [
        [r["panel"], r["task"], r["init"], r["direction"], r["condition"]]
        for r in rows if not r["main_source_destination_same_physical_observation"]
    ]
    implementation_valid = not (
        duplicates or missing or unexpected or self_failures or write_failures or determinism_failures
        or wrong_observation_contract
    )

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["panel"], r["task"], r["condition"])].append(r)
    task_summaries = {}
    for (panel, task, condition), group in sorted(grouped.items()):
        valid = [r for r in group if not r["negligible_clean_contrast"]]
        valid_by_direction = Counter(r["direction"] for r in valid)
        required_per_direction = int(np.ceil(MIN_TASK_CLEAN_FRACTION * len(init_ids)))
        task_summaries[f"{panel}|{task}|{condition}"] = {
            "panel": panel,
            "task": task,
            "condition": condition,
            "n_cells": len(group),
            "n_normalized": len(valid),
            "n_negligible": len(group) - len(valid),
            "clean_separation_fraction": len(valid) / len(group) if group else 0.0,
            "clean_separation_by_direction": dict(valid_by_direction),
            "clean_separation_gate": (
                len(valid) / len(group) >= MIN_TASK_CLEAN_FRACTION
                and all(valid_by_direction[d] >= required_per_direction for d in ("A->B", "B->A"))
            ) if group else False,
            "median_D_src": median_or_none([r["D_src"] for r in valid]),
            "median_D_dst": median_or_none([r["D_dst"] for r in valid]),
            "median_R_full": median_or_none([r["R_full"] for r in valid]),
            "median_R_axis": median_or_none([r["R_axis"] for r in valid]),
            "median_l2_src_dst": median_or_none([r["l2_src_dst"] for r in group]),
        }
        if condition == XSCENE:
            task_summaries[f"{panel}|{task}|{condition}"].update({
                "median_D_xscene_src": median_or_none([r.get("D_xscene_src") for r in group]),
                "median_l2_to_xscene_src": median_or_none([r.get("l2_to_xscene_src") for r in group]),
            })
        if condition == PARAPHRASE:
            task_summaries[f"{panel}|{task}|{condition}"].update({
                "median_D_clean_para_to_dst": median_or_none([r.get("D_clean_para_to_dst") for r in group]),
                "median_D_patched_to_dst": median_or_none([r.get("D_dst") for r in valid]),
                "eligible_directions": sorted(set(r["direction"] for r in group)),
            })

    def task_metric(panel, task, condition, metric="median_D_src"):
        return task_summaries[f"{panel}|{task}|{condition}"][metric]

    legacy_tasks = sorted(panel_maps["legacy"]) if "legacy" in manifest["panels"] else []
    primary_task_rows = []
    for task in legacy_tasks:
        main = task_metric("legacy", task, PRIMARY)
        single = task_metric("legacy", task, SINGLE)
        random = task_metric("legacy", task, RANDOM)
        resamples = [task_metric("legacy", task, condition) for condition in RESAMPLES]
        clean_gate = task_summaries[f"legacy|{task}|{PRIMARY}"]["clean_separation_gate"]
        if any(v is None for v in (main, single, random, *resamples)):
            primary_task_rows.append({"task": task, "normalization_available": False})
            continue
        worst_resample_margin = min(resamples) - main
        primary_task_rows.append({
            "task": task,
            "normalization_available": True,
            "D_src_primary": main,
            "D_src_single": single,
            "D_src_random": random,
            "D_src_resamples": resamples,
            "clean_separation_gate": clean_gate,
            "clean_separation_fraction": task_summaries[f"legacy|{task}|{PRIMARY}"]["clean_separation_fraction"],
            "improvement_over_single": single - main,
            "margin_vs_random": random - main,
            "margins_vs_resamples": [value - main for value in resamples],
            "worst_margin_vs_resample": worst_resample_margin,
            "primary_D_src_le_0p50": main <= 0.50,
            "control_margins_ge_0p20": random - main >= 0.20 and worst_resample_margin >= 0.20,
            "strict_all_task_criteria": (
                main <= 0.50 and single - main >= 0.20
                and random - main >= 0.20 and worst_resample_margin >= 0.20 and clean_gate
            ),
        })
    available_primary = [r for r in primary_task_rows if r["normalization_available"]]
    median_primary = median_or_none([r.get("D_src_primary") for r in available_primary])
    median_improvement = median_or_none([r.get("improvement_over_single") for r in available_primary])
    n_control_beats = sum(bool(r.get("control_margins_ge_0p20")) for r in available_primary)
    n_strict = sum(bool(r.get("strict_all_task_criteria")) for r in available_primary)
    n_clean_gate = sum(bool(r.get("clean_separation_gate")) for r in available_primary)
    sign_flip = exact_sign_flip([r["improvement_over_single"] for r in available_primary])
    primary_support = bool(
        implementation_valid
        and len(available_primary) == 10
        and n_clean_gate == 10
        and median_primary is not None and median_primary <= 0.50
        and median_improvement is not None and median_improvement >= 0.20
        and n_control_beats >= 7
        and sign_flip["p_two_sided"] is not None and sign_flip["p_two_sided"] < 0.05
    )

    equal_tasks = sorted(panel_maps["equal"]) if "equal" in manifest["panels"] else []
    equal_rows = []
    for task in equal_tasks:
        main = task_metric("equal", task, PRIMARY)
        single = task_metric("equal", task, SINGLE)
        instr = task_metric("equal", task, "kv_instr_8_31")
        both = task_metric("equal", task, "kv_both_8_31")
        para = task_summaries.get(f"equal|{task}|{PARAPHRASE}")
        equal_rows.append({
            "task": task,
            "D_src_primary": main,
            "D_src_single": single,
            "D_src_instr": instr,
            "D_src_both": both,
            "D_clean_para_to_dst": None if para is None else para["median_D_clean_para_to_dst"],
            "D_patched_para_to_dst": None if para is None else para["median_D_patched_to_dst"],
            "paraphrase_directions": [] if para is None else para["eligible_directions"],
            "same_effect_direction": main is not None and single is not None and main < single,
        })
    n_equal_direction = sum(r["same_effect_direction"] for r in equal_rows)
    equal_gate = len(equal_rows) == 7 and n_equal_direction >= 5

    xscene = []
    for panel in manifest["panels"]:
        for task in sorted(panel_maps[panel]):
            key = f"{panel}|{task}|{XSCENE}"
            if key not in task_summaries:
                continue
            rec = task_summaries[key]
            xscene.append({
                "panel": panel,
                "task": task,
                "D_current_scene_src": rec["median_D_src"],
                "D_cross_scene_src": rec.get("median_D_xscene_src"),
                "closer_to_cross_scene_donor": (
                    rec.get("median_D_xscene_src") is not None and rec["median_D_src"] is not None
                    and rec["median_D_xscene_src"] < rec["median_D_src"]
                ),
            })

    result = {
        "manifest": manifest,
        "implementation_gates": {
            "valid": implementation_valid,
            "expected_rows": len(expected_keys),
            "observed_rows": len(rows),
            "duplicates": duplicates,
            "missing": missing,
            "unexpected": unexpected,
            "self_patch_failures": self_failures,
            "nonzero_write_failures": write_failures,
            "clean_determinism_failures": determinism_failures,
            "same_observation_contract_failures": wrong_observation_contract,
        },
        "primary_legacy_panel": {
            "tasks": primary_task_rows,
            "median_task_D_src_primary": median_primary,
            "median_task_improvement_over_single": median_improvement,
            "tasks_beating_random_and_all_three_resamples_by_0p20": n_control_beats,
            "tasks_passing_clean_separation_gate": n_clean_gate,
            "clean_separation_rule": (
                f"l2_src_dst>=1e-4 in >= {MIN_TASK_CLEAN_FRACTION:.0%} of cells and >= "
                f"ceil({MIN_TASK_CLEAN_FRACTION}*n_init) per direction"
            ),
            "tasks_passing_strict_all_criteria": n_strict,
            "exact_sign_flip_improvement_over_single": sign_flip,
            "supports_image_mediation": primary_support,
        },
        "equal_token_panel": {
            "tasks": equal_rows,
            "same_effect_direction_count": n_equal_direction,
            "required": 5,
            "gate_pass": equal_gate,
        },
        "checkpoint_level_support": primary_support,
        "scope": (
            "Released OFT checkpoint on official "
            + ("held-out " if manifest.get("held_out_default_used") else "custom ")
            + "LIBERO-goal init states; not an architecture-class claim."
        ),
        "cross_scene_diagnostic": {
            "excluded_from_primary_gates": True,
            "tasks": xscene,
            "tasks_closer_to_cross_scene_donor": sum(r["closer_to_cross_scene_donor"] for r in xscene),
            "interpretation": (
                "Closeness to the cross-scene donor flags scene/motor-state transplantation by whole-image K/V; "
                "it is not evidence for semantic instruction mediation."
            ),
        },
        "task_summaries": task_summaries,
    }
    (directory / "results.json").write_text(json.dumps(json_ready(result), indent=2) + "\n")

    lines = [
        "# OFT downstream-blocked projected-K/V result",
        "",
        f"Checkpoint: `{manifest['checkpoint']}`; official `{manifest['suite']}`; init states "
        f"`{manifest['init_ids']}`; timepoint 0; both directions.",
        "",
        "## Implementation gates",
        "",
        f"Overall: **{'PASS' if implementation_valid else 'FAIL'}**. Observed {len(rows)}/{len(expected_keys)} "
        f"frozen rows; missing={len(missing)}, duplicates={len(duplicates)}, unexpected={len(unexpected)}, "
        f"self failures={len(self_failures)}, write failures={len(write_failures)}, "
        f"determinism failures={len(determinism_failures)}.",
        "",
        "## Primary legacy ten-task panel",
        "",
        "| task | clean sep. | D_src K/V 8–31 | D_src residual L8 | improvement | margin vs random | worst margin vs C1–C3 | strict task pass |",
        "|---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for rec in primary_task_rows:
        if not rec["normalization_available"]:
            lines.append(f"| {rec['task']} | fail | excluded | excluded | — | — | — | no |")
        else:
            lines.append(
                f"| {rec['task']} | {'pass' if rec['clean_separation_gate'] else 'fail'} "
                f"({rec['clean_separation_fraction']:.0%}) | {rec['D_src_primary']:.3f} | {rec['D_src_single']:.3f} | "
                f"{rec['improvement_over_single']:.3f} | {rec['margin_vs_random']:.3f} | "
                f"{rec['worst_margin_vs_resample']:.3f} | {'yes' if rec['strict_all_task_criteria'] else 'no'} |"
            )
    lines += [
        "",
        f"Median task D_src={median_primary}; median improvement over single-layer residual={median_improvement}; "
        f"beats random and every C1–C3 donor by ≥0.20 in {n_control_beats}/10 tasks; clean separation passes "
        f"in {n_clean_gate}/10. Exact task-level sign-flip: "
        f"p(one-sided)={sign_flip['p_greater']}, p(two-sided)={sign_flip['p_two_sided']}.",
        "",
        f"Frozen checkpoint-level verdict (includes exact two-sided p<0.05): **{'PASS' if primary_support else 'FAIL'}**.",
        "",
        "## Equal-token positional-artifact panel",
        "",
        "| task family | D_src K/V 8–31 | D_src residual L8 | D_src instruction K/V | D_src IMG+INSTR | clean D para→dst | patched para→dst | same direction |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for rec in equal_rows:
        fmt = lambda x: "excluded" if x is None else f"{x:.3f}"
        lines.append(
            f"| {rec['task']} | {fmt(rec['D_src_primary'])} | {fmt(rec['D_src_single'])} | "
            f"{fmt(rec['D_src_instr'])} | {fmt(rec['D_src_both'])} | {fmt(rec['D_clean_para_to_dst'])} | "
            f"{fmt(rec['D_patched_para_to_dst'])} | {'yes' if rec['same_effect_direction'] else 'no'} |"
        )
    lines += [
        "",
        f"Same effect direction in {n_equal_direction}/7 families (requires ≥5): **{'PASS' if equal_gate else 'FAIL'}**.",
        "",
        "This panel is a positional robustness check on this checkpoint. It does not license an architecture-class verdict.",
        "",
        "## Cross-scene trajectory-copying diagnostic (not a primary gate)",
        "",
        "| panel | task | D to current-scene source | D to other-scene donor | closer to other-scene donor |",
        "|---|---:|---:|---:|:---:|",
    ]
    for rec in xscene:
        fmt = lambda x: "excluded" if x is None else f"{x:.3f}"
        lines.append(
            f"| {rec['panel']} | {rec['task']} | {fmt(rec['D_current_scene_src'])} | "
            f"{fmt(rec['D_cross_scene_src'])} | {'yes' if rec['closer_to_cross_scene_donor'] else 'no'} |"
        )
    lines += [
        "",
        "The cross-scene condition diagnoses whether persistent whole-image K/V carries a donor scene/motor state. "
        "It cannot rescue a failed primary result and does not establish semantic instruction representation.",
        "",
        "Licensed scope: immediate action-chunk causal mediation only; this experiment is neither closed-loop repair "
        "nor a learned portable instruction concept.",
    ]
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
