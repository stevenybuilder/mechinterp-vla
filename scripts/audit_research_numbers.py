#!/usr/bin/env python3
"""Recompute the MATS project's headline numbers from preserved records.

This script deliberately uses only the Python standard library.  It writes one
machine-readable audit artifact used by the prose audit and every figure.  It
does not treat a findings Markdown file as a numerical source.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "numbers-audit-derived.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: Path):
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc


def median(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return None if not vals else statistics.median(vals)


def file_record(relative: str, expected_rows: int | None = None) -> dict:
    path = ROOT / relative
    n_rows = 0
    if path.suffix == ".jsonl":
        for _ in jsonl(path):
            n_rows += 1
    result = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "jsonl_rows": n_rows if path.suffix == ".jsonl" else None,
    }
    if expected_rows is not None:
        result["expected_rows"] = expected_rows
        result["row_count_matches"] = n_rows == expected_rows
    return result


def prefill() -> dict:
    rel = "artifacts/pi05_prefill_mediation_2026-08-31/rows.jsonl"
    rows = list(jsonl(ROOT / rel))
    conditions = (
        "prefill_instr_src",
        "prefill_restore_instr_dst",
        "prefill_restore_noninstr_dst",
        "prefill_restore_img_dst",
        "prefill_restore_randimg_dst",
        "postcache_instr_src",
    )
    medians = {}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        medians[condition] = {
            "n": len(subset),
            "D_dst_10": median(r["D_dst_10"] for r in subset),
            "D_src_10": median(r["D_src_10"] for r in subset),
            "D_dst_50": median(r["D_dst_50"] for r in subset),
            "D_src_50": median(r["D_src_50"] for r in subset),
        }
    by_unit = defaultdict(dict)
    for r in rows:
        by_unit[(r["pair"], r["direction"], r["init"])][r["condition"]] = r
    advantages = []
    for unit in by_unit.values():
        advantages.append(
            unit["prefill_restore_randimg_dst"]["D_dst_10"]
            - unit["prefill_restore_img_dst"]["D_dst_10"]
        )
    identity = [r for r in rows if r["condition"] == "identity_dst"]
    return {
        "file": file_record(rel, 1350),
        "independent_units": len(by_unit),
        "conditions": Counter(r["condition"] for r in rows),
        "medians": medians,
        "median_random_minus_all_image_D_dst_10": median(advantages),
        "positive_advantage_fraction": sum(x > 0 for x in advantages) / len(advantages),
        "identity_rows": len(identity),
        "identity_rows_exact_D_dst_zero": sum(r["D_dst_10"] == 0 for r in identity),
    }


def state_repair() -> dict:
    rel = "artifacts/pi05_instruction_repair_2026-08-31/state_confirm/episodes.jsonl"
    rows = list(jsonl(ROOT / rel))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task_id"], row["condition"])].append(row)
    cells = {}
    for (task, condition), group in sorted(grouped.items()):
        cells[f"task{task}:{condition}"] = {
            "n": len(group),
            "success": sum(bool(r["success"]) for r in group),
            "correct_first_touch": sum(bool(r["correct_target_first_touched"]) for r in group),
            "conflict_first_touch": sum(bool(r["conflict_target_first_touched"]) for r in group),
        }
    pooled = {}
    for condition in sorted({r["condition"] for r in rows}):
        group = [r for r in rows if r["condition"] == condition]
        pooled[condition] = {
            "n": len(group),
            "success": sum(bool(r["success"]) for r in group),
            "correct_first_touch": sum(bool(r["correct_target_first_touched"]) for r in group),
        }
    return {"file": file_record(rel, 80), "cells": cells, "pooled": pooled}


def mediation() -> dict:
    rel = "artifacts/pi05_mediation_2026-08-31/run2/rows.jsonl"
    rows = list(jsonl(ROOT / rel))
    units = defaultdict(dict)
    for row in rows:
        units[(row["cell"], row["init"])][row["condition"]] = row
    normalized = defaultdict(list)
    by_cell = defaultdict(lambda: defaultdict(list))
    donor_copy = []
    for (cell, _), group in units.items():
        # The frozen analysis normalizes both distances by the clean A/B
        # separation recorded as clean-B's distance to A for this init.
        denom_b = group["C1_clean_B"]["l2_to_A"]
        denom_a = denom_b
        for condition, row in group.items():
            db = row["l2_to_B"] / denom_b
            da = row["l2_to_A"] / denom_a
            normalized[condition].append((db, da, row["R"]))
            by_cell[cell][condition].append((db, da, row["R"]))
            if condition == "C3X_carrier_xlay":
                donor_copy.append(row["l2_to_B_donor"] < row["l2_to_B"])
    summaries = {}
    for condition, vals in sorted(normalized.items()):
        per_cell = [
            (
                median(x[0] for x in cell_conditions[condition]),
                median(x[1] for x in cell_conditions[condition]),
                median(x[2] for x in cell_conditions[condition]),
            )
            for cell_conditions in by_cell.values()
            if condition in cell_conditions
        ]
        summaries[condition] = {
            "n": len(vals),
            "median_D_B": median(x[0] for x in per_cell),
            "median_D_A": median(x[1] for x in per_cell),
            "median_R": median(x[2] for x in per_cell),
            "cell_medians_B_below_0_5": sum(
                median(x[0] for x in cell_conditions[condition]) < 0.5
                for cell_conditions in by_cell.values()
                if condition in cell_conditions
            ),
            "eligible_cells": sum(condition in cell_conditions for cell_conditions in by_cell.values()),
        }
    dose_rel = "artifacts/pi05_mediation_2026-08-31/dose1/rows.jsonl"
    dose_rows = list(jsonl(ROOT / dose_rel))
    dose_baseline = defaultdict(float)
    for row in dose_rows:
        key = (row["cell"], row["init"])
        dose_baseline[key] = max(dose_baseline[key], float(row["l2_to_B"]))
    dose = {}
    for arm in ("LOC", "RAND"):
        for n_pos in sorted({r["n_pos"] for r in dose_rows}):
            vals = []
            for row in dose_rows:
                if row["arm"] == arm and row["n_pos"] == n_pos:
                    denom = dose_baseline[(row["cell"], row["init"])]
                    vals.append(row["l2_to_B"] / denom)
            dose[f"{arm}:{n_pos}"] = median(vals)
    return {
        "file": file_record(rel, 696),
        "dose_file": file_record(dose_rel, 384),
        "independent_units": len(units),
        "summary": summaries,
        "cross_layout_rows_closer_to_donor_trajectory": sum(donor_copy),
        "cross_layout_rows": len(donor_copy),
        "dose_median_D_B": dose,
        "one_sided_sign_probability_12_of_12": 0.5**12,
    }


def sonar_lite() -> dict:
    selection_rel = "artifacts/pi05_sonar_lite_source_mediator_v1/selection_rows.jsonl"
    selection = list(jsonl(ROOT / selection_rel))
    selected = [r for r in selection if r["layer"] == 13 and r["rank"] == 16]
    cell_sel = defaultdict(list)
    for row in selected:
        cell_sel[row["cell"]].append(row)
    cell_fit = {cell: median(r["cos_fit"] for r in group) for cell, group in cell_sel.items()}
    cell_wrong = {cell: median(r["cos_wrong"] for r in group) for cell, group in cell_sel.items()}

    screen_rel = "artifacts/pi05_sonar_lite_source_mediator_v1/screen_rows.jsonl"
    rows = list(jsonl(ROOT / screen_rel))
    cell_cond = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cell_cond[row["cell"]][row["condition"]].append(row)
    cell_medians = {}
    for cell, conditions in cell_cond.items():
        cell_medians[cell] = {
            c: {
                "D_A": median(r["nL2_to_A"] for r in group),
                "D_B": median(r["nL2_to_B"] for r in group),
                "R": median(r["R"] for r in group),
            }
            for c, group in conditions.items()
        }
    insert_lands = [v["insert_fit"]["D_B"] < 0.5 for v in cell_medians.values()]
    remove_lands = [v["remove_fit"]["D_A"] < 0.5 for v in cell_medians.values()]
    beats = {}
    for control in ("random", "reverse", "wrong"):
        beats[f"insert_{control}"] = sum(
            v["insert_fit"]["D_B"] < v[f"insert_{control}"]["D_B"] for v in cell_medians.values()
        )
        beats[f"remove_{control}"] = sum(
            v["remove_fit"]["D_A"] < v[f"remove_{control}"]["D_A"] for v in cell_medians.values()
        )
    return {
        "selection_file": file_record(selection_rel),
        "screen_file": file_record(screen_rel, 288),
        "selected_layer": 13,
        "selected_rank": 16,
        "selection_cells": len(cell_sel),
        "median_cos_fit": median(cell_fit.values()),
        "median_cos_wrong": median(cell_wrong.values()),
        "median_selectivity": median(cell_fit[c] - cell_wrong[c] for c in cell_fit),
        "fit_beats_wrong_cells": sum(cell_fit[c] > cell_wrong[c] for c in cell_fit),
        "joint_target_landings": sum(i and r for i, r in zip(insert_lands, remove_lands)),
        "insert_target_landings": sum(insert_lands),
        "remove_target_landings": sum(remove_lands),
        "median_insert_D_B": median(v["insert_fit"]["D_B"] for v in cell_medians.values()),
        "median_remove_D_A": median(v["remove_fit"]["D_A"] for v in cell_medians.values()),
        "insert_remained_source_like": sum(
            v["insert_fit"]["D_A"] < v["insert_fit"]["D_B"] for v in cell_medians.values()
        ),
        "remove_remained_source_like": sum(
            v["remove_fit"]["D_B"] < v["remove_fit"]["D_A"] for v in cell_medians.values()
        ),
        "beats_controls_cells": beats,
    }


def donor_free() -> dict:
    rel = "artifacts/pi05_donor_free_repair_2026-09-02/confirm/episodes.jsonl"
    rows = list(jsonl(ROOT / rel))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["pair_id"], row["condition"])].append(row)
    conditions = sorted({r["condition"] for r in rows})
    complete_pairs = sorted(
        pair for pair in {r["pair_id"] for r in rows}
        if all(len(grouped[(pair, c)]) == 10 for c in conditions)
    )
    incomplete_pairs = sorted(set(r["pair_id"] for r in rows) - set(complete_pairs))
    summary = {}
    for condition in conditions:
        group = [r for r in rows if r["pair_id"] in complete_pairs and r["condition"] == condition]
        summary[condition] = {
            "n": len(group),
            "success": sum(bool(r["success"]) for r in group),
            "correct_first_touch": sum(bool(r["correct_target_first_touched"]) for r in group),
            "correct_prompt_forwards": sum(int(r["correct_prompt_forwards"]) for r in group),
        }
    return {
        "file": file_record(rel, 250),
        "complete_pairs": complete_pairs,
        "incomplete_pairs": incomplete_pairs,
        "complete_pair_count": len(complete_pairs),
        "summary_complete_pairs_only": summary,
    }


def attention_pathway() -> dict:
    files = {
        "writer_screen": ("artifacts/pi05_attention_pathway_2026-09-04_v1/writer_screen_rows.jsonl", 480),
        "reader_screen": ("artifacts/pi05_attention_pathway_2026-09-04_v1/reader_screen_rows.jsonl", 240),
        "writer_development": ("artifacts/pi05_attention_resolution_2026-09-04_v1/writer_development_rows.jsonl", 1440),
        "reader_development": ("artifacts/pi05_attention_resolution_2026-09-04_v1/reader_development_rows.jsonl", 828),
        "writer_band": ("artifacts/pi05_writer_band_6_8_confirmation_2026-09-04_v3/rows.jsonl", 280),
    }
    inventory = {name: file_record(path, count) for name, (path, count) in files.items()}
    writer = list(jsonl(ROOT / files["writer_screen"][0]))
    reader = list(jsonl(ROOT / files["reader_screen"][0]))
    band = list(jsonl(ROOT / files["writer_band"][0]))

    def med(rows, condition, field="preference_A"):
        return median(r[field] for r in rows if r["condition"] == condition)

    result = {
        "files": inventory,
        "writer_screen": {
            "block_all_noninstruction_preference_A": med(writer, "writer_block_all_noninstruction"),
            "block_instruction_to_image_preference_A": med(writer, "writer_block_image"),
            "block_instruction_to_other_preference_A": med(writer, "writer_block_other"),
            "selected_block0_rescue_preference_B": -med(writer, "writer_rescue_l0_image_selected"),
        },
        "reader_screen": {
            "block_preference_A": med(reader, "reader_block"),
            "selected_rescue_preference_B": -med(reader, "reader_rescue_selected"),
        },
        "writer_band": {
            "selected_block_preference_A": med(band, "writer_block_selected_band"),
            "selected_rescue_preference_B": -med(band, "writer_rescue_selected_band"),
        },
    }
    for key in ("writer_development", "reader_development"):
        summary_path = ROOT / "artifacts/pi05_attention_resolution_2026-09-04_v1" / f"{key}.json"
        result[key] = json.loads(summary_path.read_text())
    result["writer_band_summary"] = json.loads(
        (ROOT / "artifacts/pi05_writer_band_6_8_confirmation_2026-09-04_v3/summary.json").read_text()
    )
    return result


def curvature() -> dict:
    specs = {
        "representation": (
            "artifacts/pi05_lean_midpoint_curvature_2026-09-04_v1/rows.jsonl",
            "curvature_to_chord_ratio",
            60,
        ),
        "action": (
            "artifacts/pi05_curvature_action_2026-09-04_v2/rows.jsonl",
            "action_change_over_endpoint_distance",
            60,
        ),
    }
    result = {}
    for name, (rel, field, expected) in specs.items():
        rows = list(jsonl(ROOT / rel))
        by_cell = defaultdict(list)
        for row in rows:
            by_cell[row["cell"]].append(float(row[field]))
        item = {
            "file": file_record(rel, expected),
            "median": median(r[field] for r in rows),
            "minimum": min(float(r[field]) for r in rows),
            "maximum": max(float(r[field]) for r in rows),
            "cell_medians": {cell: median(values) for cell, values in sorted(by_cell.items())},
            "cell_medians_at_least_0_1": sum(median(v) >= 0.1 for v in by_cell.values()),
        }
        if name == "representation":
            item["all_endpoint_identities_exact"] = all(bool(r["b_endpoint_bitwise_identity"]) for r in rows)
        else:
            item["all_rescues_bitwise_exact"] = all(
                bool(r["prefix_rescue_bitwise_identity"]) and bool(r["action_rescue_bitwise_identity"])
                for r in rows
            )
        result[name] = item
    return result


def matched_band_transform() -> dict:
    """Independently recompute the matched-transform headline effects.

    This does not import the experiment runner. It reconstructs the action axis
    directly from the stored first-ten-action arrays and treats each directed
    prompt-pair cell as the inferential unit.
    """
    rel = "artifacts/pi05_matched_band_transform_2026-09-04_v1/rows.jsonl"
    summary_rel = "artifacts/pi05_matched_band_transform_2026-09-04_v1/summary.json"
    rows = list(jsonl(ROOT / rel))
    summary = json.loads((ROOT / summary_rel).read_text())
    by_unit = defaultdict(dict)
    duplicate_keys = []
    for row in rows:
        key = (row["cell"], int(row["init"]))
        condition = row["condition"]
        if condition in by_unit[key]:
            duplicate_keys.append([*key, condition])
        by_unit[key][condition] = row

    def flatten(value):
        return [float(number) for action in value for number in action]

    def axis(value, endpoint_a, endpoint_b):
        vector = [b - a for a, b in zip(endpoint_a, endpoint_b)]
        delta = [x - a for x, a in zip(value, endpoint_a)]
        denominator = sum(number * number for number in vector)
        if denominator <= 1e-18:
            raise RuntimeError("matched-transform clean action endpoints collapsed")
        scale = math.sqrt(denominator)
        return {
            "progress_to_B": sum(x * y for x, y in zip(delta, vector)) / denominator,
            "D_A": math.sqrt(sum(number * number for number in delta)) / scale,
            "D_B": math.sqrt(sum((x - b) ** 2 for x, b in zip(value, endpoint_b))) / scale,
        }

    maximum_metric_error = 0.0
    for unit in by_unit.values():
        a = flatten(unit["clean_A"]["action_first_10"])
        b = flatten(unit["clean_B"]["action_first_10"])
        for row in unit.values():
            recomputed = axis(flatten(row["action_first_10"]), a, b)
            maximum_metric_error = max(
                maximum_metric_error,
                *(abs(recomputed[name] - float(row[name])) for name in recomputed),
            )

    conditions = sorted({row["condition"] for row in rows})
    cells = sorted({row["cell"] for row in rows})
    cell_values = {}
    for condition in conditions:
        cell_values[condition] = {
            cell: median(
                row["progress_to_B"]
                for row in rows
                if row["cell"] == cell and row["condition"] == condition
            )
            for cell in cells
        }

    contrasts = {
        "matched_minus_mismatched_full": [
            cell_values["matched_full"][cell]
            - cell_values["mismatched_full_norm_matched"][cell]
            for cell in cells
        ],
        "matched_full_minus_matched_attention_only": [
            cell_values["matched_full"][cell]
            - cell_values["matched_attention_only"][cell]
            for cell in cells
        ],
        "mismatched_full_minus_random": [
            cell_values["mismatched_full_norm_matched"][cell]
            - cell_values["random_norm_matched"][cell]
            for cell in cells
        ],
    }

    def sign_test(values):
        nonzero = [value for value in values if abs(value) > 1e-12]
        positives = sum(value > 0 for value in nonzero)
        lower = min(positives, len(nonzero) - positives)
        p_value = min(
            1.0,
            2.0 * sum(math.comb(len(nonzero), k) for k in range(lower + 1)) / (2 ** len(nonzero)),
        ) if nonzero else 1.0
        return {"positive": positives, "negative": len(nonzero) - positives, "p_two_sided": p_value}

    contrast_summary = {
        name: {"median_cell_effect": median(values), "sign_test": sign_test(values)}
        for name, values in contrasts.items()
    }
    condition_medians = {
        condition: median(values.values()) for condition, values in cell_values.items()
    }
    canonical_errors = []
    for condition, value in condition_medians.items():
        expected = summary["conditions"][condition]["median_cell_progress_to_B"]
        canonical_errors.append(abs(value - expected))
    for name, value in contrast_summary.items():
        expected = summary["contrasts"][name]
        canonical_errors.append(abs(value["median_cell_effect"] - expected["median_cell_effect"]))
        canonical_errors.append(abs(value["sign_test"]["p_two_sided"] - expected["exact_two_sided_sign_test"]["p_two_sided"]))

    norm_errors = []
    for unit in by_unit.values():
        full = float(unit["matched_full"]["message_frobenius_norm"])
        mismatch = float(unit["mismatched_full_norm_matched"]["message_frobenius_norm"])
        attention = float(unit["matched_attention_only"]["message_frobenius_norm"])
        mismatch_attention = float(unit["mismatched_attention_only_norm_matched"]["message_frobenius_norm"])
        norm_errors.extend([abs(full - mismatch) / full, abs(attention - mismatch_attention) / attention])

    return {
        "file": file_record(rel, 420),
        "summary_file": file_record(summary_rel),
        "units": len(by_unit),
        "directed_prompt_pair_cells": len(cells),
        "initial_states_per_cell": len({int(row["init"]) for row in rows}),
        "duplicate_keys": duplicate_keys,
        "maximum_stored_axis_metric_error": maximum_metric_error,
        "maximum_norm_match_relative_error": max(norm_errors),
        "condition_median_cell_progress_to_B": condition_medians,
        "contrasts": contrast_summary,
        "maximum_error_against_canonical_summary": max(canonical_errors),
        "canonical_bootstrap_intervals": {
            condition: summary["conditions"][condition]["bootstrap_95pct_interval"]
            for condition in (
                "matched_full",
                "mismatched_full_norm_matched",
                "matched_attention_only",
                "random_norm_matched",
            )
        },
    }


def matched_band_rollout_screen() -> dict:
    """Recompute the exploratory closed-loop screen from episode records."""
    rel = "artifacts/pi05_matched_band_rollout_screen_2026-09-04_v1/episodes.jsonl"
    summary_rel = "artifacts/pi05_matched_band_rollout_screen_2026-09-04_v1/summary.json"
    rows = list(jsonl(ROOT / rel))
    stored = json.loads((ROOT / summary_rel).read_text())
    conditions = ("clean_A", "clean_B", "matched_full")
    outcomes = (
        "success_B",
        "B_target_first_touched",
        "A_target_first_touched",
        "touched_anything",
    )
    keys = [(row["cell"], int(row["init"]), row["condition"]) for row in rows]
    counts = {
        outcome: {
            condition: sum(bool(row[outcome]) for row in rows if row["condition"] == condition)
            for condition in conditions
        }
        for outcome in outcomes
    }
    med_steps = {
        condition: median(row["n_steps"] for row in rows if row["condition"] == condition)
        for condition in conditions
    }
    cells = sorted({row["cell"] for row in rows})
    cell_rows = {(row["cell"], row["condition"]): row for row in rows}
    b_first_differences = [
        int(bool(cell_rows[(cell, "matched_full")]["B_target_first_touched"]))
        - int(bool(cell_rows[(cell, "clean_A")]["B_target_first_touched"]))
        for cell in cells
    ]
    success_differences = [
        int(bool(cell_rows[(cell, "matched_full")]["success_B"]))
        - int(bool(cell_rows[(cell, "clean_A")]["success_B"]))
        for cell in cells
    ]

    def sign_test(values):
        nonzero = [value for value in values if value != 0]
        positives = sum(value > 0 for value in nonzero)
        lower = min(positives, len(nonzero) - positives)
        p_value = min(
            1.0,
            2.0 * sum(math.comb(len(nonzero), k) for k in range(lower + 1)) / (2 ** len(nonzero)),
        ) if nonzero else 1.0
        return {"positive": positives, "negative": len(nonzero) - positives, "p_two_sided": p_value}

    matched_rows = [row for row in rows if row["condition"] == "matched_full"]
    clean_rows = [row for row in rows if row["condition"] != "matched_full"]
    recomputed = {
        "rows": len(rows),
        "counts": counts,
        "median_steps": med_steps,
    }
    stored_errors = [
        abs(counts[outcome][condition] - stored["counts"][outcome][condition])
        for outcome in outcomes
        for condition in conditions
    ] + [
        abs(med_steps[condition] - stored["median_steps"][condition])
        for condition in conditions
    ]
    return {
        "file": file_record(rel, 36),
        "summary_file": file_record(summary_rel),
        "directed_prompt_pair_cells": len(cells),
        "initial_states_per_cell": len({int(row["init"]) for row in rows}),
        "conditions": Counter(row["condition"] for row in rows),
        "duplicate_keys": [list(key) for key, n in Counter(keys).items() if n > 1],
        "counts": counts,
        "median_steps": med_steps,
        "matched_cells_with_B_first_touch": [
            row["cell"] for row in matched_rows if bool(row["B_target_first_touched"])
        ],
        "B_first_touch_sign_test_matched_vs_clean_A": sign_test(b_first_differences),
        "success_sign_test_matched_vs_clean_A": sign_test(success_differences),
        "all_matched_replans_have_nonzero_message": all(
            all(float(replan["message_frobenius_norm"]) > 0 for replan in row["replans"])
            for row in matched_rows
        ),
        "all_clean_replans_have_zero_message": all(
            all(float(replan["message_frobenius_norm"]) == 0 for replan in row["replans"])
            for row in clean_rows
        ),
        "maximum_error_against_stored_summary": max(stored_errors),
        "recomputed_summary_subset": recomputed,
    }


def oft() -> dict:
    rel = "artifacts/oft_downstream_kv/v2_20260831/rows.jsonl"
    rows = list(jsonl(ROOT / rel))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["panel"], int(row["task"]), row["condition"])].append(row)

    def task_med(panel, task, condition):
        return median(r["D_src"] for r in grouped[(panel, task, condition)] if not r["negligible_clean_contrast"])

    legacy_tasks = sorted({task for panel, task, _ in grouped if panel == "legacy"})
    improvements = []
    controls_pass = 0
    task_rows = []
    for task in legacy_tasks:
        primary = task_med("legacy", task, "kv_img_8_31")
        single = task_med("legacy", task, "single_resid_l8")
        random = task_med("legacy", task, "kv_img_8_31_random")
        resamples = [task_med("legacy", task, f"kv_img_8_31_resample_c{i}") for i in range(1, 4)]
        improvement = single - primary
        improvements.append(improvement)
        passed = all(c - primary >= 0.2 for c in [random, *resamples])
        controls_pass += passed
        task_rows.append({
            "task": task,
            "primary_D_src": primary,
            "single_residual_D_src": single,
            "improvement": improvement,
            "beats_all_controls_by_0_2": passed,
        })
    equal_tasks = sorted({task for panel, task, _ in grouped if panel == "equal"})
    equal = {
        condition: median(task_med("equal", task, condition) for task in equal_tasks)
        for condition in ("kv_img_8_31", "kv_instr_8_31", "kv_both_8_31")
    }
    return {
        "file": file_record(rel, 3440),
        "legacy_task_count": len(legacy_tasks),
        "legacy_tasks": task_rows,
        "median_task_primary_D_src": median(r["primary_D_src"] for r in task_rows),
        "median_task_improvement_over_single": median(improvements),
        "same_sign_improvement_tasks": sum(x > 0 for x in improvements),
        "two_sided_unanimous_sign_probability": 2 / (2 ** len(improvements)),
        "tasks_beating_all_controls_by_0_2": controls_pass,
        "equal_token_panel_median_task_D_src": equal,
        "implementation_checks": {
            "all_clean_repeats_exact": all(r["clean_repeat_max_abs"] == 0 for r in rows),
            "same_observation_contract": all(bool(r["main_source_destination_same_physical_observation"]) for r in rows),
            "self_patch_max_abs": max(
                float(r["action_max_abs_to_dst"]) for r in rows if r["condition"] == "kv_img_8_31_self"
            ),
        },
    }


def stage2() -> dict:
    base = "artifacts/vla_stage2/20260830-094027"
    files = {
        "object_discovery": (f"{base}/libero_object_discovery/rows.jsonl", 24000),
        "goal_discovery": (f"{base}/libero_goal_discovery/rows.jsonl", 18000),
        "goal_confirm": (f"{base}/libero_goal_confirm/rows.jsonl", 18000),
    }
    inventory = {name: file_record(rel, expected) for name, (rel, expected) in files.items()}
    results = json.loads((ROOT / f"{base}/libero_goal_confirm/results.json").read_text())
    probes = results["probes"]["resid"]
    instruction_acc = [probes[f"L{layer}_INSTR"]["acc"] for layer in range(18)]
    image_acc = [probes[f"L{layer}_IMG"]["acc"] for layer in range(18)]
    attention = results["attention_mass_by_layer"]["clean_A"]
    late = attention[12:18]
    instr_per_token = statistics.mean(row[2] for row in late) / 9
    image_per_token = statistics.mean(row[0] for row in late) / 512

    condition_cell_medians = defaultdict(list)
    for pair in results["pairs"].values():
        for direction in pair.values():
            for condition, values in direction["conditions"].items():
                condition_cell_medians[condition].append(values["R_median"])
    return {
        "files": inventory,
        "total_rows": sum(x[1] for x in files.values()),
        "goal_confirm_units": results["n_units"],
        "residual_probe_instruction_accuracy_min": min(instruction_acc),
        "residual_probe_instruction_accuracy_layers_0_16": instruction_acc[:17],
        "residual_probe_image_accuracy_layer_0": image_acc[0],
        "residual_probe_image_accuracy_layer_1": image_acc[1],
        "residual_probe_image_accuracy_layer_4": image_acc[4],
        "late_attention_instruction_per_token": instr_per_token,
        "late_attention_image_per_token": image_per_token,
        "late_attention_per_token_ratio": instr_per_token / image_per_token,
        "late_attention_total_image_to_instruction_ratio": statistics.mean(row[0] for row in late)
        / statistics.mean(row[2] for row in late),
        "median_cell_R": {
            c: median(condition_cell_medians[c])
            for c in ("KV[INSTR]@all", "KO[INSTR]", "KV[IMG]@12-17", "KV[IMG]@all", "RS[IMG]@all")
        },
        "cell_R_values": {
            c: condition_cell_medians[c]
            for c in ("KV[INSTR]@all", "KO[INSTR]", "KV[IMG]@12-17", "KV[IMG]@all", "RS[IMG]@all")
        },
    }


def behavior() -> dict:
    cells_rel = "artifacts/two_stage_cells.json"
    cells = json.loads((ROOT / cells_rel).read_text())
    in_rows = [c for c in cells if c["in_region"]]
    out_rows = [c for c in cells if not c["in_region"]]

    def weighted(group):
        return sum(float(x["obey"]) * int(x["n"]) for x in group) / sum(int(x["n"]) for x in group)

    arbitration_files = {}
    for suite in ("libero_goal", "libero_object"):
        rel = f"artifacts/vla_arbitration/20260830-192300/{suite}/per_episode.jsonl"
        rows = list(jsonl(ROOT / rel))
        arbitration_files[suite] = {
            **file_record(rel),
            "outcomes": Counter(r["outcome"] for r in rows),
            "tasks": sorted({r["task_id"] for r in rows}),
        }
    return {
        "two_stage_file": file_record(cells_rel),
        "cells": len(cells),
        "in_region_cells": len(in_rows),
        "out_region_cells": len(out_rows),
        "in_region_obedience": weighted(in_rows),
        "out_region_obedience": weighted(out_rows),
        "difference": weighted(in_rows) - weighted(out_rows),
        "arbitration_files": arbitration_files,
    }


def monitor() -> dict:
    rel = "artifacts/pi05_monitor/monitor_cost.json"
    rows = json.loads((ROOT / rel).read_text())
    result = []
    for row in rows:
        f1 = 2 * row["tp"] / (2 * row["tp"] + row["fp"] + row["fn"])
        positives = row["tp"] + row["fn"]
        total = row["tp"] + row["fp"] + row["fn"] + row["tn"]
        constant_positive_f1 = 2 * positives / (total + positives)
        result.append({**row, "f1_recomputed": f1, "constant_positive_f1": constant_positive_f1})
    return {"file": file_record(rel), "splits": result}


def main() -> None:
    audit = {
        "schema_version": 1,
        "generated_by": "scripts/audit_research_numbers.py",
        "scope": "Core MATS VLA evidence only; safety/COAST robotics archive is not imported as primary evidence.",
        "behavior": behavior(),
        "stage2": stage2(),
        "prefill": prefill(),
        "state_repair": state_repair(),
        "mediation": mediation(),
        "sonar_lite": sonar_lite(),
        "donor_free": donor_free(),
        "attention_pathway": attention_pathway(),
        "curvature": curvature(),
        "matched_band_transform": matched_band_transform(),
        "matched_band_rollout_screen": matched_band_rollout_screen(),
        "openvla_oft": oft(),
        "monitor": monitor(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(OUT),
        "sha256": sha256(OUT),
        "core_checks": {
            "stage2_rows": audit["stage2"]["total_rows"],
            "prefill_rows": audit["prefill"]["file"]["jsonl_rows"],
            "curvature_action_median": audit["curvature"]["action"]["median"],
            "curvature_action_cells": audit["curvature"]["action"]["cell_medians_at_least_0_1"],
            "matched_transform_rows": audit["matched_band_transform"]["file"]["jsonl_rows"],
            "matched_transform_progress": audit["matched_band_transform"]["condition_median_cell_progress_to_B"]["matched_full"],
            "matched_rollout_rows": audit["matched_band_rollout_screen"]["file"]["jsonl_rows"],
            "matched_rollout_B_first": audit["matched_band_rollout_screen"]["counts"]["B_target_first_touched"]["matched_full"],
            "matched_rollout_success_B": audit["matched_band_rollout_screen"]["counts"]["success_B"]["matched_full"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
