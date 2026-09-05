#!/usr/bin/env python
"""Lean closed-loop side-effect panel for π0.5's broad late-state repair."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from hooks import Pi05Harness
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite
from run_instruction_repair import action_hash, swap_image_kv_from_donor
from run_libero_prompt_conditions import ContactTracker, make_env, target_object_for_prompt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_spec(condition: str, task: dict, late_layers: tuple[int, ...], early_layers: tuple[int, ...]):
    task_id = int(task["task_id"])
    conflict_id = int(task["conflict_task_id"])
    wrong_id = int(task["wrong_donor_task_id"])
    specs = {
        "clean_correct": (task_id, None, None),
        "preserve_correct": (task_id, task_id, late_layers),
        "repair_live": (conflict_id, task_id, late_layers),
        "repair_early": (conflict_id, task_id, early_layers),
        "wrong_donor_live": (conflict_id, wrong_id, late_layers),
    }
    if condition not in specs:
        raise KeyError(condition)
    return specs[condition]


def path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(points, points[1:])))


def run_episode(harness, suite, cfg: dict, task: dict, init_id: int, condition: str) -> dict:
    task_id = int(task["task_id"])
    late_layers = tuple(int(x) for x in cfg["late_layers"])
    early_layers = tuple(int(x) for x in cfg["early_layers"])
    host_prompt_id, donor_prompt_id, donor_layers = condition_spec(condition, task, late_layers, early_layers)
    prompt = suite.tasks[host_prompt_id].language
    env = make_env(suite, cfg["suite"], task_id)
    tracker = ContactTracker(env._env)
    scene_objects = tracker.objects
    correct_object = target_object_for_prompt(cfg["suite"], suite, task_id, scene_objects)
    conflict_object = target_object_for_prompt(cfg["suite"], suite, int(task["conflict_task_id"]), scene_objects)
    wrong_donor_object = target_object_for_prompt(cfg["suite"], suite, int(task["wrong_donor_task_id"]), scene_objects)
    max_steps = int(TASK_SUITE_MAX_STEPS.get(cfg["suite"], 500))
    seed = int(cfg["base_seed"]) + 100_000 * task_id + 1000 * int(init_id)
    started = time.time()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.init_state_id = int(init_id)
        observation, _ = env.reset(seed=seed)
        raw_env = env._env
        queue: collections.deque[np.ndarray] = collections.deque()
        first_touch_set: list[str] | None = None
        touched_any: set[str] = set()
        grasped_any: set[str] = set()
        contact_timeline = []
        eef_path = []
        replans = []
        success = False
        done = False
        step = 0
        for step in range(max_steps):
            if not queue:
                host = harness.prefix_forward(harness.build_batch(observation, prompt))
                ratios = []
                if donor_prompt_id is not None:
                    donor_prompt = suite.tasks[donor_prompt_id].language
                    donor = harness.prefix_forward(harness.build_batch(observation, donor_prompt))
                    ratios = swap_image_kv_from_donor(host, donor, donor_layers)
                replan_index = len(replans)
                flow_seed = seed + replan_index
                normalized, _ = harness.action_forward(host, harness.make_noise(flow_seed))
                actions = harness.unnormalize(normalized)[0].float().cpu().numpy()
                if actions.shape != (50, 7):
                    raise AssertionError(f"action shape {actions.shape}")
                queue.extend(actions[: int(cfg["action_execution_horizon"])])
                replans.append({
                    "index": replan_index,
                    "flow_seed": flow_seed,
                    "action_sha256": action_hash(actions),
                    "donor_to_host_norm_ratio_median": float(np.median(ratios)) if ratios else 0.0,
                    "donor_to_host_norm_ratio_max": float(np.max(ratios)) if ratios else 0.0,
                })
            action = np.asarray(queue.popleft(), dtype=np.float32)
            raw_observation, _, done, _ = raw_env.step(action)
            observation = env._format_raw_obs(raw_observation)
            eef_path.append([float(value) for value in raw_observation["robot0_eef_pos"]])
            touched, grasped = tracker.contacts()
            touched_sorted, grasped_sorted = sorted(touched), sorted(grasped)
            if touched_sorted:
                contact_timeline.append([step, touched_sorted, grasped_sorted])
                if first_touch_set is None:
                    first_touch_set = touched_sorted
            touched_any.update(touched)
            grasped_any.update(grasped)
            success = bool(raw_env.check_success())
            if success or done:
                break
        first_touch = first_touch_set or []
        non_target_touched = sorted(set(touched_any) - {correct_object})
        non_target_grasped = sorted(set(grasped_any) - {correct_object})
        first_correct_step = next(
            (int(item[0]) for item in contact_timeline if correct_object in item[1]), None
        )
        return {
            "schema_version": 1,
            "task_id": task_id,
            "init_id": int(init_id),
            "condition": condition,
            "host_prompt_task_id": host_prompt_id,
            "donor_prompt_task_id": donor_prompt_id,
            "prompt": prompt,
            "correct_object": correct_object,
            "conflict_object": conflict_object,
            "wrong_donor_object": wrong_donor_object,
            "success": bool(success),
            "done": bool(done),
            "n_steps": step + 1,
            "first_touch_set": first_touch,
            "correct_target_first_touched": bool(correct_object in first_touch),
            "non_target_first_touched": bool(first_touch and correct_object not in first_touch),
            "conflict_target_first_touched": bool(conflict_object in first_touch),
            "wrong_donor_target_first_touched": bool(wrong_donor_object in first_touch),
            "first_correct_touch_step": first_correct_step,
            "touched_any": sorted(touched_any),
            "grasped_any": sorted(grasped_any),
            "non_target_touched": non_target_touched,
            "non_target_grasped": non_target_grasped,
            "n_non_target_touched": len(non_target_touched),
            "any_non_target_grasp": bool(non_target_grasped),
            "contact_timeline": contact_timeline,
            "eef_path_length": path_length(eef_path),
            "replans": replans,
            "donor_layers": list(donor_layers) if donor_layers is not None else None,
            "seed": seed,
            "wall_seconds": time.time() - started,
        }
    finally:
        env.close()


def exact_mcnemar(values_a: list[bool], values_b: list[bool]) -> dict:
    positive = sum((not a) and b for a, b in zip(values_a, values_b))
    negative = sum(a and (not b) for a, b in zip(values_a, values_b))
    n = positive + negative
    if n == 0:
        p_value = 1.0
    else:
        low = min(positive, negative)
        p_value = min(1.0, 2 * sum(math.comb(n, k) for k in range(low + 1)) / (2**n))
    return {"positive": positive, "negative": negative, "discordant": n, "p_two_sided": p_value}


def summarize(rows: list[dict]) -> dict:
    conditions = sorted({row["condition"] for row in rows})
    metrics = (
        "success",
        "correct_target_first_touched",
        "non_target_first_touched",
        "any_non_target_grasp",
    )
    counts = {
        condition: {
            "n": sum(row["condition"] == condition for row in rows),
            **{
                metric: sum(bool(row[metric]) for row in rows if row["condition"] == condition)
                for metric in metrics
            },
            "median_steps": statistics.median(
                row["n_steps"] for row in rows if row["condition"] == condition
            ),
            "median_eef_path_length": statistics.median(
                row["eef_path_length"] for row in rows if row["condition"] == condition
            ),
            "median_n_non_target_touched": statistics.median(
                row["n_non_target_touched"] for row in rows if row["condition"] == condition
            ),
        }
        for condition in conditions
    }
    by_key = {(int(row["task_id"]), int(row["init_id"]), row["condition"]): row for row in rows}
    units = sorted({(int(row["task_id"]), int(row["init_id"])) for row in rows})
    paired = {}
    for metric in metrics:
        clean = [bool(by_key[(*unit, "clean_correct")][metric]) for unit in units]
        live = [bool(by_key[(*unit, "repair_live")][metric]) for unit in units]
        paired[metric] = {
            "repair_live_minus_clean_correct": [int(b) - int(a) for a, b in zip(clean, live)],
            "exact_mcnemar": exact_mcnemar(clean, live),
        }
    preserve_exact = []
    for unit in units:
        clean = by_key[(*unit, "clean_correct")]
        preserve = by_key[(*unit, "preserve_correct")]
        preserve_exact.append({
            "task_id": unit[0],
            "init_id": unit[1],
            "action_hashes_equal": [x["action_sha256"] for x in clean["replans"]]
            == [x["action_sha256"] for x in preserve["replans"]],
            "outcomes_equal": all(
                clean[field] == preserve[field]
                for field in ("success", "n_steps", "first_touch_set", "touched_any", "grasped_any")
            ),
            "all_edit_ratios_zero": all(
                float(x["donor_to_host_norm_ratio_max"]) == 0.0 for x in preserve["replans"]
            ),
        })
    return {
        "schema_version": 1,
        "rows": len(rows),
        "independent_units": len(units),
        "counts": counts,
        "paired_repair_live_vs_clean_correct": paired,
        "preserve_correct": preserve_exact,
        "preserve_all_exact": all(
            item["action_hashes_equal"] and item["outcomes_equal"] and item["all_edit_ratios_zero"]
            for item in preserve_exact
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--processor-path", default=None)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    suite = _get_suite(cfg["suite"])
    harness = Pi05Harness(
        cfg["policy"], revision=cfg["revision"], dtype="float32", processor_path=args.processor_path
    )
    output_path = args.out / "episodes.jsonl"
    rows = []
    completed = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                completed.add((int(row["task_id"]), int(row["init_id"]), row["condition"]))
    with output_path.open("a") as output:
        for task in cfg["tasks"]:
            for init_id in cfg["initial_states"]:
                for condition in cfg["conditions"]:
                    key = (int(task["task_id"]), int(init_id), condition)
                    if key in completed:
                        continue
                    row = run_episode(harness, suite, cfg, task, int(init_id), condition)
                    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    output.flush()
                    rows.append(row)
                    print(
                        f"[side-effects] task={key[0]} init={key[1]} condition={condition} "
                        f"success={int(row['success'])} correct_first={int(row['correct_target_first_touched'])} "
                        f"wrong_first={int(row['non_target_first_touched'])} steps={row['n_steps']} "
                        f"wall={row['wall_seconds']:.1f}s",
                        flush=True,
                    )
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tracked = [
        Path(__file__).resolve(),
        args.config.resolve(),
        (Path(__file__).resolve().parent / "run_instruction_repair.py"),
        (Path(__file__).resolve().parent / "hooks.py"),
        (Path(__file__).resolve().parents[2] / "docs/PREREG-pi05-state-repair-side-effects-2026-09-04.md"),
        (Path(__file__).resolve().parents[2] / "tests/test_pi05_state_repair_side_effects.py"),
    ]
    manifest = {
        "schema_version": 1,
        "config": cfg,
        "files": {str(path): sha256(path) for path in tracked},
        "rows_sha256": sha256(output_path),
        "summary_sha256": sha256(args.out / "summary.json"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
