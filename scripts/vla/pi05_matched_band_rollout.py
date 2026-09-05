#!/usr/bin/env python
"""Run the fixed pi0.5 layer-6--8 transformation through LIBERO closed loop."""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch

from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite
from pi05_attention_resolution import cell_name, load_pairs, make_env, require_free_space, sha256, task_object, write_json
from pi05_matched_band_transform import (
    bootstrap_median_interval,
    exact_two_sided_sign_test,
    measure_messages,
    prefix_run,
    random_matched,
    validate_pair_layout,
)
from hooks import Pi05Harness
from run_libero_prompt_conditions import ContactTracker


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def existing_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int, str]] = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        keys.add((row["cell"], int(row["init"]), row["condition"]))
    return keys


def action_hash(actions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(actions, dtype=np.float32).tobytes()).hexdigest()


def episode_seed(base: int, task_a: int, task_b: int, init_id: int) -> int:
    return int(base + 10_000 * task_a + 1_000 * task_b + init_id)


def intervention_prefix(
    harness: Pi05Harness,
    batch_a: dict[str, Any],
    batch_b: dict[str, Any],
    fixed: dict[str, Any],
    condition: str,
    *,
    random_seed: int,
) -> tuple[dict[str, Any], float]:
    if condition == "clean_A":
        return harness.prefix_forward(batch_a), 0.0
    if condition == "clean_B":
        return harness.prefix_forward(batch_b), 0.0
    if condition not in {"matched_full", "matched_attention_only", "random_norm_matched"}:
        raise KeyError(condition)

    messages = measure_messages(harness, batch_a, batch_b, fixed)
    if condition == "matched_full":
        message = messages["full"]
    elif condition == "matched_attention_only":
        message = messages["attention_only"]
    elif condition == "random_norm_matched":
        message = random_matched(messages["full"], random_seed)
    prefix, _, _ = prefix_run(
        harness,
        batch_a,
        input_layer=int(fixed["input_residual_layer"]),
        output_layer=int(fixed["intervention_residual_layer"]),
        band_layers=[int(value) for value in fixed["band_decoder_layers"]],
        n_image=int(fixed["image_positions"]),
        output_delta=message,
    )
    return prefix, float(torch.linalg.vector_norm(message))


def run_episode(
    harness: Pi05Harness,
    suite: Any,
    suite_name: str,
    prompts: list[str],
    task_a: int,
    task_b: int,
    init_id: int,
    condition: str,
    fixed: dict[str, Any],
) -> dict[str, Any]:
    env = make_env(suite, suite_name, task_b)
    tracker = ContactTracker(env._env)
    target_a, _ = task_object(suite_name, task_a)
    target_b, _ = task_object(suite_name, task_b)
    seed = episode_seed(int(fixed["seed_base"]), task_a, task_b, init_id)
    max_steps = int(TASK_SUITE_MAX_STEPS.get(suite_name, 500))
    actions_per_replan = int(fixed["actions_per_replan"])
    queue: collections.deque[np.ndarray] = collections.deque()
    first_touch_set: list[str] | None = None
    touched_any: set[str] = set()
    grasped_any: set[str] = set()
    replans: list[dict[str, Any]] = []
    contact_timeline: list[list[Any]] = []
    success = False
    done = False
    started = time.time()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.init_state_id = init_id
        observation, _ = env.reset(seed=seed)
        raw_env = env._env
        for step in range(max_steps):
            if not queue:
                batch_a = harness.build_batch(observation, prompts[task_a])
                batch_b = harness.build_batch(observation, prompts[task_b])
                if not replans:
                    validate_pair_layout(
                        harness,
                        batch_a,
                        batch_b,
                        n_slots=int(fixed["image_slots"]),
                        n_image=int(fixed["image_positions"]),
                    )
                replan_index = len(replans)
                flow_seed = seed + replan_index
                prefix, message_norm = intervention_prefix(
                    harness,
                    batch_a,
                    batch_b,
                    fixed,
                    condition,
                    random_seed=int(hashlib.sha256(f"{task_a}:{task_b}:{init_id}:{replan_index}".encode()).hexdigest()[:8], 16),
                )
                noise = harness.make_noise(flow_seed)
                normalized, _ = harness.action_forward(prefix, noise)
                actions = harness.unnormalize(normalized)[0].float().cpu().numpy()
                if actions.shape != (50, 7):
                    raise RuntimeError(f"unexpected action shape {actions.shape}")
                queue.extend(actions[:actions_per_replan])
                replans.append({
                    "index": replan_index,
                    "flow_seed": flow_seed,
                    "message_frobenius_norm": message_norm,
                    "action_sha256": action_hash(actions[:actions_per_replan]),
                })
                del batch_a, batch_b, prefix, noise, normalized, actions
                gc.collect()

            action = np.asarray(queue.popleft(), dtype=np.float32)
            raw_observation, _, done, _ = raw_env.step(action)
            observation = env._format_raw_obs(raw_observation)
            touched, grasped = tracker.contacts()
            if touched:
                touched_sorted = sorted(touched)
                grasped_sorted = sorted(grasped)
                contact_timeline.append([step, touched_sorted, grasped_sorted])
                if first_touch_set is None:
                    first_touch_set = touched_sorted
            touched_any.update(touched)
            grasped_any.update(grasped)
            success = bool(raw_env.check_success())
            if success or done:
                break
        return {
            "schema_version": 1,
            "cell": cell_name(suite_name, task_a, task_b),
            "task_a": task_a,
            "task_b": task_b,
            "init": init_id,
            "condition": condition,
            "prompt": prompts[task_a] if condition != "clean_B" else prompts[task_b],
            "target_a": target_a,
            "target_b": target_b,
            "success_B": success,
            "done": bool(done),
            "n_steps": step + 1,
            "first_touch_set": first_touch_set or [],
            "B_target_first_touched": bool(first_touch_set and target_b in first_touch_set),
            "A_target_first_touched": bool(first_touch_set and target_a in first_touch_set),
            "touched_anything": bool(touched_any),
            "touched_any": sorted(touched_any),
            "grasped_any": sorted(grasped_any),
            "contact_timeline": contact_timeline,
            "n_replans": len(replans),
            "replans": replans,
            "seed": seed,
            "wall_seconds": time.time() - started,
        }
    finally:
        env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def median(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.median(values)) if values else float("nan")


def analyze(rows_path: Path, cfg: dict[str, Any], pairs: list[tuple[int, int]], init_ids: list[int]) -> dict[str, Any]:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    conditions = list(cfg["conditions"])
    expected = len(pairs) * len(init_ids) * len(conditions)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete rollout panel: {len(rows)} != {expected}")
    cells = sorted({row["cell"] for row in rows})
    outcomes = ("success_B", "B_target_first_touched", "A_target_first_touched", "touched_anything")
    cell_values: dict[str, dict[str, dict[str, float]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        cell_values[outcome] = {}
        counts[outcome] = {}
        for condition in conditions:
            subset = [row for row in rows if row["condition"] == condition]
            counts[outcome][condition] = sum(bool(row[outcome]) for row in subset)
            cell_values[outcome][condition] = {
                cell: float(np.mean([bool(row[outcome]) for row in subset if row["cell"] == cell]))
                for cell in cells
            }

    comparisons = {
        "matched_full_minus_clean_A": ("matched_full", "clean_A"),
        "matched_full_minus_random_norm_matched": ("matched_full", "random_norm_matched"),
        "matched_full_minus_matched_attention_only": ("matched_full", "matched_attention_only"),
    }
    comparison_summary: dict[str, Any] = {}
    analysis_cfg = cfg["analysis"]
    for outcome_index, outcome in enumerate(("success_B", "B_target_first_touched")):
        comparison_summary[outcome] = {}
        for comparison_index, (name, (left, right)) in enumerate(comparisons.items()):
            if left not in conditions or right not in conditions:
                continue
            differences = [cell_values[outcome][left][cell] - cell_values[outcome][right][cell] for cell in cells]
            comparison_summary[outcome][name] = {
                "median_cell_difference": median(differences),
                "mean_cell_difference": float(np.mean(differences)),
                "bootstrap_95pct_interval_for_median": bootstrap_median_interval(
                    differences,
                    resamples=int(analysis_cfg["bootstrap_resamples"]),
                    seed=int(analysis_cfg["bootstrap_seed"]) + 10 * outcome_index + comparison_index,
                ),
                "exact_two_sided_sign_test_nonzero_cells": exact_two_sided_sign_test(differences),
                "cell_differences": dict(zip(cells, differences)),
            }
    return {
        "schema_version": 1,
        "rows": len(rows),
        "episodes_per_condition": len(pairs) * len(init_ids),
        "cells": cells,
        "counts": counts,
        "cell_rates": cell_values,
        "comparisons": comparison_summary,
        "median_steps": {
            condition: median(row["n_steps"] for row in rows if row["condition"] == condition)
            for condition in conditions
        },
    }


def ensure_manifest(args: argparse.Namespace, cfg: dict[str, Any], smoke: bool) -> None:
    root = Path(__file__).resolve().parents[2]
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "smoke" if smoke else "full",
        "config": cfg,
        "hashes": {
            "runner": sha256(Path(__file__)),
            "config": sha256(args.config),
            "preregistration": sha256(args.preregistration),
            "pairs": sha256(Path(cfg["pairs_file"])),
            "transform_runner": sha256(root / "scripts/vla/pi05_matched_band_transform.py"),
            "hooks": sha256(root / "scripts/vla/hooks.py"),
        },
    }
    path = args.out / "manifest.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["mode"] != payload["mode"] or prior["config"] != cfg or prior["hashes"] != payload["hashes"]:
            raise RuntimeError("sealed rollout inputs changed; use a new output directory")
    else:
        write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    pairs = load_pairs(Path(cfg["pairs_file"]))
    init_ids = [int(value) for value in cfg["eval_init_ids"]]
    conditions = list(cfg["conditions"])
    if args.smoke:
        pairs = pairs[:1]
        init_ids = init_ids[:1]
        conditions = conditions[:3]
    args.out.mkdir(parents=True, exist_ok=True)
    require_free_space(args.out, args.minimum_free_gib)
    ensure_manifest(args, cfg, args.smoke)

    suite_name = cfg["suite"]
    suite = _get_suite(suite_name)
    prompts = [task.language for task in suite.tasks]
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    rows_path = args.out / "episodes.jsonl"
    completed = existing_keys(rows_path)
    for pair_index, (task_a, task_b) in enumerate(pairs):
        cell = cell_name(suite_name, task_a, task_b)
        for init_id in init_ids:
            for condition in conditions:
                key = (cell, init_id, condition)
                if key in completed:
                    continue
                row = run_episode(harness, suite, suite_name, prompts, task_a, task_b, init_id, condition, cfg["fixed"])
                append_row(rows_path, row)
                completed.add(key)
                print(
                    f"[rollout] pair={pair_index + 1}/{len(pairs)} cell={cell} init={init_id} "
                    f"condition={condition} success_B={int(row['success_B'])} "
                    f"B_first={int(row['B_target_first_touched'])} A_first={int(row['A_target_first_touched'])} "
                    f"steps={row['n_steps']} wall={row['wall_seconds']:.1f}s",
                    flush=True,
                )
    if args.smoke:
        print(json.dumps({"smoke_rows": len(completed)}, indent=2), flush=True)
        return
    summary = analyze(rows_path, cfg, pairs, init_ids)
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
