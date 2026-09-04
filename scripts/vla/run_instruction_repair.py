#!/usr/bin/env python
"""Closed-loop mechanism-guided repair for π0.5 conflicting-instruction failures."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from hooks import Pi05Harness, cache_kv_lists
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite
from run_libero_prompt_conditions import ContactTracker, make_env, target_object_for_prompt


TASKS = {1: 5, 2: 4}
LAYERS = tuple(range(12, 18))
PILOT_ALPHAS = (0.5, 1.0, 2.0, 4.0)
BASE_SEED = 20260831


def action_hash(actions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(actions).view(np.uint8)).hexdigest()


def make_orthogonal(direction: dict[int, dict[str, torch.Tensor]], task_id: int) -> dict[int, dict[str, torch.Tensor]]:
    result = {}
    for layer in LAYERS:
        result[layer] = {}
        for kv_index, name in enumerate(("k", "v")):
            target = direction[layer][name].float().cpu()
            generator = torch.Generator(device="cpu").manual_seed(
                BASE_SEED + 1000 * task_id + 100 * layer + kv_index
            )
            random = torch.randn(target.shape, generator=generator, dtype=torch.float32)
            denom = target.flatten().dot(target.flatten()).clamp_min(1e-12)
            random.sub_(target * (random.flatten().dot(target.flatten()) / denom))
            random.mul_(target.norm() / random.norm().clamp_min(1e-12))
            result[layer][name] = random
    return result


def rescale_unrelated(
    target: dict[int, dict[str, torch.Tensor]], unrelated: dict[int, dict[str, torch.Tensor]]
) -> dict[int, dict[str, torch.Tensor]]:
    result = {}
    for layer in LAYERS:
        result[layer] = {}
        for name in ("k", "v"):
            source = unrelated[layer][name].float().cpu()
            result[layer][name] = source * (target[layer][name].norm() / source.norm().clamp_min(1e-12))
    return result


def edit_prefix(prefix: dict, direction: dict[int, dict[str, torch.Tensor]], alpha: float) -> list[float]:
    keys, values = cache_kv_lists(prefix["cache"])
    n_img_valid = int(prefix["n_img_valid"])
    ratios = []
    with torch.no_grad():
        for layer in LAYERS:
            for tensor, name in ((keys[layer], "k"), (values[layer], "v")):
                host = tensor[:, :, :n_img_valid]
                delta = direction[layer][name].to(device=host.device, dtype=host.dtype)
                if delta.shape != host.shape:
                    raise AssertionError(f"direction shape {delta.shape} != host shape {host.shape}")
                ratios.append(float(abs(alpha) * delta.norm().item() / host.norm().clamp_min(1e-12).item()))
                host.add_(delta, alpha=float(alpha))
    return ratios


def swap_image_kv_from_donor(prefix: dict, donor: dict, layers: tuple[int, ...]) -> list[float]:
    keys, values = cache_kv_lists(prefix["cache"])
    donor_keys, donor_values = cache_kv_lists(donor["cache"])
    n_img_valid = int(prefix["n_img_valid"])
    if n_img_valid != int(donor["n_img_valid"]):
        raise AssertionError("receiver/donor valid-image count differs")
    ratios = []
    with torch.no_grad():
        for layer in layers:
            for receiver, source in ((keys[layer], donor_keys[layer]), (values[layer], donor_values[layer])):
                host = receiver[:, :, :n_img_valid]
                delta = source[:, :, :n_img_valid] - host
                ratios.append(float(delta.norm().item() / host.norm().clamp_min(1e-12).item()))
                host.copy_(source[:, :, :n_img_valid])
    return ratios


def lambda_zero_check(harness: Pi05Harness, suite, vector_record: dict, out_dir: Path) -> dict:
    task_id = 1
    init_id = 10
    conflict_task_id = TASKS[task_id]
    env = make_env(suite, "libero_object", task_id)
    try:
        seed = BASE_SEED + 100_000 * task_id + 1000 * init_id
        torch.manual_seed(seed)
        env.init_state_id = init_id
        observation, _ = env.reset(seed=seed)
        batch = harness.build_batch(observation, suite.tasks[conflict_task_id].language)
        clean_prefix = harness.prefix_forward(batch)
        zero_prefix = harness.prefix_forward(batch)
        ratios = edit_prefix(zero_prefix, vector_record[task_id]["direction"], 0.0)
        noise = harness.make_noise(seed)
        clean_norm, _ = harness.action_forward(clean_prefix, noise)
        zero_norm, _ = harness.action_forward(zero_prefix, noise)
        clean = harness.unnormalize(clean_norm)
        zero = harness.unnormalize(zero_norm)
        record = {
            "task_id": task_id,
            "init_id": init_id,
            "normalized_bitwise_equal": bool(torch.equal(clean_norm, zero_norm)),
            "environment_bitwise_equal": bool(torch.equal(clean, zero)),
            "max_abs_normalized": float((clean_norm - zero_norm).abs().max().item()),
            "max_abs_environment": float((clean - zero).abs().max().item()),
            "edit_norm_ratios": ratios,
        }
        (out_dir / "lambda_zero_check.json").write_text(json.dumps(record, indent=2) + "\n")
        if not record["normalized_bitwise_equal"] or not record["environment_bitwise_equal"]:
            raise AssertionError(f"lambda-zero identity failed: {record}")
        return record
    finally:
        env.close()


def run_episode(
    harness: Pi05Harness,
    suite,
    task_id: int,
    init_id: int,
    condition: str,
    alpha: float,
    direction: dict[int, dict[str, torch.Tensor]] | None,
    donor_layers: tuple[int, ...] | None = None,
) -> dict:
    conflict_task_id = TASKS[task_id]
    prompt_task_id = task_id if condition == "correct" else conflict_task_id
    prompt = suite.tasks[prompt_task_id].language
    env = make_env(suite, "libero_object", task_id)
    tracker = ContactTracker(env._env)
    scene_objects = tracker.objects
    correct_object = target_object_for_prompt("libero_object", suite, task_id, scene_objects)
    conflict_object = target_object_for_prompt("libero_object", suite, conflict_task_id, scene_objects)
    max_steps = TASK_SUITE_MAX_STEPS.get("libero_object", 500)
    seed = BASE_SEED + 100_000 * task_id + 1000 * init_id
    started = time.time()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.init_state_id = init_id
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
                batch = harness.build_batch(observation, prompt)
                prefix = harness.prefix_forward(batch)
                ratios = []
                if donor_layers is not None:
                    donor_batch = harness.build_batch(observation, suite.tasks[task_id].language)
                    donor = harness.prefix_forward(donor_batch)
                    ratios = swap_image_kv_from_donor(prefix, donor, donor_layers)
                elif direction is not None:
                    ratios = edit_prefix(prefix, direction, alpha)
                replan_index = len(replans)
                flow_seed = seed + replan_index
                noise = harness.make_noise(flow_seed)
                normalized, _ = harness.action_forward(prefix, noise)
                actions = harness.unnormalize(normalized)[0].float().cpu().numpy()
                if actions.shape != (50, 7):
                    raise AssertionError(f"action shape {actions.shape}")
                queue.extend(actions[:10])
                replans.append(
                    {
                        "index": replan_index,
                        "flow_seed": flow_seed,
                        "action_sha256": action_hash(actions),
                        "direction_to_host_norm_ratio_median": float(np.median(ratios)) if ratios else 0.0,
                        "direction_to_host_norm_ratio_max": float(np.max(ratios)) if ratios else 0.0,
                    }
                )
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
        return {
            "schema_version": 1,
            "phase": None,
            "task_id": task_id,
            "init_id": init_id,
            "condition": condition,
            "alpha": float(alpha),
            "prompt_task_id": prompt_task_id,
            "prompt": prompt,
            "correct_object": correct_object,
            "conflict_object": conflict_object,
            "success": success,
            "done": bool(done),
            "n_steps": step + 1,
            "first_touch_set": first_touch_set or [],
            "correct_target_first_touched": bool(first_touch_set and correct_object in first_touch_set),
            "conflict_target_first_touched": bool(first_touch_set and conflict_object in first_touch_set),
            "touched_any": sorted(touched_any),
            "grasped_any": sorted(grasped_any),
            "contact_timeline": contact_timeline,
            "eef_path": eef_path,
            "replans": replans,
            "donor_layers": list(donor_layers) if donor_layers is not None else None,
            "seed": seed,
            "wall_seconds": time.time() - started,
        }
    finally:
        env.close()


def select_alpha(rows: list[dict]) -> dict:
    summary = {}
    selected = None
    for alpha in PILOT_ALPHAS:
        by_task = {}
        passes = True
        for task_id in TASKS:
            group = [
                row
                for row in rows
                if row["condition"] == "repair" and row["task_id"] == task_id and row["alpha"] == alpha
            ]
            if len(group) != 5:
                passes = False
            correct_touch = sum(row["correct_target_first_touched"] for row in group)
            success = sum(row["success"] for row in group)
            coherent = sum(bool(row["touched_any"]) for row in group)
            task_pass = len(group) == 5 and correct_touch >= 3 and success >= 3 and coherent >= 4
            by_task[str(task_id)] = {
                "n": len(group),
                "correct_target_first_touched": correct_touch,
                "success": success,
                "touched_anything": coherent,
                "pass": task_pass,
            }
            passes = passes and task_pass
        summary[str(alpha)] = {"by_task": by_task, "pass": passes}
        if passes and selected is None:
            selected = alpha
    return {"selected_alpha": selected, "alphas": summary, "pilot_pass": selected is not None}


def summarize_state_pilot(rows: list[dict]) -> dict:
    by_task = {}
    overall = True
    for task_id in TASKS:
        live = [row for row in rows if row["task_id"] == task_id and row["condition"] == "state_live"]
        early = [row for row in rows if row["task_id"] == task_id and row["condition"] == "state_early"]
        result = {
            "n_live": len(live),
            "live_success": sum(row["success"] for row in live),
            "live_correct_first": sum(row["correct_target_first_touched"] for row in live),
            "live_touched_anything": sum(bool(row["touched_any"]) for row in live),
            "n_early": len(early),
            "early_success": sum(row["success"] for row in early),
            "early_correct_first": sum(row["correct_target_first_touched"] for row in early),
        }
        result["pass"] = (
            result["n_live"] == 5
            and result["n_early"] == 5
            and result["live_success"] >= 3
            and result["live_correct_first"] >= 4
            and result["live_touched_anything"] == 5
            and result["early_success"] <= 1
            and result["early_correct_first"] <= 1
        )
        overall = overall and result["pass"]
        by_task[str(task_id)] = result
    return {"pilot_pass": overall, "by_task": by_task}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "confirm", "state_pilot", "state_confirm"), required=True)
    parser.add_argument("--selected-alpha", type=float)
    parser.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--revision", default="8e174154ef5f6c60a8da12ae99c303d8963138c1")
    parser.add_argument("--processor-path", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.vectors, map_location="cpu", weights_only=False)
    vectors = payload["vectors"]
    if sorted(vectors) != sorted(TASKS):
        raise AssertionError(f"vector task drift: {sorted(vectors)}")
    suite = _get_suite("libero_object")
    harness = Pi05Harness(
        args.policy,
        revision=args.revision,
        dtype="float32",
        processor_path=args.processor_path,
    )
    identity = lambda_zero_check(harness, suite, vectors, args.out)
    output_path = args.out / "episodes.jsonl"
    rows = []
    completed = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                completed.add((row["phase"], row["task_id"], row["init_id"], row["condition"], row["alpha"]))

    if args.phase == "pilot":
        init_ids = range(10, 15)
        specs = [("conflict", 0.0), ("correct", 0.0)] + [("repair", alpha) for alpha in PILOT_ALPHAS]
    elif args.phase == "confirm":
        if args.selected_alpha not in PILOT_ALPHAS:
            raise ValueError("confirmation requires the pilot-selected alpha")
        init_ids = range(20, 30)
        specs = [
            ("conflict", 0.0),
            ("correct", 0.0),
            ("repair", args.selected_alpha),
            ("orthogonal", args.selected_alpha),
            ("unrelated", args.selected_alpha),
        ]
    elif args.phase == "state_pilot":
        init_ids = range(10, 15)
        specs = [("conflict", 0.0), ("correct", 0.0), ("state_live", 0.0), ("state_early", 0.0)]
    else:
        init_ids = range(20, 30)
        specs = [("conflict", 0.0), ("correct", 0.0), ("state_live", 0.0), ("state_early", 0.0)]

    controls = {}
    for task_id in TASKS:
        other_task = next(candidate for candidate in TASKS if candidate != task_id)
        controls[task_id] = {
            "repair": vectors[task_id]["direction"],
            "orthogonal": make_orthogonal(vectors[task_id]["direction"], task_id),
            "unrelated": rescale_unrelated(vectors[task_id]["direction"], vectors[other_task]["direction"]),
        }

    with output_path.open("a") as output:
        for task_id in TASKS:
            for init_id in init_ids:
                for condition, alpha in specs:
                    key = (args.phase, task_id, init_id, condition, float(alpha))
                    if key in completed:
                        continue
                    direction = controls[task_id].get(condition)
                    donor_layers = LAYERS if condition == "state_live" else tuple(range(0, 6)) if condition == "state_early" else None
                    row = run_episode(
                        harness, suite, task_id, init_id, condition, alpha, direction, donor_layers=donor_layers
                    )
                    row["phase"] = args.phase
                    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    output.flush()
                    rows.append(row)
                    print(
                        f"[repair] phase={args.phase} task={task_id} init={init_id} condition={condition} "
                        f"alpha={alpha:g} success={int(row['success'])} "
                        f"correct_first={int(row['correct_target_first_touched'])} steps={row['n_steps']} "
                        f"wall={row['wall_seconds']:.1f}s",
                        flush=True,
                    )
    manifest = {
        "phase": args.phase,
        "vectors": str(args.vectors),
        "identity": identity,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.phase == "pilot":
        selection = select_alpha([row for row in rows if row["phase"] == "pilot"])
        (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2), flush=True)
    elif args.phase == "state_pilot":
        selection = summarize_state_pilot([row for row in rows if row["phase"] == "state_pilot"])
        (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
