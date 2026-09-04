#!/usr/bin/env python
"""Run the locked donor-free π0.5 repair confirmation and matched controls."""

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

from donor_free_repair import (
    apply_proposal,
    match_early_to_live,
    orthogonal_matched,
    predict_proposal,
    proposal_diagnostics,
    random_matched,
    rescale_like,
    sha256_file,
)
from hooks import Pi05Harness
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite
from run_libero_prompt_conditions import ContactTracker, make_env, target_object_for_prompt


NO_CORRECT_FORWARD_CONDITIONS = {
    "conflict",
    "repair",
    "random_matched",
    "orthogonal_matched",
    "wrong_instruction",
    "early_matched",
}
MATCHED_CONDITIONS = {
    "random_matched",
    "orthogonal_matched",
    "wrong_instruction",
    "early_matched",
}


def action_hash(actions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(actions).view(np.uint8)).hexdigest()


def load_and_validate(config_path: Path, model_path: Path) -> tuple[dict, dict, str]:
    config = json.loads(config_path.read_text())
    model_sha256 = sha256_file(model_path)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    manifest = model["manifest"]
    if manifest["config_sha256"] != sha256_file(config_path):
        raise AssertionError("frozen model was fit with a different config")
    if int(manifest["rank"]) != int(config["rank"]):
        raise AssertionError("model/config rank drift")
    training_edges = set(manifest["training_edges"])
    heldout_edges = {
        f"{int(item['target_task_id'])}<-{int(item['source_task_id'])}"
        for item in config["heldout_pairs"]
    }
    if training_edges & heldout_edges:
        raise AssertionError(f"held-out pair leaked into model: {training_edges & heldout_edges}")
    if set(manifest["heldout_edges"]) != heldout_edges:
        raise AssertionError("model/config held-out edge drift")
    expected_maps = {
        f"{int(layer)}:{name}"
        for layer in config["live_layers"] + config["early_layers"]
        for name in ("k", "v")
    }
    if set(model["maps"]) != expected_maps:
        raise AssertionError("model map set does not cover the frozen layer bands")
    return config, model, model_sha256


def move_model_tensors(model: dict, device: str) -> None:
    for record in model["maps"].values():
        for key, value in list(record.items()):
            if isinstance(value, torch.Tensor):
                record[key] = value.to(device=device)


def lambda_zero_check(
    harness: Pi05Harness,
    suite,
    config: dict,
    model: dict,
    out_dir: Path,
) -> dict:
    sentinel = config["calibration_sentinel"]
    target_task_id = int(sentinel["target_task_id"])
    source_task_id = int(sentinel["source_task_id"])
    init_id = int(sentinel["init_ids"][0])
    seed = int(config["base_seed"]) + 100_000 * target_task_id + 1000 * init_id
    env = make_env(suite, config["suite"], target_task_id)
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.init_state_id = init_id
        observation, _ = env.reset(seed=seed)
        source_prompt = suite.tasks[source_task_id].language
        clean = harness.prefix_forward(harness.build_batch(observation, source_prompt))
        zero = harness.prefix_forward(harness.build_batch(observation, source_prompt))
        proposal = predict_proposal(
            zero,
            model,
            target_task_id,
            source_task_id,
            config["live_layers"],
            int(config["rank"]),
        )
        ratios = apply_proposal(zero, proposal, 0.0)
        noise = harness.make_noise(seed)
        clean_normalized, _ = harness.action_forward(clean, noise)
        zero_normalized, _ = harness.action_forward(zero, noise)
        clean_environment = harness.unnormalize(clean_normalized)
        zero_environment = harness.unnormalize(zero_normalized)
        record = {
            "target_task_id": target_task_id,
            "source_task_id": source_task_id,
            "init_id": init_id,
            "correct_prompt_forwards": 0,
            "normalized_bitwise_equal": bool(torch.equal(clean_normalized, zero_normalized)),
            "environment_bitwise_equal": bool(torch.equal(clean_environment, zero_environment)),
            "max_abs_normalized": float((clean_normalized - zero_normalized).abs().max().item()),
            "max_abs_environment": float((clean_environment - zero_environment).abs().max().item()),
            "edit_norm_ratios": ratios,
        }
        (out_dir / "lambda_zero_check.json").write_text(json.dumps(record, indent=2) + "\n")
        if not record["normalized_bitwise_equal"] or not record["environment_bitwise_equal"]:
            raise AssertionError(f"scale-zero identity failed: {record}")
        return record
    finally:
        env.close()


def intervention_for_condition(
    prefix: dict,
    model: dict,
    config: dict,
    target_task_id: int,
    source_task_id: int,
    wrong_control_target_task_id: int,
    condition: str,
    control_seed: int,
) -> tuple[dict[int, dict[str, torch.Tensor]] | None, dict | None, bool]:
    """Return (applied proposal, diagnostics against live repair, identity-gated)."""
    if condition in {"conflict", "correct"}:
        return None, None, False
    if condition == "preserve_correct":
        return None, None, True
    live = predict_proposal(
        prefix,
        model,
        target_task_id,
        source_task_id,
        config["live_layers"],
        int(config["rank"]),
    )
    if condition == "repair":
        applied = live
        diagnostics = proposal_diagnostics(applied, live)
    elif condition == "random_matched":
        applied = random_matched(live, control_seed)
        diagnostics = proposal_diagnostics(applied, live)
    elif condition == "orthogonal_matched":
        applied = orthogonal_matched(live, control_seed)
        diagnostics = proposal_diagnostics(applied, live)
    elif condition == "wrong_instruction":
        wrong = predict_proposal(
            prefix,
            model,
            wrong_control_target_task_id,
            source_task_id,
            config["live_layers"],
            int(config["rank"]),
        )
        applied = rescale_like(wrong, live)
        diagnostics = proposal_diagnostics(applied, live)
    elif condition == "early_matched":
        early = predict_proposal(
            prefix,
            model,
            target_task_id,
            source_task_id,
            config["early_layers"],
            int(config["rank"]),
        )
        applied = match_early_to_live(early, live, config["early_layers"], config["live_layers"])
        correspondence = {
            int(early_layer): int(live_layer)
            for early_layer, live_layer in zip(config["early_layers"], config["live_layers"], strict=True)
        }
        diagnostics = proposal_diagnostics(applied, live, correspondence, global_norm_match=True)
    else:
        raise ValueError(condition)
    if condition in MATCHED_CONDITIONS and diagnostics["max_relative_norm_error"] > 1e-5:
        raise AssertionError(
            f"{condition} norm mismatch: {diagnostics['max_relative_norm_error']:.3e}"
        )
    return applied, diagnostics, False


def run_episode(
    harness: Pi05Harness,
    suite,
    config: dict,
    model: dict,
    model_sha256: str,
    phase: str,
    pair: dict,
    init_id: int,
    condition: str,
) -> dict:
    target_task_id = int(pair["target_task_id"])
    source_task_id = int(pair["source_task_id"])
    wrong_control_target_task_id = int(pair["wrong_control_target_task_id"])
    uses_correct_prompt = condition in {"correct", "preserve_correct"}
    prompt_task_id = target_task_id if uses_correct_prompt else source_task_id
    # Deliberately access only the prompt used by this arm. Repair paths never construct target-prompt text.
    prompt = suite.tasks[prompt_task_id].language
    env = make_env(suite, config["suite"], target_task_id)
    tracker = ContactTracker(env._env)
    scene_objects = tracker.objects
    correct_object = target_object_for_prompt(config["suite"], suite, target_task_id, scene_objects)
    source_object = target_object_for_prompt(config["suite"], suite, source_task_id, scene_objects)
    wrong_control_object = target_object_for_prompt(
        config["suite"], suite, wrong_control_target_task_id, scene_objects
    )
    if source_object is None or source_object not in scene_objects:
        raise AssertionError(f"source target is absent from task {target_task_id} scene")
    if wrong_control_object is None or wrong_control_object not in scene_objects:
        raise AssertionError(f"wrong-control target is absent from task {target_task_id} scene")
    max_steps = TASK_SUITE_MAX_STEPS.get(config["suite"], 500)
    seed = int(config["base_seed"]) + 100_000 * target_task_id + 1000 * init_id
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
        correct_prompt_forwards = 0
        total_prefix_forwards = 0
        for step in range(max_steps):
            if not queue:
                batch = harness.build_batch(observation, prompt)
                prefix = harness.prefix_forward(batch)
                total_prefix_forwards += 1
                correct_prompt_forwards += int(uses_correct_prompt)
                replan_index = len(replans)
                control_seed = seed + 10_000_000 + 100 * replan_index
                proposal, diagnostics, identity_gated = intervention_for_condition(
                    prefix,
                    model,
                    config,
                    target_task_id,
                    source_task_id,
                    wrong_control_target_task_id,
                    condition,
                    control_seed,
                )
                ratios = [] if proposal is None else apply_proposal(prefix, proposal, float(config["alpha"]))
                flow_seed = seed + replan_index
                noise = harness.make_noise(flow_seed)
                normalized, _ = harness.action_forward(prefix, noise)
                actions = harness.unnormalize(normalized)[0].float().cpu().numpy()
                if actions.shape != (50, 7):
                    raise AssertionError(f"action shape drift: {actions.shape}")
                queue.extend(actions[:10])
                replans.append(
                    {
                        "index": replan_index,
                        "flow_seed": flow_seed,
                        "control_seed": control_seed if condition in MATCHED_CONDITIONS else None,
                        "action_sha256": action_hash(actions),
                        "identity_gated": identity_gated,
                        "edit_to_host_norm_ratio_median": float(np.median(ratios)) if ratios else 0.0,
                        "edit_to_host_norm_ratio_max": float(np.max(ratios)) if ratios else 0.0,
                        "edit_diagnostics": diagnostics,
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
        if condition in NO_CORRECT_FORWARD_CONDITIONS and correct_prompt_forwards != 0:
            raise AssertionError(f"correct-prompt leakage in {condition}")
        if total_prefix_forwards != len(replans):
            raise AssertionError("unexpected extra prefix forward")
        if condition == "preserve_correct" and any(
            replan["edit_to_host_norm_ratio_max"] != 0.0 or not replan["identity_gated"]
            for replan in replans
        ):
            raise AssertionError("correct-prompt preservation gate failed")
        return {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "phase": phase,
            "pair_id": f"{target_task_id}<-{source_task_id}",
            "target_task_id": target_task_id,
            "source_task_id": source_task_id,
            "wrong_control_target_task_id": wrong_control_target_task_id,
            "init_id": init_id,
            "condition": condition,
            "prompt_task_id": prompt_task_id,
            "prompt": prompt,
            "prompt_role": "correct-control-only" if uses_correct_prompt else "source",
            "correct_object": correct_object,
            "source_object": source_object,
            "wrong_control_object": wrong_control_object,
            "success": success,
            "done": bool(done),
            "n_steps": step + 1,
            "first_touch_set": first_touch_set or [],
            "correct_target_first_touched": bool(first_touch_set and correct_object in first_touch_set),
            "source_target_first_touched": bool(first_touch_set and source_object in first_touch_set),
            "touched_any": sorted(touched_any),
            "grasped_any": sorted(grasped_any),
            "contact_timeline": contact_timeline,
            "eef_path": eef_path,
            "replans": replans,
            "total_prefix_forwards": total_prefix_forwards,
            "correct_prompt_forwards": correct_prompt_forwards,
            "repair_path_correct_prompt_leakage": bool(
                condition in NO_CORRECT_FORWARD_CONDITIONS and correct_prompt_forwards
            ),
            "rank": int(config["rank"]),
            "alpha": float(config["alpha"]),
            "model_sha256": model_sha256,
            "seed": seed,
            "wall_seconds": time.time() - started,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration_sentinel", "confirm"), required=True)
    parser.add_argument("--processor-path", default=None)
    args = parser.parse_args()
    config, model, model_sha256 = load_and_validate(args.config, args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    suite = _get_suite(config["suite"])
    harness = Pi05Harness(
        config["policy"],
        revision=config["revision"],
        dtype=config["dtype"],
        processor_path=args.processor_path,
    )
    move_model_tensors(model, harness.device)
    identity = lambda_zero_check(harness, suite, config, model, args.out)
    if args.phase == "confirm":
        pairs = config["heldout_pairs"]
        init_ids = [int(value) for value in config["confirmation_init_ids"]]
        conditions = list(config["confirmation_conditions"])
    else:
        sentinel = config["calibration_sentinel"]
        pairs = [
            {
                "target_task_id": int(sentinel["target_task_id"]),
                "source_task_id": int(sentinel["source_task_id"]),
                "wrong_control_target_task_id": int(sentinel["wrong_control_target_task_id"]),
            }
        ]
        init_ids = [int(value) for value in sentinel["init_ids"]]
        conditions = ["conflict", "correct", "repair", "preserve_correct"]

    output_path = args.out / "episodes.jsonl"
    completed = set()
    rows = []
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["model_sha256"] != model_sha256:
                    raise AssertionError("resume file contains a different model hash")
                rows.append(row)
                completed.add((row["phase"], row["pair_id"], row["init_id"], row["condition"]))

    with output_path.open("a") as output:
        for pair in pairs:
            pair_id = f"{int(pair['target_task_id'])}<-{int(pair['source_task_id'])}"
            for init_id in init_ids:
                for condition in conditions:
                    key = (args.phase, pair_id, init_id, condition)
                    if key in completed:
                        continue
                    row = run_episode(
                        harness,
                        suite,
                        config,
                        model,
                        model_sha256,
                        args.phase,
                        pair,
                        init_id,
                        condition,
                    )
                    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    output.flush()
                    rows.append(row)
                    print(
                        f"[donor-free] phase={args.phase} pair={pair_id} init={init_id} "
                        f"condition={condition} success={int(row['success'])} "
                        f"correct_first={int(row['correct_target_first_touched'])} "
                        f"correct_forwards={row['correct_prompt_forwards']} steps={row['n_steps']} "
                        f"wall={row['wall_seconds']:.1f}s",
                        flush=True,
                    )

    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "phase": args.phase,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "model_path": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "model_frozen_at": model["manifest"]["frozen_at"],
        "training_edges": model["manifest"]["training_edges"],
        "heldout_edges": model["manifest"]["heldout_edges"],
        "identity": identity,
        "n_expected": len(pairs) * len(init_ids) * len(conditions),
        "n_observed": sum(row["phase"] == args.phase for row in rows),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
