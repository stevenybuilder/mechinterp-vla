#!/usr/bin/env python
"""Fit and freeze the calibration-only low-rank π0.5 cache translator."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from donor_free_repair import KV_NAMES, cache_kv_lists, map_key, sha256_file, signed_task_code
from hooks import Pi05Harness
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite
from run_libero_prompt_conditions import ContactTracker, make_env, target_object_for_prompt


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise AssertionError("unsupported donor-free repair config")
    calibration = set(int(value) for value in config["calibration_init_ids"])
    confirmation = set(int(value) for value in config["confirmation_init_ids"])
    if calibration & confirmation:
        raise AssertionError("calibration and confirmation scene splits overlap")
    training_edges = {
        (int(target), int(source))
        for target, sources in config["training_edges"].items()
        for source in sources
    }
    heldout_edges = {
        (int(item["target_task_id"]), int(item["source_task_id"]))
        for item in config["heldout_pairs"]
    }
    if training_edges & heldout_edges:
        raise AssertionError(f"held-out edge leaked into fit: {training_edges & heldout_edges}")
    if len(heldout_edges) < 5:
        raise AssertionError("fewer than five held-out task/prompt pairs")
    return config


def new_stats(feature_dim: int, output_dim: int, device: str) -> dict:
    return {
        "n": 0,
        "sum_feature": torch.zeros(feature_dim, device=device, dtype=torch.float32),
        "sum_delta": torch.zeros(output_dim, device=device, dtype=torch.float32),
        "feature_cross": torch.zeros(feature_dim, feature_dim, device=device, dtype=torch.float32),
        "feature_delta_cross": torch.zeros(feature_dim, output_dim, device=device, dtype=torch.float32),
        "delta_square_sum": torch.zeros((), device=device, dtype=torch.float32),
    }


def update_stats(stats: dict, features: torch.Tensor, delta: torch.Tensor) -> None:
    features = features.float()
    delta = delta.float()
    if features.ndim != 2 or delta.ndim != 2 or features.shape[0] != delta.shape[0]:
        raise AssertionError((features.shape, delta.shape))
    stats["n"] += int(features.shape[0])
    stats["sum_feature"].add_(features.sum(dim=0))
    stats["sum_delta"].add_(delta.sum(dim=0))
    stats["feature_cross"].add_(features.T @ features)
    stats["feature_delta_cross"].add_(features.T @ delta)
    stats["delta_square_sum"].add_(delta.square().sum())


def add_cache_pair(
    all_stats: dict[str, dict],
    correct: dict,
    source: dict,
    target_task_id: int,
    source_task_id: int,
    config: dict,
    layers: tuple[int, ...],
) -> int:
    if int(correct["n_img_valid"]) != int(source["n_img_valid"]):
        raise AssertionError("correct/source valid-image counts differ")
    n_img_valid = int(source["n_img_valid"])
    stride = int(config["image_position_stride"])
    positions = torch.arange(0, n_img_valid, stride, device=source["prefix_embs"].device)
    correct_keys, correct_values = cache_kv_lists(correct["cache"])
    source_keys, source_values = cache_kv_lists(source["cache"])
    num_task_codes = int(config["num_task_codes"])
    rows_added = None
    for layer in layers:
        for name, target_tensor, source_tensor in (
            ("k", correct_keys[layer], source_keys[layer]),
            ("v", correct_values[layer], source_values[layer]),
        ):
            target_rows = target_tensor[:, :, positions, :].float().reshape(-1, target_tensor.shape[-1])
            source_rows = source_tensor[:, :, positions, :].float().reshape(-1, source_tensor.shape[-1])
            delta = target_rows - source_rows
            code = signed_task_code(
                target_task_id,
                source_task_id,
                num_task_codes,
                device=source_rows.device,
                dtype=source_rows.dtype,
            ).expand(source_rows.shape[0], -1)
            features = torch.cat((source_rows, code), dim=-1)
            key = map_key(layer, name)
            if key not in all_stats:
                all_stats[key] = new_stats(features.shape[-1], delta.shape[-1], str(features.device))
            update_stats(all_stats[key], features, delta)
            rows_added = int(features.shape[0]) if rows_added is None else rows_added
            if int(features.shape[0]) != rows_added:
                raise AssertionError("row-count drift across cache tensors")
    return int(rows_added or 0)


def fit_low_rank(stats: dict, ridge: float, max_rank: int) -> tuple[dict, dict]:
    n = int(stats["n"])
    if n <= 1:
        raise AssertionError(f"insufficient fit rows: {n}")
    sum_feature = stats["sum_feature"].double().cpu()
    sum_delta = stats["sum_delta"].double().cpu()
    mean_feature = sum_feature / n
    mean_delta = sum_delta / n
    feature_cov = stats["feature_cross"].double().cpu() - torch.outer(sum_feature, sum_feature) / n
    feature_delta_cov = (
        stats["feature_delta_cross"].double().cpu() - torch.outer(sum_feature, sum_delta) / n
    )
    variance = feature_cov.diag().div(n).clamp_min(1e-12)
    scale = variance.sqrt().clamp_min(1e-6)
    normalized_cov = feature_cov / torch.outer(scale, scale)
    normalized_cross = feature_delta_cov / scale[:, None]
    regularized = normalized_cov + float(ridge) * n * torch.eye(normalized_cov.shape[0], dtype=torch.float64)
    weight = torch.linalg.solve(regularized, normalized_cross)
    u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    kept_rank = min(int(max_rank), int(singular.numel()))
    total_spectral_energy = float(singular.square().sum().item())
    kept_spectral_energy = float(singular[:kept_rank].square().sum().item())
    record = {
        "n_rows": n,
        "feature_dim": int(weight.shape[0]),
        "output_dim": int(weight.shape[1]),
        "max_rank": kept_rank,
        "mean_feature": mean_feature.float(),
        "scale_feature": scale.float(),
        "mean_delta": mean_delta.float(),
        "u": u[:, :kept_rank].float(),
        "s": singular[:kept_rank].float(),
        "vh": vh[:kept_rank].float(),
    }
    diagnostics = {
        "n_rows": n,
        "feature_dim": int(weight.shape[0]),
        "output_dim": int(weight.shape[1]),
        "rank": kept_rank,
        "ridge": float(ridge),
        "minimum_feature_scale": float(scale.min().item()),
        "maximum_feature_scale": float(scale.max().item()),
        "leading_singular_values": [float(value) for value in singular[: min(16, len(singular))]],
        "rank_kept_spectral_energy_fraction": (
            kept_spectral_energy / total_spectral_energy if total_spectral_energy else 0.0
        ),
    }
    return record, diagnostics


def run_calibration_episode(
    harness: Pi05Harness,
    suite,
    target_task_id: int,
    source_task_ids: list[int],
    init_id: int,
    config: dict,
    all_stats: dict[str, dict],
) -> dict:
    env = make_env(suite, config["suite"], target_task_id)
    tracker = ContactTracker(env._env)
    scene_objects = tracker.objects
    for source_task_id in source_task_ids:
        source_object = target_object_for_prompt(config["suite"], suite, source_task_id, scene_objects)
        if source_object is None or source_object not in scene_objects:
            raise AssertionError(
                f"calibration source task {source_task_id} is absent from target {target_task_id} scene"
            )
    max_steps = TASK_SUITE_MAX_STEPS.get(config["suite"], 500)
    base_seed = int(config["base_seed"])
    seed = base_seed + 100_000 * target_task_id + 1000 * init_id
    live_layers = tuple(int(value) for value in config["live_layers"])
    early_layers = tuple(int(value) for value in config["early_layers"])
    layers = tuple(sorted(set(live_layers + early_layers)))
    started = time.time()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.init_state_id = init_id
        observation, _ = env.reset(seed=seed)
        raw_env = env._env
        queue: collections.deque[np.ndarray] = collections.deque()
        success = False
        done = False
        step = 0
        replans = 0
        sampled_rows_per_map = 0
        while step < max_steps and replans < int(config["calibration_max_replans"]):
            if not queue:
                correct_batch = harness.build_batch(observation, suite.tasks[target_task_id].language)
                correct = harness.prefix_forward(correct_batch)
                for source_task_id in source_task_ids:
                    source_batch = harness.build_batch(observation, suite.tasks[source_task_id].language)
                    source = harness.prefix_forward(source_batch)
                    sampled_rows_per_map += add_cache_pair(
                        all_stats,
                        correct,
                        source,
                        target_task_id,
                        source_task_id,
                        config,
                        layers,
                    )
                    del source
                flow_seed = seed + replans
                noise = harness.make_noise(flow_seed)
                normalized, _ = harness.action_forward(correct, noise)
                actions = harness.unnormalize(normalized)[0].float().cpu().numpy()
                if actions.shape != (50, 7):
                    raise AssertionError(f"action shape drift: {actions.shape}")
                queue.extend(actions[:10])
                replans += 1
                del correct
            action = np.asarray(queue.popleft(), dtype=np.float32)
            raw_observation, _, done, _ = raw_env.step(action)
            observation = env._format_raw_obs(raw_observation)
            step += 1
            success = bool(raw_env.check_success())
            if success or done:
                break
        return {
            "target_task_id": target_task_id,
            "source_task_ids": source_task_ids,
            "init_id": init_id,
            "seed": seed,
            "replans": replans,
            "steps": step,
            "success": success,
            "done": bool(done),
            "sampled_rows_per_map": sampled_rows_per_map,
            "wall_seconds": time.time() - started,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--processor-path", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "calibration_episodes.jsonl"
    if rows_path.exists() or args.out_model.exists():
        raise FileExistsError("calibration outputs already exist; use a new output path to preserve the frozen fit")

    harness = Pi05Harness(
        config["policy"],
        revision=config["revision"],
        dtype=config["dtype"],
        processor_path=args.processor_path,
    )
    suite = _get_suite(config["suite"])
    all_stats: dict[str, dict] = {}
    calibration_rows = []
    with rows_path.open("x") as output:
        for target_text, source_values in config["training_edges"].items():
            target_task_id = int(target_text)
            source_task_ids = [int(value) for value in source_values]
            for init_id in config["calibration_init_ids"]:
                row = run_calibration_episode(
                    harness,
                    suite,
                    target_task_id,
                    source_task_ids,
                    int(init_id),
                    config,
                    all_stats,
                )
                row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                calibration_rows.append(row)
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                output.flush()
                print(
                    f"[calibration] target={target_task_id} init={init_id} replans={row['replans']} "
                    f"success={int(row['success'])} rows/map={row['sampled_rows_per_map']} "
                    f"wall={row['wall_seconds']:.1f}s",
                    flush=True,
                )

    maps = {}
    fit_diagnostics = {}
    required_keys = {
        map_key(layer, name)
        for layer in config["live_layers"] + config["early_layers"]
        for name in KV_NAMES
    }
    if set(all_stats) != required_keys:
        raise AssertionError(f"fitted map keys drift: {sorted(all_stats)} != {sorted(required_keys)}")
    for key in sorted(all_stats):
        maps[key], fit_diagnostics[key] = fit_low_rank(
            all_stats[key], float(config["ridge"]), int(config["rank"])
        )

    training_edges = sorted(
        f"{int(target)}<-{int(source)}"
        for target, sources in config["training_edges"].items()
        for source in sources
    )
    heldout_edges = sorted(
        f"{int(item['target_task_id'])}<-{int(item['source_task_id'])}"
        for item in config["heldout_pairs"]
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "policy": config["policy"],
        "revision": config["revision"],
        "dtype": config["dtype"],
        "suite": config["suite"],
        "num_task_codes": int(config["num_task_codes"]),
        "rank": int(config["rank"]),
        "ridge": float(config["ridge"]),
        "training_edges": training_edges,
        "heldout_edges": heldout_edges,
        "calibration_init_ids": [int(value) for value in config["calibration_init_ids"]],
        "n_calibration_episodes": len(calibration_rows),
        "fit_diagnostics": fit_diagnostics,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload = {"schema_version": 1, "manifest": manifest, "maps": maps}
    torch.save(payload, args.out_model)
    model_sha256 = sha256_file(args.out_model)
    manifest_file = {**manifest, "model_path": str(args.out_model.resolve()), "model_sha256": model_sha256}
    (args.out_dir / "model_manifest.json").write_text(json.dumps(manifest_file, indent=2) + "\n")
    print(json.dumps(manifest_file, indent=2), flush=True)


if __name__ == "__main__":
    main()
