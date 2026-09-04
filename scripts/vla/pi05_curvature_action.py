#!/usr/bin/env python
"""Remove and restore layer-6--8 midpoint curvature, then measure pi0.5 actions."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch

from hooks import Pi05Harness
from lerobot.envs.libero import _get_suite
from pi05_attention_resolution import (
    action_chunk,
    cell_name,
    load_pairs,
    make_env,
    median,
    prefix_equal,
    require_free_space,
    sha256,
    task_object,
    write_json,
)


def relative_action_change(
    native: np.ndarray,
    linearized: np.ndarray,
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    steps: int,
) -> float:
    denominator = float(np.linalg.norm(endpoint_b[:steps] - endpoint_a[:steps]))
    if denominator <= 1e-9:
        raise RuntimeError("interpolated image endpoints have collapsed actions")
    return float(np.linalg.norm(linearized[:steps] - native[:steps]) / denominator)


def ensure_manifest(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    path = args.out / "manifest.json"
    root = Path(__file__).resolve().parents[2]
    hashes = {
        "runner": sha256(Path(__file__)),
        "config": sha256(args.config),
        "preregistration": sha256(args.preregistration),
        "pairs": sha256(Path(cfg["pairs_file"])),
        "hooks": sha256(root / "scripts/vla/hooks.py"),
    }
    if path.exists():
        current = json.loads(path.read_text())
        if current["hashes"] != hashes or current["config"] != cfg:
            raise RuntimeError("sealed causal-test inputs changed; use a new output directory")
        return
    write_json(path, {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg,
        "hashes": hashes,
    })


def prefix_run(
    harness: Pi05Harness,
    batch: dict[str, Any],
    *,
    input_layer: int,
    output_layer: int,
    n_image: int,
    input_image: torch.Tensor | None = None,
    output_image: torch.Tensor | None = None,
    capture_layers: Iterable[int] = (),
) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    wanted = set(int(layer) for layer in capture_layers)
    active = set(wanted)
    if input_image is not None:
        active.add(input_layer)
    if output_image is not None:
        active.add(output_layer)
    captured: dict[int, torch.Tensor] = {}

    def hook_for(layer: int):
        def hook(hidden: torch.Tensor) -> torch.Tensor:
            result = hidden
            replacement = input_image if layer == input_layer else output_image if layer == output_layer else None
            if replacement is not None:
                result = hidden.clone()
                result[:, :n_image] = replacement.to(device=hidden.device, dtype=hidden.dtype)
            if layer in wanted:
                captured[layer] = result[:, :n_image].detach().float().cpu().clone()
            return result
        return hook

    prefix = harness.prefix_forward(batch, resid_hooks={layer: hook_for(layer) for layer in active})
    if set(captured) != wanted:
        raise RuntimeError(f"missing captures: {sorted(wanted - set(captured))}")
    return prefix, captured


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def existing(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {
        (row["cell"], int(row["init"]))
        for row in (json.loads(line) for line in path.read_text().splitlines())
    }


def collect(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    rows_path = args.out / "rows.jsonl"
    done = existing(rows_path)
    fixed = cfg["fixed"]
    input_layer = int(fixed["input_residual_layer"])
    output_layer = int(fixed["output_residual_layer"])
    n_image = int(fixed["image_positions"])
    n_slots = int(fixed["image_slots"])
    steps = int(fixed["action_steps_for_metric"])
    alpha = float(fixed["interpolation_alpha"])
    if alpha != 0.5:
        raise RuntimeError("curvature test is frozen to the midpoint")

    suite_name = cfg["suite"]
    suite = _get_suite(suite_name)
    prompts = [task.language for task in suite.tasks]
    pairs = load_pairs(Path(cfg["pairs_file"]))
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    expected_layout = (n_slots, n_image)

    for cell_index, (task_a, task_b) in enumerate(pairs):
        cell = cell_name(suite_name, task_a, task_b)
        env = make_env(suite, suite_name, task_a)
        print(f"[curvature-action] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in cfg["eval_init_ids"]:
                if (cell, init_id) in done:
                    continue
                env.init_state_id = init_id
                observation, _ = env.reset(seed=1000 + init_id)
                batch_a = harness.build_batch(observation, prompts[task_a])
                batch_b = harness.build_batch(observation, prompts[task_b])
                _, name_a = task_object(suite_name, task_a)
                _, name_b = task_object(suite_name, task_b)
                segment_a, _ = harness.segments(batch_a, object_name=name_a, n_img=n_slots)
                segment_b, _ = harness.segments(batch_b, object_name=name_b, n_img=n_slots)
                if segment_a["INSTR"] != segment_b["INSTR"] or segment_a["TEXT_VALID"] != segment_b["TEXT_VALID"]:
                    raise RuntimeError(f"pair {task_a}/{task_b} is not position matched")

                prefix_a, captured_a = prefix_run(
                    harness, batch_a, input_layer=input_layer, output_layer=output_layer,
                    n_image=n_image, capture_layers=(input_layer,),
                )
                image_a = captured_a[input_layer]
                del prefix_a, captured_a

                prefix_b, captured_b = prefix_run(
                    harness, batch_b, input_layer=input_layer, output_layer=output_layer,
                    n_image=n_image, capture_layers=(input_layer, output_layer),
                )
                if (prefix_b["n_img_slots"], prefix_b["n_img_valid"]) != expected_layout:
                    raise RuntimeError("unexpected image layout")
                image_b = captured_b[input_layer]
                output_b = captured_b[output_layer]
                noise = harness.make_noise(seed=int(fixed["noise_seed_multiplier"]) * init_id)
                action_b = action_chunk(harness, prefix_b, noise)
                del prefix_b, captured_b

                prefix_zero, captured_zero = prefix_run(
                    harness, batch_b, input_layer=input_layer, output_layer=output_layer,
                    n_image=n_image, input_image=image_a, capture_layers=(output_layer,),
                )
                output_zero = captured_zero[output_layer]
                action_zero = action_chunk(harness, prefix_zero, noise)
                del prefix_zero, captured_zero

                input_midpoint = 0.5 * (image_a + image_b)
                prefix_native, captured_native = prefix_run(
                    harness, batch_b, input_layer=input_layer, output_layer=output_layer,
                    n_image=n_image, input_image=input_midpoint, capture_layers=(output_layer,),
                )
                output_native = captured_native[output_layer]
                action_native = action_chunk(harness, prefix_native, noise)
                del captured_native

                output_linear = 0.5 * (output_zero + output_b)
                curvature = output_native - output_linear
                prefix_linear, _ = prefix_run(
                    harness, batch_b, input_layer=input_layer, output_layer=output_layer,
                    n_image=n_image, input_image=input_midpoint, output_image=output_linear,
                )
                action_linear = action_chunk(harness, prefix_linear, noise)
                del prefix_linear

                prefix_rescue, _ = prefix_run(
                    harness, batch_b, input_layer=input_layer, output_layer=output_layer,
                    n_image=n_image, input_image=input_midpoint,
                    # Restoring the captured native field is exactly equivalent
                    # to adding the measured curvature, without a second
                    # float32 subtraction/addition round trip.
                    output_image=output_native,
                )
                action_rescue = action_chunk(harness, prefix_rescue, noise)
                prefix_identity = prefix_equal(prefix_native, prefix_rescue)
                action_identity = np.array_equal(action_native, action_rescue)
                if not prefix_identity or not action_identity:
                    raise RuntimeError("curvature rescue failed bitwise identity")

                ratio = relative_action_change(
                    action_native, action_linear, action_zero, action_b, steps
                )
                append_row(rows_path, {
                    "cell": cell,
                    "task_a": task_a,
                    "task_b": task_b,
                    "init": init_id,
                    "action_change_over_endpoint_distance": ratio,
                    "native_to_linearized_action_l2": float(
                        np.linalg.norm(action_linear[:steps] - action_native[:steps])
                    ),
                    "endpoint_action_l2": float(np.linalg.norm(action_b[:steps] - action_zero[:steps])),
                    "curvature_norm": float(torch.linalg.vector_norm(curvature)),
                    "prefix_rescue_bitwise_identity": prefix_identity,
                    "action_rescue_bitwise_identity": action_identity,
                    "chunks_first_10": {
                        "endpoint_A_image": np.asarray(action_zero[:steps]).round(7).tolist(),
                        "endpoint_B_image": np.asarray(action_b[:steps]).round(7).tolist(),
                        "midpoint_native": np.asarray(action_native[:steps]).round(7).tolist(),
                        "midpoint_linearized": np.asarray(action_linear[:steps]).round(7).tolist(),
                        "midpoint_rescue": np.asarray(action_rescue[:steps]).round(7).tolist()
                    }
                })
                done.add((cell, init_id))
                del prefix_native, prefix_rescue, image_a, image_b, output_b, output_zero
                del input_midpoint, output_native, output_linear, curvature
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            env.close()
    return rows_path


def analyze(rows_path: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    threshold = float(cfg["gates"]["median_action_change_over_endpoint_distance_min"])
    cells = []
    for cell in sorted({row["cell"] for row in rows}):
        ratio = median(
            row["action_change_over_endpoint_distance"] for row in rows if row["cell"] == cell
        )
        cells.append({"cell": cell, "median_ratio": ratio, "ratio_at_least_0_1": ratio >= threshold})
    overall = median(row["action_change_over_endpoint_distance"] for row in rows)
    passing_cells = sum(row["ratio_at_least_0_1"] for row in cells)
    all_rescued = all(
        row["prefix_rescue_bitwise_identity"] and row["action_rescue_bitwise_identity"]
        for row in rows
    )
    gate = cfg["gates"]
    passed = (
        overall >= threshold
        and passing_cells >= gate["directed_pairs_with_median_ratio_at_least_0_1_min"]
        and all_rescued
    )
    expected = len(load_pairs(Path(cfg["pairs_file"]))) * len(cfg["eval_init_ids"])
    if len(rows) != expected:
        raise RuntimeError(f"incomplete causal test: {len(rows)} != {expected}")
    result = {
        "status": "pass" if passed else "fail",
        "interpretation": (
            "downstream_action_sensitive_to_interpolated_state_curvature"
            if passed else "no_confirmed_action_sensitivity_to_interpolated_state_curvature"
        ),
        "metrics": {
            "median_action_change_over_endpoint_distance": overall,
            "directed_pairs_with_median_ratio_at_least_0_1": passing_cells,
            "minimum_ratio": min(row["action_change_over_endpoint_distance"] for row in rows),
            "maximum_ratio": max(row["action_change_over_endpoint_distance"] for row in rows),
            "all_rescues_bitwise_exact": all_rescued
        },
        "gate": gate,
        "cells": cells,
        "rows": len(rows),
        "expected_rows": expected,
        "rows_sha256": sha256(rows_path),
        "clean_behavior_claim": False,
        "further_search_authorized": False
    }
    write_json(args.out / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pi05_curvature_action.json"))
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("docs/PREREG-pi05-curvature-removal-rescue-action-2026-09-04.md"),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--minimum-free-gib", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    if (
        cfg["fixed"]["input_residual_layer"] != 5
        or cfg["fixed"]["output_residual_layer"] != 8
        or cfg["fixed"]["interpolation_alpha"] != 0.5
    ):
        raise RuntimeError("frozen curvature-action test changed")
    args.out.mkdir(parents=True, exist_ok=True)
    require_free_space(args.out, args.minimum_free_gib)
    ensure_manifest(args, cfg)
    print(json.dumps(analyze(collect(args, cfg), args, cfg), indent=2), flush=True)


if __name__ == "__main__":
    main()
