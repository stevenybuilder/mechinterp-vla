#!/usr/bin/env python
"""Direct midpoint-curvature test for pi0.5's layer-6--8 image-state mapping."""
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
    cell_name,
    load_pairs,
    make_env,
    median,
    require_free_space,
    sha256,
    task_object,
    write_json,
)


def curvature_ratio(output_a: np.ndarray, output_mid: np.ndarray, output_b: np.ndarray) -> float:
    expected_mid = 0.5 * (np.asarray(output_a, dtype=np.float64) + np.asarray(output_b, dtype=np.float64))
    curvature = np.linalg.norm(np.asarray(output_mid, dtype=np.float64) - expected_mid)
    chord = np.linalg.norm(np.asarray(output_b, dtype=np.float64) - np.asarray(output_a, dtype=np.float64))
    return float(curvature / max(float(chord), 1e-12))


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
            raise RuntimeError("sealed diagnostic inputs changed; use a new output directory")
        return
    write_json(path, {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg,
        "hashes": hashes,
    })


def capture_layers(
    harness: Pi05Harness,
    batch: dict[str, Any],
    layers: Iterable[int],
    *,
    replacement_layer: int | None = None,
    replacement_image: torch.Tensor | None = None,
    n_image: int = 512,
) -> tuple[dict[int, torch.Tensor], dict[str, int]]:
    wanted = tuple(sorted(set(int(layer) for layer in layers)))
    captured: dict[int, torch.Tensor] = {}

    def hook_for(layer: int):
        def hook(hidden: torch.Tensor) -> torch.Tensor:
            result = hidden
            if layer == replacement_layer:
                if replacement_image is None:
                    raise RuntimeError("replacement layer requested without an image field")
                result = hidden.clone()
                result[:, :n_image] = replacement_image.to(device=hidden.device, dtype=hidden.dtype)
            if layer in wanted:
                captured[layer] = result[:, :n_image].detach().float().cpu().clone()
            return result
        return hook

    active = set(wanted)
    if replacement_layer is not None:
        active.add(int(replacement_layer))
    prefix = harness.prefix_forward(batch, resid_hooks={layer: hook_for(layer) for layer in active})
    layout = {
        "n_img_slots": int(prefix["n_img_slots"]),
        "n_img_valid": int(prefix["n_img_valid"]),
    }
    del prefix
    if set(captured) != set(wanted):
        raise RuntimeError(f"missing residual captures: {sorted(set(wanted) - set(captured))}")
    return captured, layout


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def existing(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        result.add((row["cell"], int(row["init"])))
    return result


def collect(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    rows_path = args.out / "rows.jsonl"
    done = existing(rows_path)
    pairs = load_pairs(Path(cfg["pairs_file"]))
    fixed = cfg["fixed"]
    input_layer = int(fixed["input_residual_layer"])
    output_layer = int(fixed["output_residual_layer"])
    n_image = int(fixed["image_positions"])
    n_slots = int(fixed["image_slots"])
    alpha = float(fixed["interpolation_alpha"])
    if alpha != 0.5:
        raise RuntimeError("this diagnostic is frozen to the exact midpoint")

    suite_name = cfg["suite"]
    suite = _get_suite(suite_name)
    prompts = [task.language for task in suite.tasks]
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    expected_layout = {"n_img_slots": n_slots, "n_img_valid": n_image}

    for cell_index, (task_a, task_b) in enumerate(pairs):
        cell = cell_name(suite_name, task_a, task_b)
        env = make_env(suite, suite_name, task_a)
        print(f"[midpoint-curvature] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
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
                if (
                    segment_a["INSTR"] != segment_b["INSTR"]
                    or segment_a["TEXT_VALID"] != segment_b["TEXT_VALID"]
                ):
                    raise RuntimeError(f"pair {task_a}/{task_b} is not position matched")

                clean_a, layout_a = capture_layers(harness, batch_a, (input_layer,), n_image=n_image)
                clean_b, layout_b = capture_layers(
                    harness, batch_b, (input_layer, output_layer), n_image=n_image
                )
                if layout_a != expected_layout or layout_b != expected_layout:
                    raise RuntimeError(f"unexpected image layout: {layout_a}/{layout_b}")
                image_a = clean_a[input_layer]
                image_b = clean_b[input_layer]
                midpoint = (1.0 - alpha) * image_a + alpha * image_b

                endpoint_a, _ = capture_layers(
                    harness, batch_b, (output_layer,), replacement_layer=input_layer,
                    replacement_image=image_a, n_image=n_image,
                )
                midpoint_output, _ = capture_layers(
                    harness, batch_b, (output_layer,), replacement_layer=input_layer,
                    replacement_image=midpoint, n_image=n_image,
                )
                endpoint_b, _ = capture_layers(
                    harness, batch_b, (output_layer,), replacement_layer=input_layer,
                    replacement_image=image_b, n_image=n_image,
                )
                if not torch.equal(endpoint_b[output_layer], clean_b[output_layer]):
                    maximum = float(
                        (endpoint_b[output_layer] - clean_b[output_layer]).abs().max()
                    )
                    raise RuntimeError(f"B endpoint identity failed, maxabs={maximum}")

                output_a = endpoint_a[output_layer].numpy()
                output_mid = midpoint_output[output_layer].numpy()
                output_b = endpoint_b[output_layer].numpy()
                ratio = curvature_ratio(output_a, output_mid, output_b)
                append_row(rows_path, {
                    "cell": cell,
                    "task_a": task_a,
                    "task_b": task_b,
                    "init": init_id,
                    "curvature_to_chord_ratio": ratio,
                    "input_chord_norm": float(torch.linalg.vector_norm(image_b - image_a)),
                    "output_chord_norm": float(np.linalg.norm(output_b.astype(np.float64) - output_a.astype(np.float64))),
                    "curvature_norm": float(np.linalg.norm(
                        output_mid.astype(np.float64)
                        - 0.5 * (output_a.astype(np.float64) + output_b.astype(np.float64))
                    )),
                    "b_endpoint_bitwise_identity": True,
                })
                done.add((cell, init_id))
                del clean_a, clean_b, endpoint_a, midpoint_output, endpoint_b
                del image_a, image_b, midpoint, output_a, output_mid, output_b
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            env.close()
    return rows_path


def analyze(rows_path: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    cell_results = []
    for cell in sorted({row["cell"] for row in rows}):
        ratio = median(
            row["curvature_to_chord_ratio"] for row in rows if row["cell"] == cell
        )
        cell_results.append({
            "cell": cell,
            "median_curvature_to_chord_ratio": ratio,
            "ratio_at_least_0_1": ratio >= 0.1,
        })
    overall = median(row["curvature_to_chord_ratio"] for row in rows)
    passing_cells = sum(row["ratio_at_least_0_1"] for row in cell_results)
    gate = cfg["gates"]
    passed = (
        overall >= gate["median_curvature_to_chord_ratio_min"]
        and passing_cells >= gate["directed_pairs_with_median_ratio_at_least_0_1_min"]
    )
    expected_rows = len(load_pairs(Path(cfg["pairs_file"]))) * len(cfg["eval_init_ids"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"incomplete diagnostic: {len(rows)} rows != {expected_rows}")
    result = {
        "status": "pass" if passed else "fail",
        "interpretation": (
            "meaningful_non_affinity_along_instruction_chords"
            if passed else "approximately_affine_at_tested_scale"
        ),
        "metrics": {
            "median_curvature_to_chord_ratio": overall,
            "directed_pairs_with_median_ratio_at_least_0_1": passing_cells,
            "minimum_ratio": min(row["curvature_to_chord_ratio"] for row in rows),
            "maximum_ratio": max(row["curvature_to_chord_ratio"] for row in rows),
        },
        "gate": gate,
        "cells": cell_results,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "rows_sha256": sha256(rows_path),
        "model_actions_sampled": 0,
        "manifold_identified": False,
        "further_search_authorized": False,
    }
    write_json(args.out / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pi05_lean_nonlinear_writer.json"))
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("docs/PREREG-pi05-lean-nonlinear-writer-diagnostic-2026-09-04.md"),
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
        raise RuntimeError("frozen midpoint diagnostic changed")
    args.out.mkdir(parents=True, exist_ok=True)
    require_free_space(args.out, args.minimum_free_gib)
    ensure_manifest(args, cfg)
    print(json.dumps(analyze(collect(args, cfg), args, cfg), indent=2), flush=True)


if __name__ == "__main__":
    main()
