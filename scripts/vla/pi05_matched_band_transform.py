#!/usr/bin/env python
"""Test whether pi0.5's layer-6--8 instruction-writing update transfers across scenes."""
from __future__ import annotations

import argparse
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
    write_json,
)


CONDITIONS = (
    "clean_A",
    "clean_B",
    "matched_full",
    "mismatched_full_norm_matched",
    "matched_attention_only",
    "mismatched_attention_only_norm_matched",
    "random_norm_matched",
)


def _tensor_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_output(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (replacement,) + tuple(output[1:])
    return replacement


def prefix_run(
    harness: Pi05Harness,
    batch: dict[str, Any],
    *,
    input_layer: int,
    output_layer: int,
    band_layers: Iterable[int],
    n_image: int,
    input_image: torch.Tensor | None = None,
    output_delta: torch.Tensor | None = None,
    capture_layers: Iterable[int] = (),
    capture_mlp: bool = False,
    mlp_image_reference: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Run prefix computation with image-only residual and MLP interventions."""
    wanted = set(int(layer) for layer in capture_layers)
    active = set(wanted)
    if input_image is not None:
        active.add(input_layer)
    if output_delta is not None:
        active.add(output_layer)
    captured_resid: dict[int, torch.Tensor] = {}
    captured_mlp: dict[int, torch.Tensor] = {}

    def residual_hook(layer: int):
        def hook(hidden: torch.Tensor) -> torch.Tensor:
            result = hidden
            if layer == input_layer and input_image is not None:
                result = result.clone()
                result[:, :n_image] = input_image.to(result.device, result.dtype)
            if layer == output_layer and output_delta is not None:
                result = result.clone()
                result[:, :n_image] += output_delta.to(result.device, result.dtype)
            if layer in wanted:
                captured_resid[layer] = result[:, :n_image].detach().float().cpu().clone()
            return result
        return hook

    handles = []
    for layer in sorted(set(int(value) for value in band_layers)):
        module = harness.vlm_layers[layer].mlp

        def mlp_hook(mod: Any, inputs: Any, output: Any, layer: int = layer):
            tensor = _tensor_from_output(output)
            result = tensor
            if mlp_image_reference is not None:
                if layer not in mlp_image_reference:
                    raise RuntimeError(f"missing MLP reference for layer {layer}")
                result = tensor.clone()
                result[:, :n_image] = mlp_image_reference[layer].to(result.device, result.dtype)
            if capture_mlp:
                captured_mlp[layer] = result[:, :n_image].detach().float().cpu().clone()
            return None if result is tensor else _replace_output(output, result)

        handles.append(module.register_forward_hook(mlp_hook))
    try:
        prefix = harness.prefix_forward(
            batch,
            resid_hooks={layer: residual_hook(layer) for layer in sorted(active)},
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured_resid) != wanted:
        raise RuntimeError(f"missing residual captures: {sorted(wanted - set(captured_resid))}")
    expected_mlp = set(int(value) for value in band_layers) if capture_mlp else set()
    if set(captured_mlp) != expected_mlp:
        raise RuntimeError(f"missing MLP captures: {sorted(expected_mlp - set(captured_mlp))}")
    return prefix, captured_resid, captured_mlp


def norm_match(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_norm = float(torch.linalg.vector_norm(source))
    target_norm = float(torch.linalg.vector_norm(target))
    if target_norm <= 1e-12:
        return torch.zeros_like(source)
    if source_norm <= 1e-12:
        raise RuntimeError("cannot norm-match a zero source to a nonzero target")
    return source * (target_norm / source_norm)


def random_matched(target: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(target.shape, generator=generator, dtype=torch.float32)
    return norm_match(random, target)


def axis_metrics(value: np.ndarray, endpoint_a: np.ndarray, endpoint_b: np.ndarray) -> dict[str, float]:
    value64 = np.asarray(value, dtype=np.float64).reshape(-1)
    a64 = np.asarray(endpoint_a, dtype=np.float64).reshape(-1)
    b64 = np.asarray(endpoint_b, dtype=np.float64).reshape(-1)
    axis = b64 - a64
    denominator = float(axis @ axis)
    if denominator <= 1e-18:
        raise RuntimeError("clean A/B endpoints collapsed")
    scale = math.sqrt(denominator)
    return {
        "progress_to_B": float(((value64 - a64) @ axis) / denominator),
        "D_A": float(np.linalg.norm(value64 - a64) / scale),
        "D_B": float(np.linalg.norm(value64 - b64) / scale),
    }


def exact_two_sided_sign_test(values: Iterable[float]) -> dict[str, Any]:
    signs = [float(value) for value in values if abs(float(value)) > 1e-12]
    positive = sum(value > 0 for value in signs)
    n = len(signs)
    if n == 0:
        return {"positive": 0, "negative": 0, "ties": 0, "p_two_sided": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(positive, n - positive) + 1)) / (2 ** n)
    return {
        "positive": positive,
        "negative": n - positive,
        "ties": 0,
        "p_two_sided": float(min(1.0, 2.0 * tail)),
    }


def bootstrap_median_interval(values: Iterable[float], *, resamples: int, seed: int) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    samples = np.median(array[indices], axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def existing_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        result.add((row["cell"], int(row["init"]), row["condition"]))
    return result


def validate_pair_layout(
    harness: Pi05Harness,
    batch_a: dict[str, Any],
    batch_b: dict[str, Any],
    *,
    n_slots: int,
    n_image: int,
) -> None:
    _, _, _, slots_a, valid_a = harness._prefix_embed(batch_a)
    _, _, _, slots_b, valid_b = harness._prefix_embed(batch_b)
    if (slots_a, valid_a) != (n_slots, n_image) or (slots_b, valid_b) != (n_slots, n_image):
        raise RuntimeError(f"unexpected image layout: {(slots_a, valid_a)} / {(slots_b, valid_b)}")
    seg_a, _ = harness.segments(batch_a, n_img=n_slots)
    seg_b, _ = harness.segments(batch_b, n_img=n_slots)
    if seg_a["INSTR"] != seg_b["INSTR"] or seg_a["TEXT_VALID"] != seg_b["TEXT_VALID"]:
        raise RuntimeError("instruction pair is not token-position matched")


def make_batches(env: Any, harness: Pi05Harness, prompts: list[str], task_a: int, task_b: int, init_id: int):
    env.init_state_id = init_id
    observation, _ = env.reset(seed=1000 + init_id)
    return harness.build_batch(observation, prompts[task_a]), harness.build_batch(observation, prompts[task_b])


def measure_messages(
    harness: Pi05Harness,
    batch_a: dict[str, Any],
    batch_b: dict[str, Any],
    fixed: dict[str, Any],
) -> dict[str, torch.Tensor]:
    input_layer = int(fixed["input_residual_layer"])
    output_layer = int(fixed["intervention_residual_layer"])
    band = [int(value) for value in fixed["band_decoder_layers"]]
    n_image = int(fixed["image_positions"])
    prefix_a, resid_a, mlp_a = prefix_run(
        harness, batch_a, input_layer=input_layer, output_layer=output_layer,
        band_layers=band, n_image=n_image, capture_layers=(input_layer, output_layer),
        capture_mlp=True,
    )
    del prefix_a
    _, full_b, _ = prefix_run(
        harness, batch_b, input_layer=input_layer, output_layer=output_layer,
        band_layers=band, n_image=n_image, input_image=resid_a[input_layer],
        capture_layers=(output_layer,),
    )
    _, attention_b, _ = prefix_run(
        harness, batch_b, input_layer=input_layer, output_layer=output_layer,
        band_layers=band, n_image=n_image, input_image=resid_a[input_layer],
        capture_layers=(output_layer,), mlp_image_reference=mlp_a,
    )
    result = {
        "full": full_b[output_layer] - resid_a[output_layer],
        "attention_only": attention_b[output_layer] - resid_a[output_layer],
    }
    return {name: value.detach().float().cpu().clone() for name, value in result.items()}


def ensure_manifest(args: argparse.Namespace, cfg: dict[str, Any], *, smoke: bool) -> None:
    path = args.out / "manifest.json"
    root = Path(__file__).resolve().parents[2]
    hashes = {
        "runner": sha256(Path(__file__)),
        "config": sha256(args.config),
        "preregistration": sha256(args.preregistration),
        "pairs": sha256(Path(cfg["pairs_file"])),
        "hooks": sha256(root / "scripts/vla/hooks.py"),
    }
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "smoke" if smoke else "full",
        "config": cfg,
        "hashes": hashes,
    }
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["mode"] != payload["mode"] or prior["config"] != cfg or prior["hashes"] != hashes:
            raise RuntimeError("sealed inputs changed; use a new output directory")
        return
    write_json(path, payload)


def collect(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    *,
    pairs: list[tuple[int, int]],
    init_ids: list[int],
) -> Path:
    rows_path = args.out / "rows.jsonl"
    done = existing_keys(rows_path)
    fixed = cfg["fixed"]
    input_layer = int(fixed["input_residual_layer"])
    output_layer = int(fixed["intervention_residual_layer"])
    band = [int(value) for value in fixed["band_decoder_layers"]]
    n_image = int(fixed["image_positions"])
    n_slots = int(fixed["image_slots"])
    state_layers = [int(value) for value in fixed["state_layers"]]
    steps = int(fixed["action_steps_for_metric"])
    suite_name = cfg["suite"]
    suite = _get_suite(suite_name)
    prompts = [task.language for task in suite.tasks]
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)

    for cell_index, (task_a, task_b) in enumerate(pairs):
        cell = cell_name(suite_name, task_a, task_b)
        env = make_env(suite, suite_name, task_a)
        print(f"[matched-transform] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            messages: dict[int, dict[str, torch.Tensor]] = {}
            for init_id in init_ids:
                batch_a, batch_b = make_batches(env, harness, prompts, task_a, task_b, init_id)
                validate_pair_layout(harness, batch_a, batch_b, n_slots=n_slots, n_image=n_image)
                messages[init_id] = measure_messages(harness, batch_a, batch_b, fixed)
                del batch_a, batch_b
                gc.collect()

            for init_index, init_id in enumerate(init_ids):
                if all((cell, init_id, condition) in done for condition in CONDITIONS):
                    continue
                donor_init = init_ids[(init_index + 1) % len(init_ids)]
                batch_a, batch_b = make_batches(env, harness, prompts, task_a, task_b, init_id)
                prefix_a, resid_a, _ = prefix_run(
                    harness, batch_a, input_layer=input_layer, output_layer=output_layer,
                    band_layers=band, n_image=n_image,
                    capture_layers=tuple([output_layer] + state_layers),
                )
                prefix_b, resid_b, _ = prefix_run(
                    harness, batch_b, input_layer=input_layer, output_layer=output_layer,
                    band_layers=band, n_image=n_image,
                    capture_layers=tuple([output_layer] + state_layers),
                )
                noise = harness.make_noise(seed=int(fixed["noise_seed_multiplier"]) * init_id)
                action_a = action_chunk(harness, prefix_a, noise)[:steps]
                action_b = action_chunk(harness, prefix_b, noise)[:steps]
                action_axis = axis_metrics(action_a, action_a, action_b)
                if action_axis["D_B"] <= 1e-12:
                    raise RuntimeError("clean action endpoints collapsed")

                matched_full = messages[init_id]["full"]
                matched_attention = messages[init_id]["attention_only"]
                mismatch_full = norm_match(messages[donor_init]["full"], matched_full)
                mismatch_attention = norm_match(messages[donor_init]["attention_only"], matched_attention)
                random = random_matched(
                    matched_full,
                    seed=int(hashlib.sha256(f"{cell}:{init_id}".encode()).hexdigest()[:8], 16),
                )
                interventions = {
                    "matched_full": matched_full,
                    "mismatched_full_norm_matched": mismatch_full,
                    "matched_attention_only": matched_attention,
                    "mismatched_attention_only_norm_matched": mismatch_attention,
                    "random_norm_matched": random,
                }

                def record(condition: str, action: np.ndarray, states: dict[int, torch.Tensor], message: torch.Tensor | None):
                    key = (cell, init_id, condition)
                    if key in done:
                        return
                    payload: dict[str, Any] = {
                        "cell": cell,
                        "task_a": task_a,
                        "task_b": task_b,
                        "init": init_id,
                        "mismatch_donor_init": donor_init,
                        "condition": condition,
                        **axis_metrics(action, action_a, action_b),
                        "action_first_10": np.asarray(action).round(7).tolist(),
                        "message_frobenius_norm": None if message is None else float(torch.linalg.vector_norm(message)),
                        "state": {},
                    }
                    for layer in state_layers:
                        payload["state"][str(layer)] = axis_metrics(
                            states[layer].numpy(), resid_a[layer].numpy(), resid_b[layer].numpy()
                        )
                    append_row(rows_path, payload)
                    done.add(key)

                record("clean_A", action_a, resid_a, None)
                record("clean_B", action_b, resid_b, None)

                identity_prefix, _, _ = prefix_run(
                    harness, batch_a, input_layer=input_layer, output_layer=output_layer,
                    band_layers=band, n_image=n_image, output_delta=torch.zeros_like(matched_full),
                )
                if not prefix_equal(prefix_a, identity_prefix):
                    raise RuntimeError("zero-message identity intervention changed the prefix cache")
                del identity_prefix

                for condition, message in interventions.items():
                    if (cell, init_id, condition) in done:
                        continue
                    prefix, states, _ = prefix_run(
                        harness, batch_a, input_layer=input_layer, output_layer=output_layer,
                        band_layers=band, n_image=n_image, output_delta=message,
                        capture_layers=state_layers,
                    )
                    action = action_chunk(harness, prefix, noise)[:steps]
                    record(condition, action, states, message)
                    del prefix, states, action

                del batch_a, batch_b, prefix_a, prefix_b, resid_a, resid_b, noise
                del action_a, action_b, interventions
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            env.close()
    return rows_path


def analyze(
    rows_path: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    *,
    pairs: list[tuple[int, int]],
    init_ids: list[int],
) -> dict[str, Any]:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    expected = len(pairs) * len(init_ids) * len(CONDITIONS)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete experiment: {len(rows)} rows != {expected}")
    cells = sorted({row["cell"] for row in rows})

    def cell_metric(condition: str, metric: str = "progress_to_B") -> dict[str, float]:
        return {
            cell: median(
                row[metric] for row in rows
                if row["cell"] == cell and row["condition"] == condition
            )
            for cell in cells
        }

    metrics = {condition: cell_metric(condition) for condition in CONDITIONS}
    matched = metrics["matched_full"]
    mismatch = metrics["mismatched_full_norm_matched"]
    attention = metrics["matched_attention_only"]
    random = metrics["random_norm_matched"]
    contrasts = {
        "matched_minus_mismatched_full": [matched[cell] - mismatch[cell] for cell in cells],
        "matched_full_minus_matched_attention_only": [matched[cell] - attention[cell] for cell in cells],
        "mismatched_full_minus_random": [mismatch[cell] - random[cell] for cell in cells],
    }
    analysis_cfg = cfg["analysis"]
    resamples = int(analysis_cfg["bootstrap_resamples"])
    seed = int(analysis_cfg["bootstrap_seed"])
    contrast_summary = {}
    for index, (name, values) in enumerate(contrasts.items()):
        contrast_summary[name] = {
            "median_cell_effect": median(values),
            "bootstrap_95pct_interval": bootstrap_median_interval(
                values, resamples=resamples, seed=seed + index
            ),
            "exact_two_sided_sign_test": exact_two_sided_sign_test(values),
            "cell_effects": dict(zip(cells, values)),
        }

    condition_summary = {}
    for index, condition in enumerate(CONDITIONS):
        values = list(metrics[condition].values())
        condition_summary[condition] = {
            "median_cell_progress_to_B": median(values),
            "bootstrap_95pct_interval": bootstrap_median_interval(
                values, resamples=resamples, seed=seed + 100 + index
            ),
            "cells_with_positive_progress": sum(value > 0 for value in values),
            "cell_values": metrics[condition],
        }

    matched_vs_mismatch_ci = contrast_summary["matched_minus_mismatched_full"]["bootstrap_95pct_interval"]
    mismatch_vs_random_ci = contrast_summary["mismatched_full_minus_random"]["bootstrap_95pct_interval"]
    full_vs_attention_ci = contrast_summary["matched_full_minus_matched_attention_only"]["bootstrap_95pct_interval"]
    result = {
        "status": "complete",
        "interpretation_flags": {
            "evidence_for_scene_conditioning": matched_vs_mismatch_ci[0] > 0,
            "evidence_for_cross_scene_reuse": (
                condition_summary["mismatched_full_norm_matched"]["bootstrap_95pct_interval"][0] > 0
                and mismatch_vs_random_ci[0] > 0
            ),
            "evidence_that_mlp_improves_over_attention_only": full_vs_attention_ci[0] > 0,
        },
        "conditions": condition_summary,
        "contrasts": contrast_summary,
        "rows": len(rows),
        "expected_rows": expected,
        "directed_prompt_pair_cells": len(cells),
        "initial_states_per_cell": len(init_ids),
        "rows_sha256": sha256(rows_path),
        "claim_limits": [
            "The intervention is an additive full-image-field message measured from paired runs, not a discovered low-dimensional manifold.",
            "A matching-scene effect alone is not a portable causal rule.",
            "Immediate action chunks are not closed-loop task-success measurements.",
        ],
    }
    write_json(args.out / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pi05_matched_band_transform.json"))
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("docs/PREREG-pi05-matched-layer6-8-transform-2026-09-04.md"),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--minimum-free-gib", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    fixed = cfg["fixed"]
    if (
        fixed["input_residual_layer"] != 5
        or fixed["band_decoder_layers"] != [6, 7, 8]
        or fixed["intervention_residual_layer"] != 8
        or fixed["image_positions"] != 512
    ):
        raise RuntimeError("frozen matched-transform design changed")
    pairs = load_pairs(Path(cfg["pairs_file"]))
    init_ids = [int(value) for value in cfg["eval_init_ids"]]
    if args.smoke:
        pairs = pairs[:1]
        init_ids = init_ids[:1]
    args.out.mkdir(parents=True, exist_ok=True)
    require_free_space(args.out, args.minimum_free_gib)
    ensure_manifest(args, cfg, smoke=args.smoke)
    result = analyze(
        collect(args, cfg, pairs=pairs, init_ids=init_ids),
        args, cfg, pairs=pairs, init_ids=init_ids,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
