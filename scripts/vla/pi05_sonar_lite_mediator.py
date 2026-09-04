#!/usr/bin/env python
"""Sealed, donor-free-at-evaluation pi0.5 instruction-mediator experiment.

Stages are intentionally separate so later data remain unopened until the
preceding gate has been written. See the active amendment in cross model plan.md.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
from robosuite.utils import binding_utils

from build_stage1_twins import _patched_read_pixels
from hooks import Pi05Harness, axis_metric
from lerobot.envs.libero import LiberoEnv, _get_suite
from sonar_lite_geometry import (
    StableSVD,
    cosine,
    fit_stable_svd,
    matched_spectrum_random,
    materialize,
    median,
    parse_id_spec,
    rescale_like,
    stable_coefficients,
)
from stage2_discovery import GOAL_OBJ

binding_utils.MjRenderContext.read_pixels = _patched_read_pixels

LIVE_LAYERS = tuple(range(12, 18))
RANKS = (1, 2, 4, 8, 16, 32)
REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
EXPECTED_CONDITIONS = (
    "clean_A",
    "clean_B",
    "insert_fit",
    "remove_fit",
    "insert_reverse",
    "remove_reverse",
    "insert_wrong",
    "remove_wrong",
    "insert_random",
    "remove_random",
    "insert_ceiling",
    "remove_ceiling",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_free_space(path: Path, minimum_gib: float) -> None:
    free = shutil.disk_usage(path).free
    if free < minimum_gib * 1024**3:
        raise RuntimeError(f"only {free / 1024**3:.2f} GiB free; require {minimum_gib:.2f} GiB")


def cell_name(suite: str, task_a: int, task_b: int) -> str:
    return f"{suite}_t{task_a}_vs_t{task_b}"


def factor_payload(factors: StableSVD) -> dict[str, torch.Tensor]:
    return {
        "token_basis": factors.token_basis.half().cpu(),
        "singular_values": factors.singular_values.float().cpu(),
        "feature_basis": factors.feature_basis.half().cpu(),
        "reliability": factors.reliability.float().cpu(),
    }


def load_factor(payload: dict[str, torch.Tensor]) -> StableSVD:
    return StableSVD(
        token_basis=payload["token_basis"].float(),
        singular_values=payload["singular_values"].float(),
        feature_basis=payload["feature_basis"].float(),
        reliability=payload["reliability"].float(),
    )


def load_operators(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def make_env(suite: Any, suite_name: str, task_id: int) -> LiberoEnv:
    return LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        obs_type="pixels_agent_pos",
        observation_height=360,
        observation_width=360,
        init_states=True,
        episode_index=0,
        n_envs=1,
    )


def capture_pair(
    harness: Pi05Harness,
    env: LiberoEnv,
    prompts: list[str],
    task_a: int,
    task_b: int,
    init_id: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    env.init_state_id = init_id
    observation, _ = env.reset(seed=1000 + init_id)
    batch_a = harness.build_batch(observation, prompts[task_a])
    batch_b = harness.build_batch(observation, prompts[task_b])
    prefix_a = harness.prefix_forward(batch_a, capture=True)
    prefix_b = harness.prefix_forward(batch_b, capture=True)
    if prefix_a["n_img_valid"] != 512 or prefix_b["n_img_valid"] != 512:
        raise RuntimeError(f"expected 512 valid image positions, got {prefix_a['n_img_valid']}/{prefix_b['n_img_valid']}")
    return batch_a, batch_b, prefix_a, prefix_b


def extract_deltas(prefix_a: dict[str, Any], prefix_b: dict[str, Any]) -> dict[int, torch.Tensor]:
    n_img = int(prefix_a["n_img_valid"])
    return {
        layer: (prefix_b["resid"][layer][0, :n_img] - prefix_a["resid"][layer][0, :n_img]).float()
        for layer in LIVE_LAYERS
    }


def run_fit(args: argparse.Namespace, pairs: list[list[int]]) -> None:
    operator_path = args.out / "operators.pt"
    if operator_path.exists():
        raise FileExistsError(f"refusing to overwrite {operator_path}")
    ids = parse_id_spec(args.fit_ids)
    harness = Pi05Harness(args.policy, revision=args.revision, dtype="float32")
    suite = _get_suite(args.suite)
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    operators: dict[str, Any] = {
        "schema_version": 1,
        "fit_ids": ids,
        "live_layers": list(LIVE_LAYERS),
        "max_rank": max(RANKS),
        "cells": {},
    }
    audit_rows: list[dict[str, Any]] = []

    for cell_index, (task_a, task_b, task_c) in enumerate(pairs):
        require_free_space(args.out, args.minimum_free_gib)
        cell = cell_name(args.suite, task_a, task_b)
        env = make_env(suite, args.suite, task_a)
        by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in LIVE_LAYERS}
        print(f"[fit] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in ids:
                _, _, prefix_a, prefix_b = capture_pair(harness, env, prompts, task_a, task_b, init_id)
                deltas = extract_deltas(prefix_a, prefix_b)
                for layer, delta in deltas.items():
                    by_layer[layer].append(delta)
                del prefix_a, prefix_b, deltas
                gc.collect()
                print(f"  captured init {init_id}", flush=True)
        finally:
            env.close()

        layer_payload: dict[str, Any] = {}
        for layer in LIVE_LAYERS:
            delta_stack = torch.stack(by_layer[layer])
            factors = fit_stable_svd(
                delta_stack,
                max_rank=max(RANKS),
                seed=args.seed + 1000 * cell_index + layer,
                compute_device=args.svd_device,
            )
            layer_payload[str(layer)] = factor_payload(factors)
            audit_rows.append(
                {
                    "cell": cell,
                    "layer": layer,
                    "n_fit": len(ids),
                    "mean_delta_norm": float(delta_stack.mean(0).norm()),
                    "rank32_energy_fraction": float(
                        factors.singular_values.square().sum()
                        / delta_stack.mean(0).square().sum().clamp_min(1e-12)
                    ),
                    "singular_values": factors.singular_values.tolist(),
                    "reliability": factors.reliability.tolist(),
                }
            )
            del delta_stack, factors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        operators["cells"][cell] = {
            "task_a": task_a,
            "task_b": task_b,
            "task_c": task_c,
            "layers": layer_payload,
        }
        del by_layer, layer_payload
        gc.collect()

    torch.save(operators, operator_path)
    with (args.out / "fit_rows.jsonl").open("w") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row) + "\n")
    write_json(
        args.out / "fit_receipt.json",
        {
            "status": "complete",
            "operator_sha256": sha256(operator_path),
            "cells": len(operators["cells"]),
            "fit_ids": ids,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(f"[fit] complete -> {operator_path}", flush=True)


def select_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cell_summaries: list[dict[str, Any]] = []
    for layer in LIVE_LAYERS:
        for rank in RANKS:
            subset = [row for row in rows if row["layer"] == layer and row["rank"] == rank]
            cells = sorted({row["cell"] for row in subset})
            per_cell = []
            for cell in cells:
                crows = [row for row in subset if row["cell"] == cell]
                fit_cos = median(row["cos_fit"] for row in crows)
                wrong_cos = median(row["cos_wrong"] for row in crows)
                per_cell.append(
                    {
                        "cell": cell,
                        "cos_fit": fit_cos,
                        "cos_wrong": wrong_cos,
                        "selectivity": fit_cos - max(0.0, wrong_cos),
                    }
                )
            cell_summaries.append(
                {
                    "layer": layer,
                    "rank": rank,
                    "median_cos_fit": median(row["cos_fit"] for row in per_cell),
                    "median_cos_wrong": median(row["cos_wrong"] for row in per_cell),
                    "median_selectivity": median(row["selectivity"] for row in per_cell),
                    "fit_beats_wrong_cells": sum(row["cos_fit"] > row["cos_wrong"] for row in per_cell),
                    "n_cells": len(per_cell),
                }
            )

    rank32 = [row for row in cell_summaries if row["rank"] == max(RANKS)]
    best_layer_score = max(row["median_selectivity"] for row in rank32)
    selected_layer = max(
        row["layer"] for row in rank32 if row["median_selectivity"] >= best_layer_score - 0.005
    )
    selected_layer_rows = [row for row in cell_summaries if row["layer"] == selected_layer]
    rank32_score = next(row["median_selectivity"] for row in selected_layer_rows if row["rank"] == max(RANKS))
    threshold = 0.95 * rank32_score if rank32_score >= 0 else rank32_score
    eligible_ranks = [row["rank"] for row in selected_layer_rows if row["median_selectivity"] >= threshold]
    selected_rank = min(eligible_ranks) if eligible_ranks else max(RANKS)
    selected = next(
        row for row in selected_layer_rows if row["rank"] == selected_rank
    )
    passed = (
        selected["median_cos_fit"] >= 0.10
        and selected["median_selectivity"] >= 0.05
        and selected["fit_beats_wrong_cells"] >= 8
        and selected["n_cells"] == 12
    )
    return {
        "status": "pass" if passed else "fail",
        "selected_layer": selected_layer,
        "selected_rank": selected_rank,
        "selected_metrics": selected,
        "rank32_layer_score": best_layer_score,
        "rank_threshold": threshold,
        "all_candidates": cell_summaries,
        "gate": {
            "median_cos_fit_min": 0.10,
            "median_selectivity_min": 0.05,
            "fit_beats_wrong_cells_min": 8,
            "required_cells": 12,
        },
    }


def run_select(args: argparse.Namespace, pairs: list[list[int]]) -> None:
    operator_path = args.out / "operators.pt"
    operators = load_operators(operator_path)
    if (args.out / "selection.json").exists():
        raise FileExistsError(f"refusing to overwrite {args.out / 'selection.json'}")
    ids = parse_id_spec(args.select_ids)
    harness = Pi05Harness(args.policy, revision=args.revision, dtype="float32")
    suite = _get_suite(args.suite)
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    rows: list[dict[str, Any]] = []

    for cell_index, (task_a, task_b, task_c) in enumerate(pairs):
        require_free_space(args.out, args.minimum_free_gib)
        cell = cell_name(args.suite, task_a, task_b)
        wrong_cell = cell_name(args.suite, task_a, task_c)
        env = make_env(suite, args.suite, task_a)
        print(f"[select] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in ids:
                _, _, prefix_a, prefix_b = capture_pair(harness, env, prompts, task_a, task_b, init_id)
                deltas = extract_deltas(prefix_a, prefix_b)
                for layer in LIVE_LAYERS:
                    fitted_factors = load_factor(operators["cells"][cell]["layers"][str(layer)])
                    wrong_factors = load_factor(operators["cells"][wrong_cell]["layers"][str(layer)])
                    delta = deltas[layer]
                    for rank in RANKS:
                        fitted = materialize(fitted_factors, rank)
                        wrong = rescale_like(materialize(wrong_factors, rank), fitted)
                        rows.append(
                            {
                                "cell": cell,
                                "task_a": task_a,
                                "task_b": task_b,
                                "task_c": task_c,
                                "init": init_id,
                                "layer": layer,
                                "rank": rank,
                                "cos_fit": cosine(fitted, delta),
                                "cos_wrong": cosine(wrong, delta),
                                "fit_norm": float(fitted.norm()),
                                "delta_norm": float(delta.norm()),
                            }
                        )
                del prefix_a, prefix_b, deltas
                gc.collect()
                print(f"  scored init {init_id}", flush=True)
        finally:
            env.close()

    with (args.out / "selection_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = select_summary(rows)
    summary.update(
        {
            "selection_ids": ids,
            "operator_sha256": sha256(operator_path),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    write_json(args.out / "selection.json", summary)
    print(json.dumps(summary["selected_metrics"], indent=2), flush=True)
    print(f"[select] gate={summary['status']} layer={summary['selected_layer']} rank={summary['selected_rank']}", flush=True)


def additive_hooks(template: torch.Tensor, layer: int, sign: float) -> dict[int, Any]:
    template = template.float().cpu()

    def hook(hidden: torch.Tensor) -> torch.Tensor:
        result = hidden.clone()
        result[:, : template.shape[0], :] = (
            result[:, : template.shape[0], :]
            + sign * template.to(device=result.device, dtype=result.dtype).unsqueeze(0)
        )
        return result

    return {layer: hook}


def replacement_hooks(source_residuals: dict[int, torch.Tensor], n_img: int) -> dict[int, Any]:
    hooks = {}
    for layer in LIVE_LAYERS:
        source = source_residuals[layer][:, :n_img].float().cpu()

        def hook(hidden: torch.Tensor, source: torch.Tensor = source) -> torch.Tensor:
            result = hidden.clone()
            result[:, : source.shape[1], :] = source.to(device=result.device, dtype=result.dtype)
            return result

        hooks[layer] = hook
    return hooks


def metric_row(
    cell: str,
    tasks: tuple[int, int, int],
    init_id: int,
    condition: str,
    chunk: np.ndarray,
    clean_a: np.ndarray,
    clean_b: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> dict[str, Any]:
    denominator = float(np.linalg.norm(clean_a[:10] - clean_b[:10]))
    if denominator <= 1e-9:
        raise RuntimeError(f"clean endpoints collapsed for {cell} init {init_id}")
    axis_denominator = axis_metric(clean_b, point_a, point_b, 10) - axis_metric(clean_a, point_a, point_b, 10)
    return {
        "cell": cell,
        "task_a": tasks[0],
        "task_b": tasks[1],
        "task_c": tasks[2],
        "init": init_id,
        "condition": condition,
        "clean_ab_l2": denominator,
        "nL2_to_A": float(np.linalg.norm(chunk[:10] - clean_a[:10]) / denominator),
        "nL2_to_B": float(np.linalg.norm(chunk[:10] - clean_b[:10]) / denominator),
        "R": float(
            (axis_metric(chunk, point_a, point_b, 10) - axis_metric(clean_a, point_a, point_b, 10))
            / (axis_denominator if abs(axis_denominator) > 1e-9 else 1e-9)
        ),
        "chunk10": np.asarray(chunk[:10]).round(7).tolist(),
    }


def action_chunk(harness: Pi05Harness, prefix: dict[str, Any], noise: torch.Tensor) -> np.ndarray:
    return harness.unnormalize(harness.action_forward(prefix, noise)[0])[0].cpu().numpy()


def run_causal(args: argparse.Namespace, pairs: list[list[int]], stage: str) -> None:
    selection = json.loads((args.out / "selection.json").read_text())
    if selection["status"] != "pass":
        raise RuntimeError("representation gate failed; causal data must remain unopened")
    if stage == "confirm":
        screen_path = args.out / "screen_summary.json"
        if not screen_path.exists() or json.loads(screen_path.read_text()).get("status") != "pass":
            raise RuntimeError("screen gate did not pass; confirmation data must remain unopened")
    ids = parse_id_spec(args.screen_ids if stage == "screen" else args.confirm_ids)
    layer, rank = int(selection["selected_layer"]), int(selection["selected_rank"])
    operators = load_operators(args.out / "operators.pt")
    rows_path = args.out / f"{stage}_rows.jsonl"
    done: set[tuple[str, int, str]] = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try:
                row = json.loads(line)
                done.add((row["cell"], int(row["init"]), row["condition"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    harness = Pi05Harness(args.policy, revision=args.revision, dtype="float32")
    suite = _get_suite(args.suite)
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]

    with rows_path.open("a") as handle:
        for cell_index, (task_a, task_b, task_c) in enumerate(pairs):
            require_free_space(args.out, args.minimum_free_gib)
            cell = cell_name(args.suite, task_a, task_b)
            wrong_cell = cell_name(args.suite, task_a, task_c)
            fitted_factors = load_factor(operators["cells"][cell]["layers"][str(layer)])
            wrong_factors = load_factor(operators["cells"][wrong_cell]["layers"][str(layer)])
            fitted = materialize(fitted_factors, rank)
            wrong = rescale_like(materialize(wrong_factors, rank), fitted)
            random = matched_spectrum_random(
                fitted_factors,
                rank,
                seed=args.seed + 10_000 * cell_index + layer * 100 + rank,
            )
            env = make_env(suite, args.suite, task_a)
            raw_env = env._env.env
            object_a, _ = GOAL_OBJ[task_a]
            object_b, _ = GOAL_OBJ[task_b]
            print(f"[{stage}] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
            try:
                for init_id in ids:
                    missing = [
                        condition for condition in EXPECTED_CONDITIONS
                        if (cell, init_id, condition) not in done
                    ]
                    if not missing:
                        print(f"  init {init_id} already complete", flush=True)
                        continue
                    batch_a, batch_b, prefix_a, prefix_b = capture_pair(
                        harness, env, prompts, task_a, task_b, init_id
                    )
                    point_a = np.asarray(raw_env.sim.data.body_xpos[raw_env.obj_body_id[object_a]])
                    point_b = np.asarray(raw_env.sim.data.body_xpos[raw_env.obj_body_id[object_b]])
                    noise = harness.make_noise(seed=init_id * 100)
                    clean_a = action_chunk(harness, prefix_a, noise)
                    clean_b = action_chunk(harness, prefix_b, noise)
                    conditions: dict[str, tuple[dict[str, Any], dict[int, Any] | None]] = {
                        "clean_A": (prefix_a, None),
                        "clean_B": (prefix_b, None),
                        "insert_fit": (batch_a, additive_hooks(fitted, layer, +1.0)),
                        "remove_fit": (batch_b, additive_hooks(fitted, layer, -1.0)),
                        "insert_reverse": (batch_a, additive_hooks(fitted, layer, -1.0)),
                        "remove_reverse": (batch_b, additive_hooks(fitted, layer, +1.0)),
                        "insert_wrong": (batch_a, additive_hooks(wrong, layer, +1.0)),
                        "remove_wrong": (batch_b, additive_hooks(wrong, layer, -1.0)),
                        "insert_random": (batch_a, additive_hooks(random, layer, +1.0)),
                        "remove_random": (batch_b, additive_hooks(random, layer, -1.0)),
                        "insert_ceiling": (
                            batch_a,
                            replacement_hooks(prefix_b["resid"], int(prefix_b["n_img_valid"])),
                        ),
                        "remove_ceiling": (
                            batch_b,
                            replacement_hooks(prefix_a["resid"], int(prefix_a["n_img_valid"])),
                        ),
                    }
                    for condition in EXPECTED_CONDITIONS:
                        if condition not in missing:
                            continue
                        source, hooks = conditions[condition]
                        if condition == "clean_A":
                            chunk = clean_a
                        elif condition == "clean_B":
                            chunk = clean_b
                        else:
                            intervened_prefix = harness.prefix_forward(source, resid_hooks=hooks)
                            chunk = action_chunk(harness, intervened_prefix, noise)
                            del intervened_prefix
                        row = metric_row(
                            cell,
                            (task_a, task_b, task_c),
                            init_id,
                            condition,
                            chunk,
                            clean_a,
                            clean_b,
                            point_a,
                            point_b,
                        )
                        row.update({"stage": stage, "selected_layer": layer, "selected_rank": rank})
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        done.add((cell, init_id, condition))
                    del prefix_a, prefix_b, conditions
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"  init {init_id} complete", flush=True)
            finally:
                env.close()

    write_json(
        args.out / f"{stage}_receipt.json",
        {
            "status": "complete",
            "ids": ids,
            "rows_sha256": sha256(rows_path),
            "row_count": len(rows_path.read_text().splitlines()),
            "selected_layer": layer,
            "selected_rank": rank,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(f"[{stage}] complete -> {rows_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("fit", "select", "screen", "confirm"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs_file", type=Path, default=Path("artifacts/mediation_pairs.json"))
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--fit_ids", default="0-5")
    parser.add_argument("--select_ids", default="6-7")
    parser.add_argument("--screen_ids", default="8-9")
    parser.add_argument("--confirm_ids", default="10-15")
    parser.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--svd_device", default="cuda")
    parser.add_argument("--minimum_free_gib", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    require_free_space(args.out, args.minimum_free_gib)
    pairs = json.loads(args.pairs_file.read_text())
    if len(pairs) != 12 or any(len(pair) != 3 for pair in pairs):
        raise RuntimeError("the sealed experiment requires exactly 12 [A, B, C] cells")
    manifest_path = args.out / "manifest.json"
    if not manifest_path.exists():
        write_json(
            manifest_path,
            {
                "schema_version": 1,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "args_at_creation": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "fixed": {
                    "live_layers": list(LIVE_LAYERS),
                    "ranks": list(RANKS),
                    "dose": 1.0,
                    "conditions": list(EXPECTED_CONDITIONS),
                    "pairs": pairs,
                },
                "hashes": {
                    "runner": sha256(Path(__file__)),
                    "geometry": sha256(Path(__file__).with_name("sonar_lite_geometry.py")),
                    "hooks": sha256(Path(__file__).with_name("hooks.py")),
                    "pairs": sha256(args.pairs_file),
                },
            },
        )
    if args.stage == "fit":
        run_fit(args, pairs)
    elif args.stage == "select":
        run_select(args, pairs)
    else:
        run_causal(args, pairs, args.stage)


if __name__ == "__main__":
    main()
