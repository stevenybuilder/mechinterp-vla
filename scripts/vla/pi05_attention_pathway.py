#!/usr/bin/env python
"""Preregistered attention-edge block-and-rescue experiment for pi0.5.

The stages are deliberately separate. Selection uses attention-output geometry
without consulting behavior. Causal screening uses different initial states,
and confirmation refuses to run unless both screen gates pass.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
from robosuite.utils import binding_utils

from attention_pathway import (
    AllowCurrent,
    AttentionCapture,
    SourceEffectCapture,
    SourceKVSubstitution,
    ranked_heads_by_write,
)
from build_stage1_twins import _patched_read_pixels
from hooks import Pi05Harness, axis_metric, cache_kv_lists
from lerobot.envs.libero import LiberoEnv, _get_suite
from stage2_discovery import GOAL_OBJ

binding_utils.MjRenderContext.read_pixels = _patched_read_pixels


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


def median(values: Iterable[float]) -> float:
    values = list(float(value) for value in values)
    return float(np.median(values)) if values else float("nan")


def cell_name(suite: str, task_a: int, task_b: int) -> str:
    return f"{suite}_t{task_a}_vs_t{task_b}"


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


def action_chunk(
    harness: Pi05Harness,
    prefix: dict[str, Any],
    noise: torch.Tensor,
    *,
    attn_hook: Any = None,
) -> np.ndarray:
    chunk, _ = harness.action_forward(prefix, noise, attn_hook=attn_hook)
    return harness.unnormalize(chunk)[0].float().cpu().numpy()


def cache_dict(prefix: dict[str, Any], layers: Iterable[int]) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    keys, values = cache_kv_lists(prefix["cache"])
    selected = tuple(int(layer) for layer in layers)
    return (
        {layer: keys[layer].detach().cpu().clone() for layer in selected},
        {layer: values[layer].detach().cpu().clone() for layer in selected},
    )


def caches_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_k, left_v = cache_kv_lists(left["cache"])
    right_k, right_v = cache_kv_lists(right["cache"])
    return all(torch.equal(a, b) for a, b in zip(left_k + left_v, right_k + right_v))


def prefix_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not caches_equal(left, right):
        return False
    if set(left["resid"]) != set(right["resid"]):
        return False
    return all(torch.equal(left["resid"][layer], right["resid"][layer]) for layer in left["resid"])


def capture_pair(
    harness: Pi05Harness,
    env: LiberoEnv,
    prompts: list[str],
    task_a: int,
    task_b: int,
    init_id: int,
    *,
    capture_residuals: bool,
    capture_attention: bool,
) -> dict[str, Any]:
    env.init_state_id = init_id
    observation, _ = env.reset(seed=1000 + init_id)
    batch_a = harness.build_batch(observation, prompts[task_a])
    batch_b = harness.build_batch(observation, prompts[task_b])
    emb_a, _, _, n_img_slots, n_img_valid = harness._prefix_embed(batch_a)
    emb_b, _, _, n_img_slots_b, n_img_valid_b = harness._prefix_embed(batch_b)
    if (n_img_slots, n_img_valid) != (n_img_slots_b, n_img_valid_b):
        raise RuntimeError("A/B image token layouts differ")
    if n_img_valid != 512:
        raise RuntimeError(f"expected 512 valid image positions, got {n_img_valid}")
    seg_a, _ = harness.segments(batch_a, object_name=GOAL_OBJ[task_a][1], n_img=n_img_slots)
    seg_b, _ = harness.segments(batch_b, object_name=GOAL_OBJ[task_b][1], n_img=n_img_slots)
    if seg_a["INSTR"] != seg_b["INSTR"] or seg_a["TEXT_VALID"] != seg_b["TEXT_VALID"]:
        raise RuntimeError("A/B token positions differ; contrast is not position matched")
    image = list(range(n_img_valid))
    instruction = list(seg_a["INSTR"])
    valid = image + list(seg_a["TEXT_VALID"])
    noninstruction = sorted(set(valid) - set(instruction))
    other = sorted(set(noninstruction) - set(image))
    if not torch.equal(emb_a[:, noninstruction], emb_b[:, noninstruction]):
        max_abs = float((emb_a[:, noninstruction].float() - emb_b[:, noninstruction].float()).abs().max())
        raise RuntimeError(f"non-instruction input embeddings differ, maxabs={max_abs}")

    cap_a = AttentionCapture(keep_output_layers={0}) if capture_attention else None
    cap_b = AttentionCapture(keep_output_layers={0}) if capture_attention else None
    prefix_a = harness.prefix_forward(batch_a, capture=capture_residuals, attn_hook=cap_a)
    prefix_b = harness.prefix_forward(batch_b, capture=capture_residuals, attn_hook=cap_b)
    raw_env = env._env.env
    object_a, _ = GOAL_OBJ[task_a]
    object_b, _ = GOAL_OBJ[task_b]
    point_a = np.asarray(raw_env.sim.data.body_xpos[raw_env.obj_body_id[object_a]], dtype=np.float64)
    point_b = np.asarray(raw_env.sim.data.body_xpos[raw_env.obj_body_id[object_b]], dtype=np.float64)
    return {
        "batch_a": batch_a,
        "batch_b": batch_b,
        "prefix_a": prefix_a,
        "prefix_b": prefix_b,
        "cap_a": cap_a,
        "cap_b": cap_b,
        "image": image,
        "instruction": instruction,
        "noninstruction": noninstruction,
        "other": other,
        "point_a": point_a,
        "point_b": point_b,
        "n_img_slots": n_img_slots,
    }


def normalized_action_metrics(
    chunk: np.ndarray,
    clean_a: np.ndarray,
    clean_b: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    steps: int,
) -> dict[str, Any]:
    denominator = float(np.linalg.norm(clean_b[:steps] - clean_a[:steps]))
    if denominator <= 1e-9:
        raise RuntimeError("clean action endpoints collapsed")
    d_a = float(np.linalg.norm(chunk[:steps] - clean_a[:steps]) / denominator)
    d_b = float(np.linalg.norm(chunk[:steps] - clean_b[:steps]) / denominator)
    axis_a = axis_metric(clean_a, point_a, point_b, steps)
    axis_b = axis_metric(clean_b, point_a, point_b, steps)
    return {
        "D_A": d_a,
        "D_B": d_b,
        "preference_A": d_b - d_a,
        "axis_R": float(
            (axis_metric(chunk, point_a, point_b, steps) - axis_a)
            / (axis_b - axis_a if abs(axis_b - axis_a) > 1e-9 else 1e-9)
        ),
        "chunk10": np.asarray(chunk[:steps]).round(7).tolist(),
    }


def normalized_state_metrics(
    prefix: dict[str, Any],
    clean_a: dict[str, Any],
    clean_b: dict[str, Any],
    image: list[int],
    layers: Iterable[int],
) -> dict[str, Any]:
    rows = []
    for layer in layers:
        state = prefix["resid"][layer][0, image].float()
        state_a = clean_a["resid"][layer][0, image].float()
        state_b = clean_b["resid"][layer][0, image].float()
        denominator = float(torch.linalg.vector_norm(state_b - state_a))
        d_a = float(torch.linalg.vector_norm(state - state_a) / max(denominator, 1e-12))
        d_b = float(torch.linalg.vector_norm(state - state_b) / max(denominator, 1e-12))
        rows.append({"layer": int(layer), "D_A": d_a, "D_B": d_b, "preference_A": d_b - d_a})
    return {
        "layers": rows,
        "median_D_A": median(row["D_A"] for row in rows),
        "median_D_B": median(row["D_B"] for row in rows),
        "median_preference_A": median(row["preference_A"] for row in rows),
    }


def metric_row(
    *,
    stage: str,
    cell: str,
    tasks: tuple[int, int, int],
    init_id: int,
    condition: str,
    chunk: np.ndarray,
    clean_a_chunk: np.ndarray,
    clean_b_chunk: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    action_steps: int,
    prefix: dict[str, Any] | None = None,
    clean_a_prefix: dict[str, Any] | None = None,
    clean_b_prefix: dict[str, Any] | None = None,
    image: list[int] | None = None,
    state_layers: Iterable[int] = (),
) -> dict[str, Any]:
    row = {
        "stage": stage,
        "cell": cell,
        "task_a": tasks[0],
        "task_b": tasks[1],
        "task_c": tasks[2],
        "init": init_id,
        "condition": condition,
    }
    row.update(normalized_action_metrics(chunk, clean_a_chunk, clean_b_chunk, point_a, point_b, action_steps))
    if prefix is not None:
        assert clean_a_prefix is not None and clean_b_prefix is not None and image is not None
        row["late_image_state"] = normalized_state_metrics(
            prefix, clean_a_prefix, clean_b_prefix, image, state_layers
        )
    return row


def writer_hook(
    donor: AttentionCapture,
    source: list[int],
    receivers: dict[int, list[int]],
    *,
    rescue_heads: list[int] | None = None,
    rescue_positions: list[int] | None = None,
) -> SourceKVSubstitution:
    allow = None
    if rescue_heads is not None and rescue_positions is not None:
        allow = {0: [AllowCurrent(tuple(rescue_positions), tuple(rescue_heads))]}
    return SourceKVSubstitution(
        donor_keys=donor.keys,
        donor_values=donor.values,
        source_positions=source,
        receiver_positions_by_layer=receivers,
        allow_current=allow,
    )


def edge_map(rows: list[dict[str, Any]], key: str) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for row in rows:
        result.setdefault(int(row["layer"]), []).append(int(row["head"]))
    return {layer: sorted(set(heads)) for layer, heads in result.items()}


def reader_hook(
    donor_keys: dict[int, torch.Tensor],
    donor_values: dict[int, torch.Tensor],
    image: list[int],
    chunk_size: int,
    layers: list[int],
    *,
    active_edges: dict[int, list[int]] | None = None,
    rescue_edges: dict[int, list[int]] | None = None,
) -> SourceKVSubstitution:
    active_layers = layers if active_edges is None else sorted(active_edges)
    heads = None if active_edges is None else active_edges
    allow = None
    if rescue_edges:
        allow = {
            layer: [AllowCurrent(tuple(range(chunk_size)), tuple(selected_heads))]
            for layer, selected_heads in rescue_edges.items()
        }
    return SourceKVSubstitution(
        donor_keys=donor_keys,
        donor_values=donor_values,
        source_positions=image,
        receiver_positions_by_layer={layer: list(range(chunk_size)) for layer in active_layers},
        heads_by_layer=heads,
        allow_current=allow,
    )


def random_control(items: list[Any], size: int, seed: int, selected: list[Any]) -> list[Any]:
    rng = random.Random(seed)
    if size >= len(items):
        return list(items)
    complement = [item for item in items if item not in set(selected)]
    if len(complement) >= size:
        return sorted(rng.sample(complement, size))
    # When the selected set occupies more than half the universe, minimize
    # overlap first and sample only the unavoidable remainder from selected.
    candidate = list(complement)
    candidate.extend(rng.sample(selected, size - len(candidate)))
    candidate = sorted(candidate)
    if candidate == sorted(selected):
        raise RuntimeError("could not draw a distinct matched-size random control")
    return candidate


def select_by_fraction(rows: list[dict[str, Any]], fraction: float, value_key: str) -> list[dict[str, Any]]:
    if not rows or sum(float(row[value_key]) for row in rows) <= 0:
        raise RuntimeError("selection effect is identically zero")
    for index, row in enumerate(rows, start=1):
        if float(row["cumulative_squared_fraction"]) >= fraction:
            return rows[:index]
    return rows


def append_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            handle.flush()


def completed(path: Path, required_conditions: Iterable[str]) -> set[tuple[str, int]]:
    result: set[tuple[str, int]] = set()
    if not path.exists():
        return result
    grouped: dict[tuple[str, int], set[str]] = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            grouped.setdefault((row["cell"], int(row["init"])), set()).add(row["condition"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    required = set(required_conditions)
    for key, conditions in grouped.items():
        if conditions.issuperset(required):
            result.add(key)
    return result


def run_preflight(args: argparse.Namespace, cfg: dict[str, Any], pairs: list[list[int]]) -> None:
    output = args.out / "preflight.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    suite = _get_suite(cfg["suite"])
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    task_a, task_b, task_c = pairs[0]
    env = make_env(suite, cfg["suite"], task_a)
    try:
        pair = capture_pair(
            harness, env, prompts, task_a, task_b, cfg["splits"]["preflight_ids"][0],
            capture_residuals=True, capture_attention=True,
        )
        all_layers = cfg["fixed"]["prefix_layers"]
        identity = harness.prefix_forward(
            pair["batch_b"],
            capture=True,
            attn_hook=writer_hook(
                pair["cap_b"], pair["instruction"],
                {layer: pair["noninstruction"] for layer in all_layers},
            ),
        )
        prefix_identity_exact = prefix_equal(pair["prefix_b"], identity)
        noise = harness.make_noise(seed=100 * cfg["splits"]["preflight_ids"][0])
        clean_b = action_chunk(harness, pair["prefix_b"], noise)
        identity_b = action_chunk(harness, identity, noise)
        action_identity_exact = bool(np.array_equal(clean_b, identity_b))
        if not prefix_identity_exact or not action_identity_exact:
            raise RuntimeError(
                f"real-model identity failed: prefix={prefix_identity_exact}, action={action_identity_exact}"
            )
        reader_layers = cfg["fixed"]["reader_layers"]
        donor_keys, donor_values = cache_dict(pair["prefix_a"], reader_layers)
        effect = SourceEffectCapture(
            donor_keys=donor_keys,
            donor_values=donor_values,
            source_positions=pair["image"],
            layers=reader_layers,
        )
        measured_b = action_chunk(harness, pair["prefix_b"], noise, attn_hook=effect)
        measurement_identity_exact = bool(np.array_equal(clean_b, measured_b))
        if not measurement_identity_exact:
            raise RuntimeError("reader measurement hook changed the live forward")
        result = {
            "status": "pass",
            "cell": cell_name(cfg["suite"], task_a, task_b),
            "task_c": task_c,
            "init": cfg["splits"]["preflight_ids"][0],
            "n_img_slots": pair["n_img_slots"],
            "n_img_valid": len(pair["image"]),
            "instruction_positions": pair["instruction"],
            "prefix_identity_bitwise_exact": prefix_identity_exact,
            "action_identity_bitwise_exact": action_identity_exact,
            "measurement_hook_identity_bitwise_exact": measurement_identity_exact,
            "prefix_query_shapes": {str(k): list(v) for k, v in pair["cap_b"].query_shapes.items()},
            "prefix_kv_shapes": {str(k): list(v.shape) for k, v in pair["cap_b"].keys.items()},
            "reader_effect_calls": {str(k): v for k, v in effect.calls.items()},
            "reader_effect_shapes": {str(k): list(v.shape) for k, v in effect.squared_effect.items()},
            "module_paths": harness.module_paths,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json(output, result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


def run_select_writer(args: argparse.Namespace, cfg: dict[str, Any], pairs: list[list[int]]) -> None:
    output = args.out / "writer_selection.json"
    rows_path = args.out / "writer_selection_rows.jsonl"
    if output.exists() or rows_path.exists():
        raise FileExistsError("writer selection outputs already exist")
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    suite = _get_suite(cfg["suite"])
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    rows: list[dict[str, Any]] = []
    totals: dict[int, float] = {}
    query_heads = None
    ids = cfg["splits"]["writer_select_ids"]
    for cell_index, (task_a, task_b, task_c) in enumerate(pairs):
        env = make_env(suite, cfg["suite"], task_a)
        cell = cell_name(cfg["suite"], task_a, task_b)
        print(f"[select-writer] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in ids:
                pair = capture_pair(
                    harness, env, prompts, task_a, task_b, init_id,
                    capture_residuals=False, capture_attention=True,
                )
                ranked = ranked_heads_by_write(
                    pair["cap_a"].outputs[0], pair["cap_b"].outputs[0],
                    pair["cap_b"].query_shapes[0], pair["image"],
                )
                query_heads = pair["cap_b"].query_heads[0]
                cell_rows = []
                for item in ranked:
                    score = float(item["write_norm"]) ** 2
                    totals[int(item["head"])] = totals.get(int(item["head"]), 0.0) + score
                    cell_rows.append({
                        "stage": "select-writer", "cell": cell, "task_a": task_a,
                        "task_b": task_b, "task_c": task_c, "init": init_id,
                        "head": int(item["head"]), "squared_write": score,
                    })
                append_rows(rows_path, cell_rows)
                rows.extend(cell_rows)
                del pair
                gc.collect()
        finally:
            env.close()
    total = sum(totals.values())
    cumulative = 0.0
    aggregate = []
    for rank, (head, score) in enumerate(sorted(totals.items(), key=lambda item: (-item[1], item[0])), start=1):
        cumulative += score
        aggregate.append({
            "rank": rank, "head": head, "squared_write": score,
            "write_norm": score**0.5,
            "cumulative_squared_fraction": cumulative / max(total, 1e-30),
        })
    selected_rows = select_by_fraction(
        aggregate, cfg["fixed"]["writer_energy_fraction"], "squared_write"
    )
    selected = [int(row["head"]) for row in selected_rows]
    random_heads = random_control(
        list(range(int(query_heads))), len(selected), cfg["fixed"]["random_seed"], selected
    )
    result = {
        "status": "complete",
        "selection_is_behavior_blind": True,
        "ids": ids,
        "n_cells": len(pairs),
        "n_rows": len(rows),
        "query_heads": query_heads,
        "energy_fraction": cfg["fixed"]["writer_energy_fraction"],
        "selected_heads": selected,
        "random_heads": random_heads,
        "compact": len(selected) <= cfg["fixed"]["writer_compact_max_heads"],
        "ranking": aggregate,
        "rows_sha256": sha256(rows_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


def run_select_reader(args: argparse.Namespace, cfg: dict[str, Any], pairs: list[list[int]]) -> None:
    output = args.out / "reader_selection.json"
    rows_path = args.out / "reader_selection_rows.jsonl"
    if output.exists() or rows_path.exists():
        raise FileExistsError("reader selection outputs already exist")
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    suite = _get_suite(cfg["suite"])
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    ids = cfg["splits"]["reader_select_ids"]
    layers = cfg["fixed"]["reader_layers"]
    totals: dict[tuple[int, int], float] = {}
    total_calls: dict[int, int] = {}
    audit_rows: list[dict[str, Any]] = []
    for cell_index, (task_a, task_b, task_c) in enumerate(pairs):
        env = make_env(suite, cfg["suite"], task_a)
        cell = cell_name(cfg["suite"], task_a, task_b)
        print(f"[select-reader] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in ids:
                pair = capture_pair(
                    harness, env, prompts, task_a, task_b, init_id,
                    capture_residuals=False, capture_attention=False,
                )
                donor_keys, donor_values = cache_dict(pair["prefix_a"], layers)
                effect = SourceEffectCapture(
                    donor_keys=donor_keys, donor_values=donor_values,
                    source_positions=pair["image"], layers=layers,
                )
                noise = harness.make_noise(seed=100 * init_id)
                action_chunk(harness, pair["prefix_b"], noise, attn_hook=effect)
                for layer, scores in effect.squared_effect.items():
                    total_calls[layer] = total_calls.get(layer, 0) + effect.calls[layer]
                    for head, score in enumerate(scores.tolist()):
                        totals[(layer, head)] = totals.get((layer, head), 0.0) + float(score)
                        audit_rows.append({
                            "stage": "select-reader", "cell": cell, "task_a": task_a,
                            "task_b": task_b, "task_c": task_c, "init": init_id,
                            "layer": layer, "head": head, "squared_effect": float(score),
                            "denoise_calls": effect.calls[layer],
                        })
                append_rows(rows_path, audit_rows[-sum(len(v) for v in effect.squared_effect.values()):])
                del pair, donor_keys, donor_values
                gc.collect()
        finally:
            env.close()
    total = sum(totals.values())
    cumulative = 0.0
    ranking = []
    for rank, ((layer, head), score) in enumerate(
        sorted(totals.items(), key=lambda item: (-item[1], item[0])), start=1
    ):
        cumulative += score
        ranking.append({
            "rank": rank, "layer": layer, "head": head,
            "squared_effect": score, "write_norm": score**0.5,
            "cumulative_squared_fraction": cumulative / max(total, 1e-30),
            "denoise_calls": total_calls[layer],
        })
    selected_rows = select_by_fraction(
        ranking, cfg["fixed"]["reader_energy_fraction"], "squared_effect"
    )
    selected = sorted((int(row["layer"]), int(row["head"])) for row in selected_rows)
    universe = sorted((int(row["layer"]), int(row["head"])) for row in ranking)
    random_edges = random_control(
        universe, len(selected), cfg["fixed"]["random_seed"] + 1, selected
    )
    result = {
        "status": "complete",
        "selection_is_behavior_blind": True,
        "ids": ids,
        "n_cells": len(pairs),
        "n_rows": len(audit_rows),
        "energy_fraction": cfg["fixed"]["reader_energy_fraction"],
        "selected_edges": [{"layer": layer, "head": head} for layer, head in selected],
        "random_edges": [{"layer": layer, "head": head} for layer, head in random_edges],
        "compact": len(selected) <= cfg["fixed"]["reader_compact_max_edges"],
        "ranking": ranking,
        "rows_sha256": sha256(rows_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


def writer_conditions(
    harness: Pi05Harness,
    pair: dict[str, Any],
    cfg: dict[str, Any],
    selected_heads: list[int],
    random_heads: list[int],
) -> dict[str, dict[str, Any]]:
    layers = cfg["fixed"]["prefix_layers"]
    all_heads = list(range(pair["cap_b"].query_heads[0]))
    all_noninstruction = {layer: pair["noninstruction"] for layer in layers}
    conditions = {
        "clean_A": pair["prefix_a"],
        "clean_B": pair["prefix_b"],
        "identity_B": harness.prefix_forward(
            pair["batch_b"], capture=True,
            attn_hook=writer_hook(pair["cap_b"], pair["instruction"], all_noninstruction),
        ),
        "writer_block_all_noninstruction": harness.prefix_forward(
            pair["batch_b"], capture=True,
            attn_hook=writer_hook(pair["cap_a"], pair["instruction"], all_noninstruction),
        ),
        "writer_block_image": harness.prefix_forward(
            pair["batch_b"], capture=True,
            attn_hook=writer_hook(
                pair["cap_a"], pair["instruction"], {layer: pair["image"] for layer in layers}
            ),
        ),
        "writer_block_other": harness.prefix_forward(
            pair["batch_b"], capture=True,
            attn_hook=writer_hook(
                pair["cap_a"], pair["instruction"], {layer: pair["other"] for layer in layers}
            ),
        ),
        "writer_rescue_l0_image_all_heads": harness.prefix_forward(
            pair["batch_b"], capture=True,
            attn_hook=writer_hook(
                pair["cap_a"], pair["instruction"], all_noninstruction,
                rescue_heads=all_heads, rescue_positions=pair["image"],
            ),
        ),
        "writer_rescue_l0_image_selected": harness.prefix_forward(
            pair["batch_b"], capture=True,
            attn_hook=writer_hook(
                pair["cap_a"], pair["instruction"], all_noninstruction,
                rescue_heads=selected_heads, rescue_positions=pair["image"],
            ),
        ),
        "writer_rescue_l0_image_random": harness.prefix_forward(
            pair["batch_b"], capture=True,
            attn_hook=writer_hook(
                pair["cap_a"], pair["instruction"], all_noninstruction,
                rescue_heads=random_heads, rescue_positions=pair["image"],
            ),
        ),
        "writer_rescue_l0_other_all_heads": harness.prefix_forward(
            pair["batch_b"], capture=True,
            attn_hook=writer_hook(
                pair["cap_a"], pair["instruction"], all_noninstruction,
                rescue_heads=all_heads, rescue_positions=pair["other"],
            ),
        ),
    }
    if not prefix_equal(pair["prefix_b"], conditions["identity_B"]):
        raise RuntimeError("writer identity prefix is not bitwise exact")
    return conditions


def screen_summary(rows: list[dict[str, Any]], kind: str, cfg: dict[str, Any]) -> dict[str, Any]:
    def values(condition: str, key: str = "preference_A") -> list[float]:
        return [float(row[key]) for row in rows if row["condition"] == condition]

    if kind == "writer":
        selected_name = "writer_rescue_l0_image_selected"
        random_name = "writer_rescue_l0_image_random"
        block_name = "writer_block_all_noninstruction"
        block_a = median(values(block_name))
        rescue_b = -median(values(selected_name))
        state_rescue_b = -median(
            row["late_image_state"]["median_preference_A"]
            for row in rows if row["condition"] == selected_name
        )
        selected_by_cell = {}
        random_by_cell = {}
        for row in rows:
            key = row["cell"]
            if row["condition"] == selected_name:
                selected_by_cell.setdefault(key, []).append(float(row["D_B"]))
            elif row["condition"] == random_name:
                random_by_cell.setdefault(key, []).append(float(row["D_B"]))
        wins = sum(
            median(selected_by_cell[cell]) < median(random_by_cell[cell])
            for cell in selected_by_cell if cell in random_by_cell
        )
        gate = cfg["gates"]["writer"]
        passed = (
            block_a >= gate["block_all_noninstruction_median_preference_for_A_min"]
            and rescue_b >= gate["selected_rescue_median_preference_for_B_min"]
            and state_rescue_b >= gate["selected_rescue_internal_state_preference_for_B_min"]
            and wins >= gate["selected_rescue_beats_random_cell_directions_min"]
        )
        metrics = {
            "block_median_preference_for_A": block_a,
            "selected_rescue_median_preference_for_B": rescue_b,
            "selected_rescue_internal_state_preference_for_B": state_rescue_b,
            "selected_rescue_beats_random_cell_directions": wins,
        }
    else:
        selected_name = "reader_rescue_selected"
        random_name = "reader_rescue_random"
        block_name = "reader_block"
        block_a = median(values(block_name))
        rescue_b = -median(values(selected_name))
        selected_by_cell = {}
        random_by_cell = {}
        for row in rows:
            key = row["cell"]
            if row["condition"] == selected_name:
                selected_by_cell.setdefault(key, []).append(float(row["D_B"]))
            elif row["condition"] == random_name:
                random_by_cell.setdefault(key, []).append(float(row["D_B"]))
        wins = sum(
            median(selected_by_cell[cell]) < median(random_by_cell[cell])
            for cell in selected_by_cell if cell in random_by_cell
        )
        gate = cfg["gates"]["reader"]
        passed = (
            block_a >= gate["block_median_preference_for_A_min"]
            and rescue_b >= gate["selected_rescue_median_preference_for_B_min"]
            and wins >= gate["selected_rescue_beats_random_cell_directions_min"]
        )
        metrics = {
            "block_median_preference_for_A": block_a,
            "selected_rescue_median_preference_for_B": rescue_b,
            "selected_rescue_beats_random_cell_directions": wins,
        }
    return {"status": "pass" if passed else "fail", "metrics": metrics, "gate": gate}


def run_screen_writer(args: argparse.Namespace, cfg: dict[str, Any], pairs: list[list[int]]) -> None:
    selection = json.loads((args.out / "writer_selection.json").read_text())
    rows_path = args.out / "writer_screen_rows.jsonl"
    required_conditions = {
        "clean_A", "clean_B", "identity_B", "writer_block_all_noninstruction",
        "writer_block_image", "writer_block_other", "writer_rescue_l0_image_all_heads",
        "writer_rescue_l0_image_selected", "writer_rescue_l0_image_random",
        "writer_rescue_l0_other_all_heads",
    }
    done = completed(rows_path, required_conditions)
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    suite = _get_suite(cfg["suite"])
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    for cell_index, (task_a, task_b, task_c) in enumerate(pairs):
        cell = cell_name(cfg["suite"], task_a, task_b)
        env = make_env(suite, cfg["suite"], task_a)
        print(f"[screen-writer] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in cfg["splits"]["causal_screen_ids"]:
                if (cell, init_id) in done:
                    continue
                pair = capture_pair(
                    harness, env, prompts, task_a, task_b, init_id,
                    capture_residuals=True, capture_attention=True,
                )
                conditions = writer_conditions(
                    harness, pair, cfg, selection["selected_heads"], selection["random_heads"]
                )
                noise = harness.make_noise(seed=100 * init_id)
                chunks = {name: action_chunk(harness, prefix, noise) for name, prefix in conditions.items()}
                if not np.array_equal(chunks["clean_B"], chunks["identity_B"]):
                    raise RuntimeError("writer identity action is not bitwise exact")
                rows = [metric_row(
                    stage="screen-writer", cell=cell, tasks=(task_a, task_b, task_c),
                    init_id=init_id, condition=name, chunk=chunk,
                    clean_a_chunk=chunks["clean_A"], clean_b_chunk=chunks["clean_B"],
                    point_a=pair["point_a"], point_b=pair["point_b"],
                    action_steps=cfg["fixed"]["action_steps_for_metric"], prefix=conditions[name],
                    clean_a_prefix=pair["prefix_a"], clean_b_prefix=pair["prefix_b"],
                    image=pair["image"], state_layers=cfg["fixed"]["late_state_layers"],
                ) for name, chunk in chunks.items()]
                append_rows(rows_path, rows)
                done.add((cell, init_id))
                del pair, conditions, chunks, rows
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            env.close()
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    summary = screen_summary(rows, "writer", cfg)
    summary.update({"rows": len(rows), "rows_sha256": sha256(rows_path)})
    write_json(args.out / "writer_screen_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def selected_reader_edges(path: Path, name: str) -> dict[int, list[int]]:
    selection = json.loads(path.read_text())
    return edge_map(selection[name], name)


def run_screen_reader(args: argparse.Namespace, cfg: dict[str, Any], pairs: list[list[int]]) -> None:
    selected = selected_reader_edges(args.out / "reader_selection.json", "selected_edges")
    random_edges = selected_reader_edges(args.out / "reader_selection.json", "random_edges")
    rows_path = args.out / "reader_screen_rows.jsonl"
    required_conditions = {
        "clean_A", "clean_B", "reader_block", "reader_rescue_selected", "reader_rescue_random",
    }
    done = completed(rows_path, required_conditions)
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    suite = _get_suite(cfg["suite"])
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    layers = cfg["fixed"]["reader_layers"]
    for cell_index, (task_a, task_b, task_c) in enumerate(pairs):
        cell = cell_name(cfg["suite"], task_a, task_b)
        env = make_env(suite, cfg["suite"], task_a)
        print(f"[screen-reader] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in cfg["splits"]["causal_screen_ids"]:
                if (cell, init_id) in done:
                    continue
                pair = capture_pair(
                    harness, env, prompts, task_a, task_b, init_id,
                    capture_residuals=False, capture_attention=False,
                )
                noise = harness.make_noise(seed=100 * init_id)
                clean_a = action_chunk(harness, pair["prefix_a"], noise)
                clean_b = action_chunk(harness, pair["prefix_b"], noise)
                donor_keys, donor_values = cache_dict(pair["prefix_a"], layers)
                hooks = {
                    "reader_block": reader_hook(
                        donor_keys, donor_values, pair["image"], harness.cfg.chunk_size, layers
                    ),
                    "reader_rescue_selected": reader_hook(
                        donor_keys, donor_values, pair["image"], harness.cfg.chunk_size, layers,
                        rescue_edges=selected,
                    ),
                    "reader_rescue_random": reader_hook(
                        donor_keys, donor_values, pair["image"], harness.cfg.chunk_size, layers,
                        rescue_edges=random_edges,
                    ),
                }
                chunks = {"clean_A": clean_a, "clean_B": clean_b}
                chunks.update({
                    name: action_chunk(harness, pair["prefix_b"], noise, attn_hook=hook)
                    for name, hook in hooks.items()
                })
                rows = [metric_row(
                    stage="screen-reader", cell=cell, tasks=(task_a, task_b, task_c),
                    init_id=init_id, condition=name, chunk=chunk,
                    clean_a_chunk=clean_a, clean_b_chunk=clean_b,
                    point_a=pair["point_a"], point_b=pair["point_b"],
                    action_steps=cfg["fixed"]["action_steps_for_metric"],
                ) for name, chunk in chunks.items()]
                append_rows(rows_path, rows)
                done.add((cell, init_id))
                del pair, chunks, rows, donor_keys, donor_values
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            env.close()
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    summary = screen_summary(rows, "reader", cfg)
    summary.update({"rows": len(rows), "rows_sha256": sha256(rows_path)})
    write_json(args.out / "reader_screen_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def run_confirm(args: argparse.Namespace, cfg: dict[str, Any], pairs: list[list[int]]) -> None:
    for name in ("writer_screen_summary.json", "reader_screen_summary.json"):
        summary = json.loads((args.out / name).read_text())
        if summary["status"] != "pass":
            raise RuntimeError(f"{name} failed; confirmation remains unopened")
    writer = json.loads((args.out / "writer_selection.json").read_text())
    reader_selected = selected_reader_edges(args.out / "reader_selection.json", "selected_edges")
    reader_random = selected_reader_edges(args.out / "reader_selection.json", "random_edges")
    rows_path = args.out / "confirmation_rows.jsonl"
    required_conditions = {
        "clean_A", "clean_B", "writer_block", "writer_rescue_selected", "writer_rescue_random",
        "reader_block", "reader_rescue_selected", "reader_rescue_random",
        "writer_rescue_then_reader_block", "downstream_path_rescue_selected",
        "downstream_path_rescue_random",
    }
    done = completed(rows_path, required_conditions)
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    suite = _get_suite(cfg["suite"])
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    prefix_layers = cfg["fixed"]["prefix_layers"]
    reader_layers = cfg["fixed"]["reader_layers"]
    for cell_index, (task_a, task_b, task_c) in enumerate(pairs):
        cell = cell_name(cfg["suite"], task_a, task_b)
        env = make_env(suite, cfg["suite"], task_a)
        print(f"[confirm] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in cfg["splits"]["confirmation_ids"]:
                if (cell, init_id) in done:
                    continue
                pair = capture_pair(
                    harness, env, prompts, task_a, task_b, init_id,
                    capture_residuals=True, capture_attention=True,
                )
                all_noninstruction = {layer: pair["noninstruction"] for layer in prefix_layers}
                writer_prefixes = {
                    "writer_block": harness.prefix_forward(
                        pair["batch_b"], capture=True,
                        attn_hook=writer_hook(pair["cap_a"], pair["instruction"], all_noninstruction),
                    ),
                    "writer_rescue_selected": harness.prefix_forward(
                        pair["batch_b"], capture=True,
                        attn_hook=writer_hook(
                            pair["cap_a"], pair["instruction"], all_noninstruction,
                            rescue_heads=writer["selected_heads"], rescue_positions=pair["image"],
                        ),
                    ),
                    "writer_rescue_random": harness.prefix_forward(
                        pair["batch_b"], capture=True,
                        attn_hook=writer_hook(
                            pair["cap_a"], pair["instruction"], all_noninstruction,
                            rescue_heads=writer["random_heads"], rescue_positions=pair["image"],
                        ),
                    ),
                }
                noise = harness.make_noise(seed=100 * init_id)
                clean_a = action_chunk(harness, pair["prefix_a"], noise)
                clean_b = action_chunk(harness, pair["prefix_b"], noise)
                donor_a_k, donor_a_v = cache_dict(pair["prefix_a"], reader_layers)
                donor_b_k, donor_b_v = cache_dict(pair["prefix_b"], reader_layers)
                reader_block = reader_hook(
                    donor_a_k, donor_a_v, pair["image"], harness.cfg.chunk_size, reader_layers
                )
                conditions: dict[str, tuple[dict[str, Any], Any]] = {
                    "clean_A": (pair["prefix_a"], None),
                    "clean_B": (pair["prefix_b"], None),
                    "writer_block": (writer_prefixes["writer_block"], None),
                    "writer_rescue_selected": (writer_prefixes["writer_rescue_selected"], None),
                    "writer_rescue_random": (writer_prefixes["writer_rescue_random"], None),
                    "reader_block": (pair["prefix_b"], reader_block),
                    "reader_rescue_selected": (
                        pair["prefix_b"], reader_hook(
                            donor_a_k, donor_a_v, pair["image"], harness.cfg.chunk_size,
                            reader_layers, rescue_edges=reader_selected,
                        ),
                    ),
                    "reader_rescue_random": (
                        pair["prefix_b"], reader_hook(
                            donor_a_k, donor_a_v, pair["image"], harness.cfg.chunk_size,
                            reader_layers, rescue_edges=reader_random,
                        ),
                    ),
                    "writer_rescue_then_reader_block": (
                        writer_prefixes["writer_rescue_selected"], reader_block,
                    ),
                    "downstream_path_rescue_selected": (
                        writer_prefixes["writer_block"], reader_hook(
                            donor_b_k, donor_b_v, pair["image"], harness.cfg.chunk_size,
                            reader_layers, active_edges=reader_selected,
                        ),
                    ),
                    "downstream_path_rescue_random": (
                        writer_prefixes["writer_block"], reader_hook(
                            donor_b_k, donor_b_v, pair["image"], harness.cfg.chunk_size,
                            reader_layers, active_edges=reader_random,
                        ),
                    ),
                }
                chunks = {
                    name: action_chunk(harness, prefix, noise, attn_hook=hook)
                    for name, (prefix, hook) in conditions.items()
                }
                rows = []
                for name, chunk in chunks.items():
                    prefix = conditions[name][0]
                    include_state = name.startswith("writer_") and name != "writer_rescue_then_reader_block"
                    rows.append(metric_row(
                        stage="confirm", cell=cell, tasks=(task_a, task_b, task_c),
                        init_id=init_id, condition=name, chunk=chunk,
                        clean_a_chunk=clean_a, clean_b_chunk=clean_b,
                        point_a=pair["point_a"], point_b=pair["point_b"],
                        action_steps=cfg["fixed"]["action_steps_for_metric"],
                        prefix=prefix if include_state else None,
                        clean_a_prefix=pair["prefix_a"] if include_state else None,
                        clean_b_prefix=pair["prefix_b"] if include_state else None,
                        image=pair["image"] if include_state else None,
                        state_layers=cfg["fixed"]["late_state_layers"],
                    ))
                append_rows(rows_path, rows)
                done.add((cell, init_id))
                del pair, writer_prefixes, conditions, chunks, rows
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            env.close()
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in sorted({row["condition"] for row in rows})
    }
    summary = {
        "status": "complete",
        "rows": len(rows),
        "rows_sha256": sha256(rows_path),
        "conditions": {
            name: {
                "median_D_A": median(row["D_A"] for row in subset),
                "median_D_B": median(row["D_B"] for row in subset),
                "median_preference_A": median(row["preference_A"] for row in subset),
                "A_like_fraction": float(np.mean([row["D_A"] < row["D_B"] for row in subset])),
                "B_like_fraction": float(np.mean([row["D_B"] < row["D_A"] for row in subset])),
            }
            for name, subset in by_condition.items()
        },
        "note": "Closed-loop behavior remains unopened until these immediate-action results are evaluated against the frozen gate.",
    }
    write_json(args.out / "confirmation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def ensure_manifest(args: argparse.Namespace, cfg: dict[str, Any], pairs_path: Path, pairs: list[list[int]]) -> None:
    manifest_path = args.out / "manifest.json"
    hashes = {
        "config": sha256(args.config),
        "runner": sha256(Path(__file__)),
        "attention_pathway": sha256(Path(__file__).with_name("attention_pathway.py")),
        "hooks": sha256(Path(__file__).with_name("hooks.py")),
        "pairs": sha256(pairs_path),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("hashes") != hashes:
            raise RuntimeError("sealed code/config hashes changed; use a new output directory")
        return
    write_json(manifest_path, {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg,
        "pairs": pairs,
        "hashes": hashes,
        "args_at_creation": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=("preflight", "select-writer", "select-reader", "screen-writer", "screen-reader", "confirm"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pi05_attention_pathway.json"))
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    pairs_path = Path(cfg["pairs_file"])
    pairs = json.loads(pairs_path.read_text())
    if len(pairs) != 12 or any(len(pair) != 3 for pair in pairs):
        raise RuntimeError("expected exactly 12 directed [A, B, C] cells")
    args.out.mkdir(parents=True, exist_ok=True)
    require_free_space(args.out, args.minimum_free_gib)
    ensure_manifest(args, cfg, pairs_path, pairs)
    stages = {
        "preflight": run_preflight,
        "select-writer": run_select_writer,
        "select-reader": run_select_reader,
        "screen-writer": run_screen_writer,
        "screen-reader": run_screen_reader,
        "confirm": run_confirm,
    }
    stages[args.stage](args, cfg, pairs)


if __name__ == "__main__":
    main()
