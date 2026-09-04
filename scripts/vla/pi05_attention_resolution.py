#!/usr/bin/env python
"""Resolve pi0.5's attention writer curve and minimum causal reader size.

This is a prospective follow-up to ``pi05_attention_pathway.py``. Development
and held-out prompt-pair stages are separate, and each holdout command requires
its corresponding frozen development gate to pass.
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
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
from robosuite.utils import binding_utils

from attention_pathway import AllowCurrent, AttentionCapture, SourceKVSubstitution
from build_stage1_twins import _patched_read_pixels
from hooks import Pi05Harness, axis_metric, cache_kv_lists
from lerobot.envs.libero import LiberoEnv, _get_suite
from stage2_discovery import GOAL_OBJ

binding_utils.MjRenderContext.read_pixels = _patched_read_pixels

OBJECT_TASKS = {
    0: ("alphabet_soup_1", "alphabet soup"),
    1: ("cream_cheese_1", "cream cheese"),
    2: ("salad_dressing_1", "salad dressing"),
    3: ("bbq_sauce_1", "bbq sauce"),
    4: ("ketchup_1", "ketchup"),
    5: ("tomato_sauce_1", "tomato sauce"),
    6: ("butter_1", "butter"),
    7: ("milk_1", "milk"),
    8: ("chocolate_pudding_1", "chocolate pudding"),
    9: ("orange_juice_1", "orange juice"),
}


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


def task_object(suite: str, task_id: int) -> tuple[str, str]:
    if suite == "libero_goal":
        return GOAL_OBJ[task_id]
    if suite == "libero_object":
        return OBJECT_TASKS[task_id]
    raise KeyError(f"unsupported suite {suite}")


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


def prefix_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_k, left_v = cache_kv_lists(left["cache"])
    right_k, right_v = cache_kv_lists(right["cache"])
    if not all(torch.equal(a, b) for a, b in zip(left_k + left_v, right_k + right_v)):
        return False
    return set(left["resid"]) == set(right["resid"]) and all(
        torch.equal(left["resid"][layer], right["resid"][layer]) for layer in left["resid"]
    )


def capture_pair(
    harness: Pi05Harness,
    suite_name: str,
    env: LiberoEnv,
    prompts: list[str],
    task_a: int,
    task_b: int,
    init_id: int,
    *,
    residuals: bool,
) -> dict[str, Any]:
    env.init_state_id = init_id
    observation, _ = env.reset(seed=1000 + init_id)
    batch_a = harness.build_batch(observation, prompts[task_a])
    batch_b = harness.build_batch(observation, prompts[task_b])
    emb_a, _, _, n_img_slots, n_img_valid = harness._prefix_embed(batch_a)
    emb_b, _, _, slots_b, valid_b = harness._prefix_embed(batch_b)
    if (n_img_slots, n_img_valid) != (slots_b, valid_b) or n_img_valid != 512:
        raise RuntimeError(f"invalid image layout: {(n_img_slots, n_img_valid)} / {(slots_b, valid_b)}")
    object_a, name_a = task_object(suite_name, task_a)
    object_b, name_b = task_object(suite_name, task_b)
    seg_a, _ = harness.segments(batch_a, object_name=name_a, n_img=n_img_slots)
    seg_b, _ = harness.segments(batch_b, object_name=name_b, n_img=n_img_slots)
    if seg_a["INSTR"] != seg_b["INSTR"] or seg_a["TEXT_VALID"] != seg_b["TEXT_VALID"]:
        raise RuntimeError(
            f"pair {task_a}/{task_b} is not position matched: "
            f"INSTR={seg_a['INSTR']}/{seg_b['INSTR']} valid={len(seg_a['TEXT_VALID'])}/{len(seg_b['TEXT_VALID'])}"
        )
    image = list(range(n_img_valid))
    instruction = list(seg_a["INSTR"])
    noninstruction = sorted(set(image + seg_a["TEXT_VALID"]) - set(instruction))
    if not torch.equal(emb_a[:, noninstruction], emb_b[:, noninstruction]):
        maximum = float((emb_a[:, noninstruction].float() - emb_b[:, noninstruction].float()).abs().max())
        raise RuntimeError(f"non-instruction input embeddings differ, maxabs={maximum}")
    raw_env = env._env.env
    if object_a not in raw_env.obj_body_id or object_b not in raw_env.obj_body_id:
        raise RuntimeError(f"scene does not contain both targets: {object_a}, {object_b}")
    point_a = np.asarray(raw_env.sim.data.body_xpos[raw_env.obj_body_id[object_a]], dtype=np.float64)
    point_b = np.asarray(raw_env.sim.data.body_xpos[raw_env.obj_body_id[object_b]], dtype=np.float64)
    cap_a = AttentionCapture()
    cap_b = AttentionCapture()
    prefix_a = harness.prefix_forward(batch_a, capture=residuals, attn_hook=cap_a)
    prefix_b = harness.prefix_forward(batch_b, capture=residuals, attn_hook=cap_b)
    return {
        "batch_a": batch_a,
        "batch_b": batch_b,
        "prefix_a": prefix_a,
        "prefix_b": prefix_b,
        "cap_a": cap_a,
        "cap_b": cap_b,
        "instruction": instruction,
        "image": image,
        "point_a": point_a,
        "point_b": point_b,
    }


def writer_intervention(
    donor: AttentionCapture,
    instruction: list[int],
    image: list[int],
    active_layers: Iterable[int],
    *,
    allow_current_layers: Iterable[int] = (),
    n_heads: int = 8,
) -> SourceKVSubstitution:
    active = sorted(set(int(layer) for layer in active_layers))
    allow = {
        int(layer): [AllowCurrent(tuple(image), tuple(range(n_heads)))]
        for layer in sorted(set(int(layer) for layer in allow_current_layers))
    }
    return SourceKVSubstitution(
        donor_keys=donor.keys,
        donor_values=donor.values,
        source_positions=instruction,
        receiver_positions_by_layer={layer: image for layer in active},
        allow_current=allow,
    )


def edge_map(edges: Iterable[Iterable[int]]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for layer, head in edges:
        result.setdefault(int(layer), []).append(int(head))
    return {layer: sorted(set(heads)) for layer, heads in result.items()}


def reader_intervention(
    donor_keys: dict[int, torch.Tensor],
    donor_values: dict[int, torch.Tensor],
    image: list[int],
    chunk_size: int,
    layers: list[int],
    *,
    active_edges: Iterable[Iterable[int]] | None = None,
    rescue_edges: Iterable[Iterable[int]] = (),
) -> SourceKVSubstitution:
    active = None if active_edges is None else edge_map(active_edges)
    active_layers = layers if active is None else sorted(active)
    allow = {
        layer: [AllowCurrent(tuple(range(chunk_size)), tuple(heads))]
        for layer, heads in edge_map(rescue_edges).items()
    }
    return SourceKVSubstitution(
        donor_keys=donor_keys,
        donor_values=donor_values,
        source_positions=image,
        receiver_positions_by_layer={layer: list(range(chunk_size)) for layer in active_layers},
        heads_by_layer=active,
        allow_current=allow,
    )


def action_metrics(
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
        "clean_ab_l2": denominator,
        "chunk10": np.asarray(chunk[:steps]).round(7).tolist(),
    }


def state_metrics(
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
        rows.append({"layer": layer, "D_A": d_a, "D_B": d_b, "preference_A": d_b - d_a})
    return {
        "median_D_A": median(row["D_A"] for row in rows),
        "median_D_B": median(row["D_B"] for row in rows),
        "median_preference_A": median(row["preference_A"] for row in rows),
        "layers": rows,
    }


def row_payload(
    *,
    stage: str,
    suite: str,
    task_a: int,
    task_b: int,
    init_id: int,
    condition: str,
    chunk: np.ndarray,
    clean_a_chunk: np.ndarray,
    clean_b_chunk: np.ndarray,
    pair: dict[str, Any],
    action_steps: int,
    prefix: dict[str, Any] | None = None,
    state_layers: Iterable[int] = (),
) -> dict[str, Any]:
    row = {
        "stage": stage,
        "suite": suite,
        "cell": cell_name(suite, task_a, task_b),
        "task_a": task_a,
        "task_b": task_b,
        "init": init_id,
        "condition": condition,
    }
    row.update(action_metrics(
        chunk, clean_a_chunk, clean_b_chunk, pair["point_a"], pair["point_b"], action_steps
    ))
    if prefix is not None:
        row["late_image_state"] = state_metrics(
            prefix, pair["prefix_a"], pair["prefix_b"], pair["image"], state_layers
        )
    return row


def existing_keys(path: Path) -> set[tuple[str, int, str]]:
    result = set()
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            result.add((row["cell"], int(row["init"]), row["condition"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return result


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def load_pairs(path: Path) -> list[tuple[int, int]]:
    raw = json.loads(path.read_text())
    return [(int(row[0]), int(row[1])) for row in raw]


def validate_config(cfg: dict[str, Any]) -> None:
    layers = [int(value) for value in cfg["fixed"]["reader_layers"]]
    ks = [int(value) for value in cfg["fixed"]["reader_top_k"]]
    ranking = [tuple(int(value) for value in edge) for edge in cfg["fixed"]["reader_ranked_edges"]]
    if ks != sorted(set(ks)) or not ks or max(ks) > len(ranking):
        raise ValueError("reader top-k values must be unique, sorted, and covered by the ranking")
    if len(ranking) != len(set(ranking)):
        raise ValueError("reader ranking contains duplicate edges")
    valid = {(layer, head) for layer in layers for head in range(8)}
    if not set(ranking).issubset(valid):
        raise ValueError("reader ranking contains an out-of-range layer/head")
    prior_control: list[tuple[int, int]] = []
    for k in ks:
        control = [
            tuple(int(value) for value in edge)
            for edge in cfg["fixed"]["reader_random_edges"][str(k)]
        ]
        if len(control) != k or len(control) != len(set(control)):
            raise ValueError(f"reader control k={k} has the wrong size or duplicates")
        if not set(control).issubset(valid) or set(control) & set(ranking[:k]):
            raise ValueError(f"reader control k={k} is invalid or overlaps selected edges")
        if control[: len(prior_control)] != prior_control:
            raise ValueError("reader controls must be nested prefixes")
        prior_control = control


def writer_curve_conditions(layers: list[int]) -> list[str]:
    return (
        ["clean_A", "clean_B", "identity_B", "writer_block_all_image"]
        + [f"writer_block_prefix_through_{layer}" for layer in layers]
        + [f"writer_block_suffix_from_{layer}" for layer in layers]
    )


def run_writer_panel(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    *,
    panel_name: str,
    extra_selection: dict[str, Any] | None = None,
) -> Path:
    panel = cfg[panel_name]
    suite_name = panel["suite"]
    pairs = load_pairs(Path(panel["pairs_file"]))
    layers = list(cfg["fixed"]["writer_layers"])
    state_layers = list(cfg["fixed"]["reader_layers"])
    rows_path = args.out / f"writer_{panel_name}_rows.jsonl"
    done = existing_keys(rows_path)
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    suite = _get_suite(suite_name)
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    required = writer_curve_conditions(layers)
    if extra_selection is not None:
        required += [
            "writer_block_selected_band", "writer_rescue_selected_band",
            "writer_block_control_band", "writer_rescue_control_band",
        ]
    for cell_index, (task_a, task_b) in enumerate(pairs):
        cell = cell_name(suite_name, task_a, task_b)
        env = make_env(suite, suite_name, task_a)
        print(f"[writer-{panel_name}] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in panel["init_ids"]:
                if all((cell, init_id, condition) in done for condition in required):
                    continue
                pair = capture_pair(
                    harness, suite_name, env, prompts, task_a, task_b, init_id, residuals=True
                )
                noise = harness.make_noise(seed=100 * init_id)
                clean_a = action_chunk(harness, pair["prefix_a"], noise)
                clean_b = action_chunk(harness, pair["prefix_b"], noise)

                def record(condition: str, prefix: dict[str, Any], chunk: np.ndarray) -> None:
                    if (cell, init_id, condition) in done:
                        return
                    row = row_payload(
                        stage=f"writer-{panel_name}", suite=suite_name,
                        task_a=task_a, task_b=task_b, init_id=init_id,
                        condition=condition, chunk=chunk,
                        clean_a_chunk=clean_a, clean_b_chunk=clean_b, pair=pair,
                        action_steps=cfg["fixed"]["action_steps_for_metric"],
                        prefix=prefix, state_layers=state_layers,
                    )
                    append_row(rows_path, row)
                    done.add((cell, init_id, condition))

                record("clean_A", pair["prefix_a"], clean_a)
                record("clean_B", pair["prefix_b"], clean_b)
                if (cell, init_id, "identity_B") not in done:
                    identity = harness.prefix_forward(
                        pair["batch_b"], capture=True,
                        attn_hook=writer_intervention(
                            pair["cap_b"], pair["instruction"], pair["image"], layers
                        ),
                    )
                    identity_chunk = action_chunk(harness, identity, noise)
                    if not prefix_equal(pair["prefix_b"], identity) or not np.array_equal(clean_b, identity_chunk):
                        raise RuntimeError("writer identity failed")
                    record("identity_B", identity, identity_chunk)
                    del identity, identity_chunk

                condition_layers = {
                    "writer_block_all_image": layers,
                    **{f"writer_block_prefix_through_{layer}": list(range(0, layer + 1)) for layer in layers},
                    **{f"writer_block_suffix_from_{layer}": list(range(layer, 18)) for layer in layers},
                }
                if extra_selection is not None:
                    selected_band = extra_selection["selected_band"]
                    control_band = extra_selection["control_band"]
                    condition_layers.update({
                        "writer_block_selected_band": selected_band,
                        "writer_block_control_band": control_band,
                    })
                for condition, active in condition_layers.items():
                    if (cell, init_id, condition) in done:
                        continue
                    prefix = harness.prefix_forward(
                        pair["batch_b"], capture=True,
                        attn_hook=writer_intervention(
                            pair["cap_a"], pair["instruction"], pair["image"], active
                        ),
                    )
                    chunk = action_chunk(harness, prefix, noise)
                    record(condition, prefix, chunk)
                    del prefix, chunk
                    gc.collect()

                if extra_selection is not None:
                    for condition, band in (
                        ("writer_rescue_selected_band", extra_selection["selected_band"]),
                        ("writer_rescue_control_band", extra_selection["control_band"]),
                    ):
                        if (cell, init_id, condition) in done:
                            continue
                        prefix = harness.prefix_forward(
                            pair["batch_b"], capture=True,
                            attn_hook=writer_intervention(
                                pair["cap_a"], pair["instruction"], pair["image"], layers,
                                allow_current_layers=band,
                            ),
                        )
                        chunk = action_chunk(harness, prefix, noise)
                        record(condition, prefix, chunk)
                        del prefix, chunk
                        gc.collect()
                del pair
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            env.close()
    return rows_path


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def effect_profile(values: list[float], baseline: float, ceiling: float) -> dict[str, Any]:
    scale = ceiling - baseline
    if scale <= 0:
        return {
            "normalized": [float("nan")] * len(values), "effects": [float("nan")] * len(values),
            "rho": 0.0, "width": 99, "critical_layer": None, "max_step_fraction": float("nan"),
        }
    normalized = [(value - baseline) / scale for value in values]
    previous = 0.0
    effects = []
    for value in normalized:
        effects.append(value - previous)
        previous = value
    positive = [max(0.0, value) for value in effects]
    positive_total = sum(positive)
    cumulative = np.cumsum(positive) / max(positive_total, 1e-12)
    lo = int(np.searchsorted(cumulative, 0.25))
    hi = int(np.searchsorted(cumulative, 0.75))
    return {
        "normalized": normalized,
        "effects": effects,
        "rho": spearman(list(range(len(values))), normalized),
        "width": hi - lo + 1,
        "transition_band": list(range(lo, hi + 1)),
        "critical_layer": int(np.argmax(positive)),
        "max_step_fraction": max(positive) / max(positive_total, 1e-12),
    }


def writer_curve_analysis(
    prefix_values: list[float],
    suffix_raw: list[float],
    baseline: float,
    ceiling: float,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Classify complementary direct interventions without assuming layer additivity."""
    # Invert the suffix response around the clean/full-block endpoints so both
    # curves run from low-layer boundary to high-layer boundary. This is only
    # an orientation transform for locating transitions, not a decomposition
    # of the nonlinear intervention effect.
    suffix_inverted = [
        ceiling - (suffix_raw[index + 1] - baseline)
        if index + 1 < len(suffix_raw)
        else ceiling
        for index in range(len(suffix_raw))
    ]
    prefix_profile = effect_profile(prefix_values, baseline, ceiling)
    suffix_profile = effect_profile(suffix_inverted, baseline, ceiling)
    enough_range = ceiling - baseline >= gate["curve_endpoint_range_min"]
    localized = (
        enough_range
        and prefix_profile["width"] <= gate["localized_transition_width_max"]
        and suffix_profile["width"] <= gate["localized_transition_width_max"]
        and prefix_profile["max_step_fraction"] >= gate["localized_max_step_fraction_min"]
        and suffix_profile["max_step_fraction"] >= gate["localized_max_step_fraction_min"]
        and abs(prefix_profile["critical_layer"] - suffix_profile["critical_layer"])
        <= gate["localized_critical_layer_tolerance"]
    )
    distributed = (
        enough_range
        and prefix_profile["width"] >= gate["distributed_transition_width_min"]
        and suffix_profile["width"] >= gate["distributed_transition_width_min"]
        and prefix_profile["max_step_fraction"] <= gate["distributed_max_step_fraction_max"]
        and suffix_profile["max_step_fraction"] <= gate["distributed_max_step_fraction_max"]
        and prefix_profile["rho"] >= gate["distributed_monotonic_rho_min"]
        and suffix_profile["rho"] >= gate["distributed_monotonic_rho_min"]
    )
    return {
        "classification": "localized" if localized else "distributed" if distributed else "ambiguous",
        "endpoint_range": ceiling - baseline,
        "suffix_inverted_curve": suffix_inverted,
        "prefix_profile": prefix_profile,
        "suffix_profile": suffix_profile,
    }


def analyze_writer_development(rows_path: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    layers = cfg["fixed"]["writer_layers"]
    med = {
        condition: median(row["preference_A"] for row in rows if row["condition"] == condition)
        for condition in sorted({row["condition"] for row in rows})
    }
    baseline = med["clean_B"]
    ceiling = med["writer_block_all_image"]
    prefix_values = [med[f"writer_block_prefix_through_{layer}"] for layer in layers]
    suffix_raw = [med[f"writer_block_suffix_from_{layer}"] for layer in layers]
    gate = cfg["development_gates"]["writer"]
    analysis = writer_curve_analysis(prefix_values, suffix_raw, baseline, ceiling, gate)
    prefix_profile = analysis["prefix_profile"]
    suffix_profile = analysis["suffix_profile"]
    classification = analysis["classification"]
    if classification == "localized":
        lo = min(prefix_profile["critical_layer"], suffix_profile["critical_layer"])
        hi = max(prefix_profile["critical_layer"], suffix_profile["critical_layer"])
    else:
        lo = hi = -1
    selected_band = list(range(lo, hi + 1)) if lo >= 0 else []
    width = len(selected_band)
    low_control = list(range(width)) if width else []
    high_control = list(range(18 - width, 18)) if width else []
    candidates = [band for band in (low_control, high_control) if not set(band) & set(selected_band)]
    control_band = candidates[0] if candidates else []
    result = {
        "status": "pass" if classification != "ambiguous" else "fail",
        "classification": classification,
        "compact_candidate": classification == "localized",
        "selected_band": selected_band,
        "control_band": control_band,
        "clean_B_preference_A": baseline,
        "full_block_preference_A": ceiling,
        "endpoint_range": analysis["endpoint_range"],
        "prefix_curve": prefix_values,
        "suffix_curve": suffix_raw,
        "suffix_inverted_curve": analysis["suffix_inverted_curve"],
        "prefix_profile": prefix_profile,
        "suffix_profile": suffix_profile,
        "gate": gate,
        "rows": len(rows),
        "rows_sha256": sha256(rows_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(args.out / "writer_development.json", result)
    return result


def run_writer_development(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    rows_path = run_writer_panel(args, cfg, panel_name="development")
    print(json.dumps(analyze_writer_development(rows_path, args, cfg), indent=2), flush=True)


def run_reader_panel(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    *,
    panel_name: str,
    selected_k: int | None = None,
) -> Path:
    panel = cfg[panel_name]
    suite_name = panel["suite"]
    pairs = load_pairs(Path(panel["pairs_file"]))
    layers = cfg["fixed"]["reader_layers"]
    ks = cfg["fixed"]["reader_top_k"] if selected_k is None else [selected_k]
    rows_path = args.out / f"reader_{panel_name}_rows.jsonl"
    done = existing_keys(rows_path)
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    suite = _get_suite(suite_name)
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    required = ["clean_A", "clean_B", "reader_block"] + [
        name
        for k in ks
        for name in (
            f"reader_rescue_top_{k}",
            f"reader_rescue_random_{k}",
            f"reader_remove_top_{k}",
            f"reader_remove_random_{k}",
        )
    ]
    ranking = cfg["fixed"]["reader_ranked_edges"]
    random_edges = cfg["fixed"]["reader_random_edges"]
    for cell_index, (task_a, task_b) in enumerate(pairs):
        cell = cell_name(suite_name, task_a, task_b)
        env = make_env(suite, suite_name, task_a)
        print(f"[reader-{panel_name}] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in panel["init_ids"]:
                if all((cell, init_id, condition) in done for condition in required):
                    continue
                pair = capture_pair(
                    harness, suite_name, env, prompts, task_a, task_b, init_id, residuals=False
                )
                noise = harness.make_noise(seed=100 * init_id)
                clean_a = action_chunk(harness, pair["prefix_a"], noise)
                clean_b = action_chunk(harness, pair["prefix_b"], noise)
                donor_keys, donor_values = cache_dict(pair["prefix_a"], layers)

                def record(condition: str, chunk: np.ndarray) -> None:
                    if (cell, init_id, condition) in done:
                        return
                    row = row_payload(
                        stage=f"reader-{panel_name}", suite=suite_name,
                        task_a=task_a, task_b=task_b, init_id=init_id,
                        condition=condition, chunk=chunk,
                        clean_a_chunk=clean_a, clean_b_chunk=clean_b, pair=pair,
                        action_steps=cfg["fixed"]["action_steps_for_metric"],
                    )
                    append_row(rows_path, row)
                    done.add((cell, init_id, condition))

                record("clean_A", clean_a)
                record("clean_B", clean_b)
                for condition, active, rescue in [
                    ("reader_block", None, []),
                    *[
                        item
                        for k in ks
                        for item in (
                            (f"reader_rescue_top_{k}", None, ranking[:k]),
                            (f"reader_rescue_random_{k}", None, random_edges[str(k)]),
                            (f"reader_remove_top_{k}", ranking[:k], []),
                            (f"reader_remove_random_{k}", random_edges[str(k)], []),
                        )
                    ],
                ]:
                    if (cell, init_id, condition) in done:
                        continue
                    hook = reader_intervention(
                        donor_keys, donor_values, pair["image"], harness.cfg.chunk_size,
                        layers, active_edges=active, rescue_edges=rescue,
                    )
                    record(condition, action_chunk(harness, pair["prefix_b"], noise, attn_hook=hook))
                del pair, donor_keys, donor_values
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            env.close()
    return rows_path


def cell_medians(rows: list[dict[str, Any]], condition: str, key: str) -> dict[str, float]:
    return {
        cell: median(row[key] for row in rows if row["condition"] == condition and row["cell"] == cell)
        for cell in sorted({row["cell"] for row in rows})
    }


def analyze_reader_development(rows_path: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    gate = cfg["development_gates"]["reader"]
    candidates = []
    for k in cfg["fixed"]["reader_top_k"]:
        rescue_name = f"reader_rescue_top_{k}"
        rescue_random_name = f"reader_rescue_random_{k}"
        removal_name = f"reader_remove_top_{k}"
        removal_random_name = f"reader_remove_random_{k}"
        rescue = [row for row in rows if row["condition"] == rescue_name]
        removal = [row for row in rows if row["condition"] == removal_name]
        rescue_cells = cell_medians(rows, rescue_name, "D_B")
        rescue_random_cells = cell_medians(rows, rescue_random_name, "D_B")
        removal_cells = cell_medians(rows, removal_name, "D_A")
        removal_random_cells = cell_medians(rows, removal_random_name, "D_A")
        endpoint_cells = sum(
            median(
                row["preference_A"]
                for row in rows
                if row["condition"] == removal_name and row["cell"] == cell
            ) > 0
            and median(
                row["preference_A"]
                for row in rows
                if row["condition"] == rescue_name and row["cell"] == cell
            ) < 0
            for cell in rescue_cells
        )
        control_wins = sum(
            rescue_cells[cell] < rescue_random_cells[cell]
            and removal_cells[cell] < removal_random_cells[cell]
            for cell in rescue_cells
        )
        metrics = {
            "k": k,
            "rescue_median_D_A": median(row["D_A"] for row in rescue),
            "rescue_median_D_B": median(row["D_B"] for row in rescue),
            "rescue_B_like_state_fraction": float(
                np.mean([row["D_B"] < row["D_A"] for row in rescue])
            ),
            "removal_median_D_A": median(row["D_A"] for row in removal),
            "removal_median_D_B": median(row["D_B"] for row in removal),
            "removal_A_like_state_fraction": float(
                np.mean([row["D_A"] < row["D_B"] for row in removal])
            ),
            "joint_endpoint_cell_directions": endpoint_cells,
            "selected_beats_random_joint_cell_directions": control_wins,
        }
        metrics["passes_compact_gate"] = (
            k <= gate["maximum_passing_k"]
            and metrics["rescue_median_D_B"] <= gate["rescue_median_D_B_max"]
            and metrics["rescue_B_like_state_fraction"] >= gate["rescue_B_like_state_fraction_min"]
            and metrics["removal_median_D_A"] <= gate["removal_median_D_A_max"]
            and metrics["removal_A_like_state_fraction"] >= gate["removal_A_like_state_fraction_min"]
            and endpoint_cells >= gate["joint_endpoint_cell_directions_min"]
            and control_wins >= gate["selected_beats_random_joint_cell_directions_min"]
        )
        candidates.append(metrics)
    passing = [row for row in candidates if row["passes_compact_gate"]]
    result = {
        "status": "pass" if passing else "fail",
        "selected_k": min(row["k"] for row in passing) if passing else None,
        "candidates": candidates,
        "gate": gate,
        "rows": len(rows),
        "rows_sha256": sha256(rows_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(args.out / "reader_development.json", result)
    return result


def run_reader_development(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    rows_path = run_reader_panel(args, cfg, panel_name="development")
    print(json.dumps(analyze_reader_development(rows_path, args, cfg), indent=2), flush=True)


def analyze_writer_holdout(rows_path: Path, args: argparse.Namespace, cfg: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    layers = cfg["fixed"]["writer_layers"]
    med = {
        condition: median(row["preference_A"] for row in rows if row["condition"] == condition)
        for condition in sorted({row["condition"] for row in rows})
    }
    prefix_curve = [med[f"writer_block_prefix_through_{layer}"] for layer in layers]
    suffix_curve = [med[f"writer_block_suffix_from_{layer}"] for layer in layers]
    prefix_corr = spearman(selection["prefix_curve"], prefix_curve)
    suffix_corr = spearman(selection["suffix_curve"], suffix_curve)
    curve_analysis = writer_curve_analysis(
        prefix_curve,
        suffix_curve,
        med["clean_B"],
        med["writer_block_all_image"],
        cfg["development_gates"]["writer"],
    )
    gate = cfg["holdout_gates"]["writer"]
    metrics: dict[str, Any] = {
        "prefix_curve_correlation": prefix_corr,
        "suffix_curve_correlation": suffix_corr,
        "holdout_classification": curve_analysis["classification"],
    }
    curve_passed = (
        prefix_corr >= gate["development_curve_correlation_min"]
        and suffix_corr >= gate["development_curve_correlation_min"]
        and curve_analysis["classification"] == selection["classification"]
    )
    compact_passed = False
    if selection["classification"] == "localized":
        block_cells = cell_medians(rows, "writer_block_selected_band", "preference_A")
        rescue_cells = cell_medians(rows, "writer_rescue_selected_band", "preference_A")
        block_control = cell_medians(rows, "writer_block_control_band", "D_A")
        rescue_control = cell_medians(rows, "writer_rescue_control_band", "D_B")
        block_distance = cell_medians(rows, "writer_block_selected_band", "D_A")
        rescue_distance = cell_medians(rows, "writer_rescue_selected_band", "D_B")
        replicated = sum(block_cells[cell] > 0 and rescue_cells[cell] < 0 for cell in block_cells)
        beats = sum(
            block_distance[cell] < block_control[cell]
            and rescue_distance[cell] < rescue_control[cell]
            for cell in block_distance
        )
        metrics.update({
            "selected_block_median_preference_for_A": med["writer_block_selected_band"],
            "selected_rescue_median_preference_for_B": -med["writer_rescue_selected_band"],
            "replicated_cell_directions": replicated,
            "selected_beats_matched_control_cell_directions": beats,
        })
        compact_passed = (
            curve_passed
            and metrics["selected_block_median_preference_for_A"]
            >= gate["selected_block_median_preference_for_A_min"]
            and metrics["selected_rescue_median_preference_for_B"]
            >= gate["selected_rescue_median_preference_for_B_min"]
            and replicated >= gate["replicated_cell_directions_min"]
            and beats >= gate["selected_beats_matched_control_cell_directions_min"]
        )
    passed = compact_passed if selection["classification"] == "localized" else curve_passed
    result = {
        "status": "pass" if passed else "fail",
        "result_type": (
            "compact_writer" if compact_passed
            else "replicated_distributed_writer" if passed else "failed_replication"
        ),
        "compact_mechanism_pass": compact_passed,
        "development_classification": selection["classification"],
        "selected_band": selection["selected_band"],
        "control_band": selection["control_band"],
        "metrics": metrics,
        "prefix_curve": prefix_curve,
        "suffix_curve": suffix_curve,
        "gate": gate,
        "rows": len(rows),
        "rows_sha256": sha256(rows_path),
    }
    write_json(args.out / "writer_holdout.json", result)
    return result


def run_writer_holdout(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    selection = json.loads((args.out / "writer_development.json").read_text())
    if selection["status"] != "pass":
        raise RuntimeError("writer development was ambiguous; writer holdout remains unopened")
    extra = selection if selection["classification"] == "localized" else None
    rows_path = run_writer_panel(args, cfg, panel_name="holdout", extra_selection=extra)
    print(json.dumps(analyze_writer_holdout(rows_path, args, cfg, selection), indent=2), flush=True)


def analyze_reader_holdout(rows_path: Path, args: argparse.Namespace, cfg: dict[str, Any], selected_k: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rescue_name = f"reader_rescue_top_{selected_k}"
    rescue_random_name = f"reader_rescue_random_{selected_k}"
    removal_name = f"reader_remove_top_{selected_k}"
    removal_random_name = f"reader_remove_random_{selected_k}"
    rescue = [row for row in rows if row["condition"] == rescue_name]
    removal = [row for row in rows if row["condition"] == removal_name]
    rescue_cells = cell_medians(rows, rescue_name, "D_B")
    rescue_random_cells = cell_medians(rows, rescue_random_name, "D_B")
    removal_cells = cell_medians(rows, removal_name, "D_A")
    removal_random_cells = cell_medians(rows, removal_random_name, "D_A")
    joint_endpoint_cells = sum(
        median(
            row["preference_A"]
            for row in removal
            if row["cell"] == cell
        ) > 0
        and median(
            row["preference_A"]
            for row in rescue
            if row["cell"] == cell
        ) < 0
        for cell in rescue_cells
    )
    control_wins = sum(
        rescue_cells[cell] < rescue_random_cells[cell]
        and removal_cells[cell] < removal_random_cells[cell]
        for cell in rescue_cells
    )
    gate = cfg["holdout_gates"]["reader"]
    metrics = {
        "selected_k": selected_k,
        "selected_rescue_median_D_A": median(row["D_A"] for row in rescue),
        "selected_rescue_median_D_B": median(row["D_B"] for row in rescue),
        "selected_removal_median_D_A": median(row["D_A"] for row in removal),
        "selected_removal_median_D_B": median(row["D_B"] for row in removal),
        "selected_joint_endpoint_cell_directions": joint_endpoint_cells,
        "selected_beats_random_joint_cell_directions": control_wins,
    }
    passed = (
        metrics["selected_rescue_median_D_B"] <= gate["selected_rescue_median_D_B_max"]
        and metrics["selected_removal_median_D_A"] <= gate["selected_removal_median_D_A_max"]
        and joint_endpoint_cells >= gate["selected_joint_endpoint_cell_directions_min"]
        and control_wins >= gate["selected_beats_random_joint_cell_directions_min"]
    )
    result = {
        "status": "pass" if passed else "fail",
        "metrics": metrics,
        "gate": gate,
        "rows": len(rows),
        "rows_sha256": sha256(rows_path),
    }
    write_json(args.out / "reader_holdout.json", result)
    return result


def run_reader_holdout(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    selection = json.loads((args.out / "reader_development.json").read_text())
    if selection["status"] != "pass":
        raise RuntimeError("no compact reader passed development; reader holdout remains unopened")
    selected_k = int(selection["selected_k"])
    rows_path = run_reader_panel(args, cfg, panel_name="holdout", selected_k=selected_k)
    print(json.dumps(analyze_reader_holdout(rows_path, args, cfg, selected_k), indent=2), flush=True)


def run_finalize(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    payload: dict[str, Any] = {
        "writer_development": json.loads((args.out / "writer_development.json").read_text()),
        "reader_development": json.loads((args.out / "reader_development.json").read_text()),
    }
    for name in ("writer_holdout", "reader_holdout"):
        path = args.out / f"{name}.json"
        payload[name] = json.loads(path.read_text()) if path.exists() else {"status": "not_opened"}
    payload["accept_level_joint_result"] = (
        payload["writer_holdout"].get("compact_mechanism_pass") is True
        and payload["reader_holdout"].get("status") == "pass"
    )
    payload["note"] = (
        "This Boolean reports the preregistered joint mechanism gate, not a guaranteed MATS admissions decision."
    )
    write_json(args.out / "final_summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def ensure_manifest(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    manifest_path = args.out / "manifest.json"
    hashes = {
        "runner": sha256(Path(__file__)),
        "attention_pathway": sha256(Path(__file__).with_name("attention_pathway.py")),
        "hooks": sha256(Path(__file__).with_name("hooks.py")),
        "analysis_tests": sha256(Path("tests/test_pi05_attention_resolution.py")),
        "config": sha256(args.config),
        "preregistration": sha256(args.preregistration),
        "development_pairs": sha256(Path(cfg["development"]["pairs_file"])),
        "holdout_pairs": sha256(Path(cfg["holdout"]["pairs_file"])),
        "reader_ranking_source": sha256(Path(cfg["fixed"]["reader_ranking_source"])),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["hashes"] != hashes:
            raise RuntimeError("sealed code/config hashes changed; use a new output directory")
        return
    write_json(manifest_path, {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg,
        "hashes": hashes,
        "args_at_creation": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=("writer-development", "reader-development", "writer-holdout", "reader-holdout", "finalize"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pi05_attention_resolution.json"))
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("docs/PREREG-pi05-attention-pathway-resolution-2026-09-04.md"),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    require_free_space(args.out, args.minimum_free_gib)
    ensure_manifest(args, cfg)
    stages = {
        "writer-development": run_writer_development,
        "reader-development": run_reader_development,
        "writer-holdout": run_writer_holdout,
        "reader-holdout": run_reader_holdout,
        "finalize": run_finalize,
    }
    stages[args.stage](args, cfg)


if __name__ == "__main__":
    main()
