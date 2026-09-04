#!/usr/bin/env python
"""Confirm the fixed pi0.5 prefix writer band at layers 6--8."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch

from hooks import Pi05Harness
from lerobot.envs.libero import _get_suite
from pi05_attention_resolution import (
    action_chunk,
    append_row,
    capture_pair,
    cell_medians,
    cell_name,
    existing_keys,
    load_pairs,
    make_env,
    median,
    prefix_equal,
    require_free_space,
    row_payload,
    sha256,
    write_json,
    writer_intervention,
)


CONDITIONS = (
    "clean_A",
    "clean_B",
    "identity_B",
    "writer_block_selected_band",
    "writer_rescue_selected_band",
    "writer_block_control_band",
    "writer_rescue_control_band",
)


def ensure_manifest(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    path = args.out / "manifest.json"
    root = Path(__file__).resolve().parents[2]
    hashes = {
        "runner": sha256(Path(__file__)),
        "config": sha256(args.config),
        "preregistration": sha256(args.preregistration),
        "implementation_addendum": sha256(
            root / "docs/ADDENDUM-pi05-writer-band-offline-init-2026-09-04.md"
        ),
        "pair_panel_addendum": sha256(
            root / "docs/ADDENDUM-pi05-writer-band-pair-panel-correction-2026-09-04.md"
        ),
        "pair_metadata_audit": sha256(
            root / "artifacts/pi05_object_pair_metadata_2026-09-04.json"
        ),
        "pair_metadata_runner": sha256(
            root / "scripts/vla/pi05_object_pair_metadata.py"
        ),
        "pairs": sha256(Path(cfg["holdout"]["pairs_file"])),
        "resolution_runner_dependency": sha256(root / "scripts/vla/pi05_attention_resolution.py"),
        "attention_primitive": sha256(root / "scripts/vla/attention_pathway.py"),
        "hooks": sha256(root / "scripts/vla/hooks.py"),
    }
    if path.exists():
        if json.loads(path.read_text())["hashes"] != hashes:
            raise RuntimeError("sealed band-confirmation hashes changed; use a new output directory")
        return
    write_json(path, {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg,
        "hashes": hashes,
    })


def run(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    panel = cfg["holdout"]
    suite_name = panel["suite"]
    pairs = load_pairs(Path(panel["pairs_file"]))
    all_layers = list(cfg["fixed"]["all_writer_layers"])
    selected = list(cfg["fixed"]["selected_band"])
    control = list(cfg["fixed"]["control_band"])
    state_layers = list(cfg["fixed"]["state_layers"])
    rows_path = args.out / "rows.jsonl"
    done = existing_keys(rows_path)
    harness = Pi05Harness(cfg["policy"], revision=cfg["revision"], dtype=args.dtype)
    suite = _get_suite(suite_name)
    prompts = [suite.tasks[index].language for index in range(len(suite.tasks))]
    for cell_index, (task_a, task_b) in enumerate(pairs):
        cell = cell_name(suite_name, task_a, task_b)
        env = make_env(suite, suite_name, task_a)
        print(f"[band-confirm] {cell} ({cell_index + 1}/{len(pairs)})", flush=True)
        try:
            for init_id in panel["init_ids"]:
                if all((cell, init_id, condition) in done for condition in CONDITIONS):
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
                    append_row(rows_path, row_payload(
                        stage="writer-band-holdout", suite=suite_name,
                        task_a=task_a, task_b=task_b, init_id=init_id,
                        condition=condition, chunk=chunk,
                        clean_a_chunk=clean_a, clean_b_chunk=clean_b,
                        pair=pair, action_steps=cfg["fixed"]["action_steps_for_metric"],
                        prefix=prefix, state_layers=state_layers,
                    ))
                    done.add((cell, init_id, condition))

                record("clean_A", pair["prefix_a"], clean_a)
                record("clean_B", pair["prefix_b"], clean_b)
                identity = harness.prefix_forward(
                    pair["batch_b"], capture=True,
                    attn_hook=writer_intervention(
                        pair["cap_b"], pair["instruction"], pair["image"], all_layers
                    ),
                )
                identity_chunk = action_chunk(harness, identity, noise)
                if not prefix_equal(pair["prefix_b"], identity) or not np.array_equal(clean_b, identity_chunk):
                    raise RuntimeError("writer identity failed")
                record("identity_B", identity, identity_chunk)

                for condition, band, rescue in (
                    ("writer_block_selected_band", selected, False),
                    ("writer_rescue_selected_band", selected, True),
                    ("writer_block_control_band", control, False),
                    ("writer_rescue_control_band", control, True),
                ):
                    hook = writer_intervention(
                        pair["cap_a"], pair["instruction"], pair["image"],
                        all_layers if rescue else band,
                        allow_current_layers=band if rescue else (),
                    )
                    prefix = harness.prefix_forward(pair["batch_b"], capture=True, attn_hook=hook)
                    record(condition, prefix, action_chunk(harness, prefix, noise))
                    del prefix, hook
                del pair, identity, identity_chunk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            env.close()
    return rows_path


def analyze(rows_path: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    block = "writer_block_selected_band"
    rescue = "writer_rescue_selected_band"
    block_control = "writer_block_control_band"
    rescue_control = "writer_rescue_control_band"
    block_pref = cell_medians(rows, block, "preference_A")
    rescue_pref = cell_medians(rows, rescue, "preference_A")
    block_d = cell_medians(rows, block, "D_A")
    rescue_d = cell_medians(rows, rescue, "D_B")
    block_control_d = cell_medians(rows, block_control, "D_A")
    rescue_control_d = cell_medians(rows, rescue_control, "D_B")
    joint = sum(block_pref[cell] > 0 and rescue_pref[cell] < 0 for cell in block_pref)
    wins = sum(
        block_d[cell] < block_control_d[cell] and rescue_d[cell] < rescue_control_d[cell]
        for cell in block_d
    )
    median_block = median(row["preference_A"] for row in rows if row["condition"] == block)
    median_rescue_b = -median(row["preference_A"] for row in rows if row["condition"] == rescue)
    gate = cfg["gates"]
    passed = (
        median_block >= gate["selected_block_median_preference_for_A_min"]
        and median_rescue_b >= gate["selected_rescue_median_preference_for_B_min"]
        and joint >= gate["joint_endpoint_cell_directions_min"]
        and wins >= gate["selected_beats_control_joint_cell_directions_min"]
    )
    result = {
        "status": "pass" if passed else "fail",
        "selected_band": cfg["fixed"]["selected_band"],
        "control_band": cfg["fixed"]["control_band"],
        "metrics": {
            "selected_block_median_preference_for_A": median_block,
            "selected_rescue_median_preference_for_B": median_rescue_b,
            "joint_endpoint_cell_directions": joint,
            "selected_beats_control_joint_cell_directions": wins,
        },
        "gate": gate,
        "rows": len(rows),
        "expected_rows": len(load_pairs(Path(cfg["holdout"]["pairs_file"]))) * len(cfg["holdout"]["init_ids"]) * len(CONDITIONS),
        "rows_sha256": sha256(rows_path),
        "nonlinear_diagnostic_authorized": not passed,
    }
    write_json(args.out / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pi05_writer_band_confirmation.json"))
    parser.add_argument(
        "--preregistration", type=Path,
        default=Path("docs/PREREG-pi05-writer-band-6-8-confirmation-2026-09-04.md"),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["fixed"]["selected_band"] != [6, 7, 8] or cfg["fixed"]["control_band"] != [14, 15, 16]:
        raise RuntimeError("frozen writer bands changed")
    args.out.mkdir(parents=True, exist_ok=True)
    require_free_space(args.out, args.minimum_free_gib)
    ensure_manifest(args, cfg)
    print(json.dumps(analyze(run(args, cfg), args, cfg), indent=2), flush=True)


if __name__ == "__main__":
    main()
