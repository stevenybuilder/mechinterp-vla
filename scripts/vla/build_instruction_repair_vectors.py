#!/usr/bin/env python
"""Estimate frozen cross-scene image-KV directions for instruction repair."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from hooks import Pi05Harness, cache_kv_lists
from lerobot.envs.libero import _get_suite
from run_libero_prompt_conditions import make_env


TASKS = {1: 5, 2: 4}  # receiver task -> task supplying the known conflicting instruction
LAYERS = tuple(range(12, 18))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-inits", default="0-9")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", default="lerobot/pi05_libero_finetuned_v044")
    parser.add_argument("--revision", default="8e174154ef5f6c60a8da12ae99c303d8963138c1")
    parser.add_argument("--processor-path", default=None)
    args = parser.parse_args()
    start, end = (int(value) for value in args.calibration_inits.split("-", 1))
    init_ids = list(range(start, end + 1))
    if init_ids != list(range(10)):
        raise AssertionError(f"calibration split drift: {init_ids}")

    harness = Pi05Harness(
        args.policy,
        revision=args.revision,
        dtype="float32",
        processor_path=args.processor_path,
    )
    suite = _get_suite("libero_object")
    vectors: dict[int, dict] = {}
    manifest = {
        "policy": args.policy,
        "revision": args.revision,
        "dtype": "float32",
        "suite": "libero_object",
        "layers": list(LAYERS),
        "calibration_inits": init_ids,
        "tasks": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    for task_id, conflict_task_id in TASKS.items():
        env = make_env(suite, "libero_object", task_id)
        sums = None
        n_img_valid_seen = None
        try:
            for init_id in init_ids:
                seed = 20260831 + 100_000 * task_id + init_id
                torch.manual_seed(seed)
                env.init_state_id = init_id
                observation, _ = env.reset(seed=seed)
                correct_batch = harness.build_batch(observation, suite.tasks[task_id].language)
                conflict_batch = harness.build_batch(observation, suite.tasks[conflict_task_id].language)
                correct = harness.prefix_forward(correct_batch)
                conflict = harness.prefix_forward(conflict_batch)
                if correct["n_img_valid"] != conflict["n_img_valid"]:
                    raise AssertionError("valid-image count differs across prompts")
                n_img_valid = int(correct["n_img_valid"])
                if n_img_valid_seen is None:
                    n_img_valid_seen = n_img_valid
                elif n_img_valid != n_img_valid_seen:
                    raise AssertionError(f"valid-image count drift: {n_img_valid} != {n_img_valid_seen}")
                kc, vc = cache_kv_lists(correct["cache"])
                kx, vx = cache_kv_lists(conflict["cache"])
                current = {}
                for layer in LAYERS:
                    current[layer] = {
                        "k": (kc[layer][:, :, :n_img_valid] - kx[layer][:, :, :n_img_valid]).float().cpu(),
                        "v": (vc[layer][:, :, :n_img_valid] - vx[layer][:, :, :n_img_valid]).float().cpu(),
                    }
                if sums is None:
                    sums = {
                        layer: {name: tensor.clone() for name, tensor in pair.items()}
                        for layer, pair in current.items()
                    }
                else:
                    for layer in LAYERS:
                        for name in ("k", "v"):
                            sums[layer][name].add_(current[layer][name])
                print(f"[vector] task={task_id} init={init_id} n_img_valid={n_img_valid}", flush=True)
        finally:
            env.close()
        if sums is None or n_img_valid_seen is None:
            raise AssertionError(f"no calibration rows for task {task_id}")
        direction = {
            layer: {name: tensor.div(len(init_ids)) for name, tensor in pair.items()}
            for layer, pair in sums.items()
        }
        vectors[task_id] = {
            "conflict_task_id": conflict_task_id,
            "n_img_valid": n_img_valid_seen,
            "direction": direction,
        }
        manifest["tasks"][str(task_id)] = {
            "conflict_task_id": conflict_task_id,
            "correct_prompt": suite.tasks[task_id].language,
            "conflict_prompt": suite.tasks[conflict_task_id].language,
            "n_img_valid": n_img_valid_seen,
            "direction_norms": {
                str(layer): {
                    name: float(direction[layer][name].norm().item()) for name in ("k", "v")
                }
                for layer in LAYERS
            },
        }

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "manifest": manifest,
            "vectors": vectors,
        },
        args.out / "vectors.pt",
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[vector] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

