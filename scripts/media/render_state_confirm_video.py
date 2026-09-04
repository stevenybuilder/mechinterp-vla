#!/usr/bin/env python3
"""Rerun one registered state-confirmation seed while capturing real simulator frames.

This script is intended for the original pi0.5 GPU/LIBERO environment. It reuses
`run_instruction_repair.run_episode` instead of duplicating the intervention. The
only monkeypatch wraps observation formatting to retain the pixels already seen by
the policy. Regenerated outcomes are checked against the saved confirmation JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch


DEFAULT_POLICY = "lerobot/pi05_libero_finetuned_v044"
DEFAULT_REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def grab_frame(observation: object) -> np.ndarray:
    pixels = observation.get("pixels", observation) if isinstance(observation, dict) else observation
    if isinstance(pixels, dict):
        key = next(
            (candidate for candidate in pixels if "agent" in candidate.lower() or "front" in candidate.lower()),
            sorted(pixels)[0],
        )
        pixels = pixels[key]
    frame = np.asarray(pixels)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise AssertionError(f"unexpected observation frame shape: {frame.shape}")
    return frame[..., :3].copy()


def matching_record(records: list[dict], task_id: int, init_id: int, condition: str) -> dict:
    matches = [
        row
        for row in records
        if int(row["task_id"]) == task_id
        and int(row["init_id"]) == init_id
        and row["condition"] == condition
        and row.get("phase") == "state_confirm"
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one saved record for task={task_id}, init={init_id}, condition={condition}; "
            f"found {len(matches)}"
        )
    return matches[0]


def assert_reproduction(rendered: dict, saved: dict) -> None:
    fields = ("success", "done", "n_steps", "first_touch_set", "correct_target_first_touched",
              "conflict_target_first_touched", "touched_any", "grasped_any", "seed")
    mismatches = {
        field: {"saved": saved.get(field), "rendered": rendered.get(field)}
        for field in fields
        if saved.get(field) != rendered.get(field)
    }
    if mismatches:
        raise AssertionError(f"captured rerun did not reproduce the saved episode: {json.dumps(mismatches)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/root/vla"))
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--raw-out", type=Path, default=Path("/dev/shm/mats_state_confirm_video"))
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--init-id", type=int, default=20)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--processor-path", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "scripts" / "vla"))
    import run_instruction_repair as repair
    from hooks import Pi05Harness
    from lerobot.envs.libero import _get_suite

    args.raw_out.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(args.records)
    suite = _get_suite("libero_object")
    harness = Pi05Harness(
        args.policy,
        revision=args.revision,
        dtype="float32",
        processor_path=args.processor_path,
    )

    frames: list[np.ndarray] = []
    original_make_env = repair.make_env

    def make_env_with_capture(*make_args, **make_kwargs):
        env = original_make_env(*make_args, **make_kwargs)
        original_format = env._format_raw_obs

        def format_and_capture(raw_observation):
            observation = original_format(raw_observation)
            frames.append(grab_frame(observation))
            return observation

        env._format_raw_obs = format_and_capture
        return env

    repair.make_env = make_env_with_capture
    specs = [
        ("conflict", None),
        ("correct", None),
        ("state_live", repair.LAYERS),
        ("state_early", tuple(range(0, 6))),
    ]
    metadata = {
        "schema_version": 1,
        "policy": args.policy,
        "revision": args.revision,
        "task_id": args.task_id,
        "init_id": args.init_id,
        "source_records": str(args.records),
        "capture_method": "wrapped env._format_raw_obs; policy and rollout code unchanged",
        "display_transform": "vertical flip only during composition",
        "panels": [],
    }
    try:
        for condition, donor_layers in specs:
            frames.clear()
            torch.cuda.empty_cache()
            rendered = repair.run_episode(
                harness,
                suite,
                args.task_id,
                args.init_id,
                condition,
                0.0,
                None,
                donor_layers=donor_layers,
            )
            rendered["phase"] = "state_confirm"
            saved = matching_record(records, args.task_id, args.init_id, condition)
            assert_reproduction(rendered, saved)
            if not frames:
                raise AssertionError(f"no frames captured for {condition}")
            frame_array = np.stack(frames).astype(np.uint8)
            np.save(args.raw_out / f"{condition}.npy", frame_array)
            metadata["panels"].append(
                {
                    "condition": condition,
                    "donor_layers": list(donor_layers) if donor_layers is not None else None,
                    "n_frames": len(frame_array),
                    "frame_shape": list(frame_array.shape[1:]),
                    "reproduction_check": "MATCHES",
                    "saved_record": {
                        key: saved[key]
                        for key in (
                            "task_id", "init_id", "condition", "prompt", "correct_object", "conflict_object",
                            "success", "n_steps", "first_touch_set", "correct_target_first_touched",
                            "conflict_target_first_touched", "seed",
                        )
                    },
                }
            )
            (args.raw_out / "render_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(
                f"[video] condition={condition} frames={len(frame_array)} "
                f"success={int(rendered['success'])} first_touch={rendered['first_touch_set']} MATCHES",
                flush=True,
            )
    finally:
        repair.make_env = original_make_env


if __name__ == "__main__":
    main()
