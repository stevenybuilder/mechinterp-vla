#!/usr/bin/env python3

"""Validate the local Push-T tensors, videos, and JEPA-WM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import pickle
import shutil
import subprocess
from pathlib import Path

import torch


CHECKPOINT_SHA256 = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
REQUIRED_CHECKPOINT_KEYS = {
    "epoch",
    "opt",
    "predictor",
    "proprio_encoder",
    "scaler",
    "stats",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def validate_split(split_dir: Path) -> None:
    with (split_dir / "seq_lengths.pkl").open("rb") as source:
        lengths = pickle.load(source)

    tensors = {
        name: torch.load(
            split_dir / f"{name}.pth",
            map_location="cpu",
            weights_only=True,
        )
        for name in ("states", "rel_actions", "abs_actions", "velocities")
    }
    rollout_count = len(lengths)
    for name, tensor in tensors.items():
        assert tensor.shape[0] == rollout_count, (
            f"{split_dir.name}/{name}: {tensor.shape[0]} rollouts, expected {rollout_count}"
        )

    assert tensors["states"].shape[-1] == 5
    assert tensors["rel_actions"].shape[-1] == 2
    assert tensors["abs_actions"].shape[-1] == 2
    assert tensors["velocities"].shape[-1] == 2

    videos = sorted((split_dir / "obses").glob("episode_*.mp4"))
    expected_names = {f"episode_{index:03d}.mp4" for index in range(rollout_count)}
    assert {video.name for video in videos} == expected_names

    sample_indices = sorted({0, rollout_count // 2, rollout_count - 1})
    for index in sample_indices:
        frames = video_frames(split_dir / "obses" / f"episode_{index:03d}.mp4")
        assert frames == lengths[index], (
            f"{split_dir.name}/episode_{index:03d}.mp4: {frames} frames, "
            f"expected {lengths[index]}"
        )

    print(
        f"{split_dir.name}: {rollout_count:,} rollouts, "
        f"{sum(lengths):,} valid timesteps, {len(videos):,} videos"
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "artifacts/data/pusht_noise",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "artifacts/checkpoints/jepa_wm_pusht.pth.tar",
    )
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required; install FFmpeg and retry")

    actual_sha = sha256(args.checkpoint)
    assert actual_sha == CHECKPOINT_SHA256, (
        f"Checkpoint checksum {actual_sha} does not match {CHECKPOINT_SHA256}"
    )

    # This file is loaded with pickle support only after verifying the exact
    # SHA-256 of Meta's published checkpoint above.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert set(checkpoint) == REQUIRED_CHECKPOINT_KEYS
    predictor = checkpoint["predictor"]
    predictor_blocks = {
        key.split("predictor_blocks.", 1)[1].split(".", 1)[0]
        for key in predictor
        if "predictor_blocks." in key
    }
    assert predictor_blocks == {str(index) for index in range(6)}
    print(
        f"checkpoint: SHA-256 verified, epoch={checkpoint['epoch']}, "
        f"predictor_blocks={len(predictor_blocks)}"
    )

    validate_split(args.dataset / "train")
    validate_split(args.dataset / "val")


if __name__ == "__main__":
    main()
