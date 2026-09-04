#!/usr/bin/env python
"""Pure tensor utilities for the frozen donor-free π0.5 repair experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


KV_NAMES = ("k", "v")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_kv_lists(cache: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return cache K/V tensors without importing the GPU/model harness."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    if hasattr(cache, "layers"):
        return [layer.keys for layer in cache.layers], [layer.values for layer in cache.layers]
    if isinstance(cache, (list, tuple)):
        return [item[0] for item in cache], [item[1] for item in cache]
    raise TypeError(f"unknown cache type {type(cache)}")


def signed_task_code(
    target_task_id: int,
    source_task_id: int,
    num_task_codes: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not 0 <= target_task_id < num_task_codes:
        raise ValueError(f"target task {target_task_id} outside [0,{num_task_codes})")
    if not 0 <= source_task_id < num_task_codes:
        raise ValueError(f"source task {source_task_id} outside [0,{num_task_codes})")
    code = torch.zeros(num_task_codes, device=device, dtype=dtype)
    code[target_task_id] += 1
    code[source_task_id] -= 1
    return code


def map_key(layer: int, name: str) -> str:
    if name not in KV_NAMES:
        raise ValueError(name)
    return f"{layer}:{name}"


def image_view(prefix: dict, layer: int, name: str) -> torch.Tensor:
    keys, values = cache_kv_lists(prefix["cache"])
    source = keys[layer] if name == "k" else values[layer]
    return source[:, :, : int(prefix["n_img_valid"]), :]


def predict_tensor_delta(
    host: torch.Tensor,
    record: dict,
    target_task_id: int,
    source_task_id: int,
    num_task_codes: int,
    rank: int,
) -> torch.Tensor:
    """Predict a delta with a frozen standardized low-rank affine map."""
    shape = host.shape
    if host.ndim != 4 or shape[0] != 1:
        raise AssertionError(f"expected [1,heads,image,dim], got {shape}")
    flat = host.float().reshape(-1, shape[-1])
    code = signed_task_code(
        target_task_id,
        source_task_id,
        num_task_codes,
        device=flat.device,
        dtype=flat.dtype,
    ).expand(flat.shape[0], -1)
    features = torch.cat((flat, code), dim=-1)
    expected = int(record["feature_dim"])
    if features.shape[-1] != expected:
        raise AssertionError(f"feature dimension {features.shape[-1]} != fitted {expected}")
    available_rank = int(record["max_rank"])
    if not 1 <= rank <= available_rank:
        raise ValueError(f"rank {rank} outside [1,{available_rank}]")
    mean_feature = record["mean_feature"].to(device=flat.device, dtype=flat.dtype)
    scale_feature = record["scale_feature"].to(device=flat.device, dtype=flat.dtype)
    mean_delta = record["mean_delta"].to(device=flat.device, dtype=flat.dtype)
    u = record["u"].to(device=flat.device, dtype=flat.dtype)[:, :rank]
    singular = record["s"].to(device=flat.device, dtype=flat.dtype)[:rank]
    vh = record["vh"].to(device=flat.device, dtype=flat.dtype)[:rank]
    standardized = (features - mean_feature) / scale_feature
    delta = mean_delta + ((standardized @ u) * singular) @ vh
    return delta.reshape(shape).to(dtype=host.dtype)


def predict_proposal(
    prefix: dict,
    model: dict,
    target_task_id: int,
    source_task_id: int,
    layers: tuple[int, ...] | list[int],
    rank: int,
) -> dict[int, dict[str, torch.Tensor]]:
    proposal: dict[int, dict[str, torch.Tensor]] = {}
    num_task_codes = int(model["manifest"]["num_task_codes"])
    for layer in layers:
        proposal[layer] = {}
        for name in KV_NAMES:
            host = image_view(prefix, layer, name)
            proposal[layer][name] = predict_tensor_delta(
                host,
                model["maps"][map_key(layer, name)],
                target_task_id,
                source_task_id,
                num_task_codes,
                rank,
            )
    return proposal


def _scale_to_norm(source: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    source_norm = source.float().norm()
    if float(target_norm) == 0.0:
        return torch.zeros_like(source)
    if float(source_norm) == 0.0:
        raise AssertionError("cannot norm-match a zero control to a nonzero proposal")
    return source * (target_norm.to(source.device, torch.float32) / source_norm).to(source.dtype)


def rescale_like(
    source: dict[int, dict[str, torch.Tensor]],
    target: dict[int, dict[str, torch.Tensor]],
) -> dict[int, dict[str, torch.Tensor]]:
    if source.keys() != target.keys():
        raise AssertionError("control and repair layer sets differ")
    result: dict[int, dict[str, torch.Tensor]] = {}
    for layer in target:
        result[layer] = {}
        for name in KV_NAMES:
            result[layer][name] = _scale_to_norm(source[layer][name], target[layer][name].float().norm())
    return result


def random_matched(
    target: dict[int, dict[str, torch.Tensor]], seed: int
) -> dict[int, dict[str, torch.Tensor]]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for layer in target:
        result[layer] = {}
        for name in KV_NAMES:
            reference = target[layer][name]
            noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
            noise = noise.to(device=reference.device, dtype=reference.dtype)
            result[layer][name] = _scale_to_norm(noise, reference.float().norm())
    return result


def orthogonal_matched(
    target: dict[int, dict[str, torch.Tensor]], seed: int
) -> dict[int, dict[str, torch.Tensor]]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for layer in target:
        result[layer] = {}
        for name in KV_NAMES:
            reference = target[layer][name]
            noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
            noise = noise.to(device=reference.device, dtype=torch.float32)
            direction = reference.float()
            denominator = direction.flatten().dot(direction.flatten())
            if float(denominator) > 0.0:
                projection = noise.flatten().dot(direction.flatten()) / denominator
                noise = noise - projection * direction
            result[layer][name] = _scale_to_norm(noise, direction.norm()).to(reference.dtype)
    return result


def match_early_to_live(
    early: dict[int, dict[str, torch.Tensor]],
    live: dict[int, dict[str, torch.Tensor]],
    early_layers: tuple[int, ...] | list[int],
    live_layers: tuple[int, ...] | list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    if len(early_layers) != len(live_layers):
        raise AssertionError("early/live layer bands must have equal size")
    early_norm = torch.sqrt(
        sum(early[layer][name].float().square().sum() for layer in early_layers for name in KV_NAMES)
    )
    live_norm = torch.sqrt(
        sum(live[layer][name].float().square().sum() for layer in live_layers for name in KV_NAMES)
    )
    if float(live_norm) > 0.0 and float(early_norm) == 0.0:
        raise AssertionError("cannot norm-match an all-zero early band to a nonzero live proposal")
    scale = live_norm / early_norm.clamp_min(1e-12)
    result: dict[int, dict[str, torch.Tensor]] = {}
    for early_layer in early_layers:
        result[early_layer] = {}
        for name in KV_NAMES:
            result[early_layer][name] = early[early_layer][name] * scale.to(
                device=early[early_layer][name].device,
                dtype=early[early_layer][name].dtype,
            )
    return result


def proposal_diagnostics(
    applied: dict[int, dict[str, torch.Tensor]],
    reference: dict[int, dict[str, torch.Tensor]],
    correspondence: dict[int, int] | None = None,
    global_norm_match: bool = False,
) -> dict:
    if correspondence is None:
        correspondence = {layer: layer for layer in applied}
    rows = []
    for applied_layer, reference_layer in correspondence.items():
        for name in KV_NAMES:
            applied_norm = float(applied[applied_layer][name].float().norm().item())
            reference_norm = float(reference[reference_layer][name].float().norm().item())
            relative_error = abs(applied_norm - reference_norm) / max(reference_norm, 1e-12)
            cosine = None
            if applied[applied_layer][name].shape == reference[reference_layer][name].shape:
                left = applied[applied_layer][name].float().flatten()
                right = reference[reference_layer][name].float().flatten()
                denominator = left.norm() * right.norm()
                cosine = float(left.dot(right).div(denominator.clamp_min(1e-12)).item())
            rows.append(
                {
                    "applied_layer": int(applied_layer),
                    "reference_layer": int(reference_layer),
                    "kv": name,
                    "applied_norm": applied_norm,
                    "reference_norm": reference_norm,
                    "relative_norm_error": relative_error,
                    "cosine_to_repair": cosine,
                }
            )
    applied_total = float(sum(row["applied_norm"] ** 2 for row in rows) ** 0.5)
    reference_total = float(sum(row["reference_norm"] ** 2 for row in rows) ** 0.5)
    global_relative_error = abs(applied_total - reference_total) / max(reference_total, 1e-12)
    return {
        "per_tensor": rows,
        "norm_match_mode": "global_band" if global_norm_match else "per_tensor",
        "applied_norm_total": applied_total,
        "reference_norm_total": reference_total,
        "max_relative_norm_error": global_relative_error
        if global_norm_match
        else max((row["relative_norm_error"] for row in rows), default=0.0),
    }


def apply_proposal(prefix: dict, proposal: dict[int, dict[str, torch.Tensor]], alpha: float) -> list[float]:
    ratios = []
    with torch.no_grad():
        for layer in proposal:
            for name in KV_NAMES:
                host = image_view(prefix, layer, name)
                delta = proposal[layer][name].to(device=host.device, dtype=host.dtype)
                if delta.shape != host.shape:
                    raise AssertionError(f"proposal shape {delta.shape} != cache view {host.shape}")
                ratios.append(float(abs(alpha) * delta.float().norm() / host.float().norm().clamp_min(1e-12)))
                host.add_(delta, alpha=float(alpha))
    return ratios
