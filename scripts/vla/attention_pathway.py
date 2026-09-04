#!/usr/bin/env python
"""Attention-edge capture and source-substitution primitives for pi0.5.

The intervention implemented here is narrower than residual or cache replacement.
For selected receiver queries and attention heads, it recomputes attention after
replacing only a named set of source-token K/V vectors with a counterfactual
donor's K/V vectors. All untargeted query/head outputs remain exactly as produced
by the receiver run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


def _positions(values: Iterable[int], *, device: torch.device, upper: int, name: str) -> torch.Tensor:
    index = torch.as_tensor(sorted(set(int(value) for value in values)), dtype=torch.long, device=device)
    if index.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if int(index.min()) < 0 or int(index.max()) >= upper:
        raise IndexError(f"{name} outside [0, {upper}): {index.tolist()}")
    return index


def output_layout(output: torch.Tensor, query: torch.Tensor) -> str:
    """Return the eager-attention output layout used by the installed Transformers build."""
    if output.ndim != 4 or query.ndim != 4:
        raise ValueError(f"expected rank-4 output/query, got {output.shape}/{query.shape}")
    query_heads, query_length = int(query.shape[1]), int(query.shape[2])
    if int(output.shape[1]) == query_length and int(output.shape[2]) == query_heads:
        return "BQHD"
    if int(output.shape[1]) == query_heads and int(output.shape[2]) == query_length:
        return "BHQD"
    raise ValueError(f"cannot align output {output.shape} with query {query.shape}")


def replace_query_head_block(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    query: torch.Tensor,
    receiver_positions: Iterable[int],
    heads: Iterable[int],
) -> torch.Tensor:
    """Copy a receiver-position × query-head block while preserving every other element."""
    if receiver.shape != donor.shape:
        raise ValueError(f"receiver/donor shape mismatch: {receiver.shape} != {donor.shape}")
    layout = output_layout(receiver, query)
    q_index = _positions(
        receiver_positions,
        device=receiver.device,
        upper=int(query.shape[2]),
        name="receiver_positions",
    )
    h_index = _positions(heads, device=receiver.device, upper=int(query.shape[1]), name="heads")
    result = receiver.clone()
    if layout == "BQHD":
        result[:, q_index[:, None], h_index[None, :], :] = donor[:, q_index[:, None], h_index[None, :], :]
    else:
        result[:, h_index[:, None], q_index[None, :], :] = donor[:, h_index[:, None], q_index[None, :], :]
    return result


def replace_weight_block(
    receiver: torch.Tensor | None,
    donor: torch.Tensor | None,
    receiver_positions: Iterable[int],
    heads: Iterable[int],
) -> torch.Tensor | None:
    """Mirror an output intervention in returned attention weights when weights are available."""
    if receiver is None or donor is None:
        return receiver
    if receiver.shape != donor.shape or receiver.ndim != 4:
        raise ValueError("attention-weight tensors must be matching [B, H, Q, K] arrays")
    q_index = _positions(
        receiver_positions,
        device=receiver.device,
        upper=int(receiver.shape[2]),
        name="receiver_positions",
    )
    h_index = _positions(heads, device=receiver.device, upper=int(receiver.shape[1]), name="heads")
    result = receiver.clone()
    result[:, h_index[:, None], q_index[None, :], :] = donor[:, h_index[:, None], q_index[None, :], :]
    return result


@dataclass
class AttentionCapture:
    """Capture pre-output-projection Q/K/V and per-head attention outputs."""

    layers: set[int] | None = None
    keep_output_layers: set[int] = field(default_factory=set)
    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    query_heads: dict[int, int] = field(default_factory=dict)
    query_shapes: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def __call__(self, **call: Any) -> None:
        layer = int(call["layer_index"])
        if self.layers is not None and layer not in self.layers:
            return None
        self.keys[layer] = call["key"].detach().cpu().clone()
        self.values[layer] = call["value"].detach().cpu().clone()
        self.query_heads[layer] = int(call["query"].shape[1])
        self.query_shapes[layer] = tuple(int(value) for value in call["query"].shape)
        if layer in self.keep_output_layers:
            self.outputs[layer] = call["output"].detach().float().cpu().clone()
        return None


@dataclass(frozen=True)
class AllowCurrent:
    """A query/head block exempted from an otherwise active source substitution."""

    receiver_positions: tuple[int, ...]
    heads: tuple[int, ...]


class SourceKVSubstitution:
    """Replace source-token K/V only for selected receiver-query/head outputs.

    ``donor_keys`` and ``donor_values`` are normally captured from clean-A.
    The receiver forward is normally clean-B. Re-running eager attention with
    A's source K/V and B's current queries/non-source K/V removes only the
    communication attributable to the chosen source positions. ``allow_current``
    supports rescue: selected head/query blocks retain the original B output.
    """

    def __init__(
        self,
        *,
        donor_keys: dict[int, torch.Tensor],
        donor_values: dict[int, torch.Tensor],
        source_positions: Iterable[int],
        receiver_positions_by_layer: dict[int, Iterable[int]],
        heads_by_layer: dict[int, Iterable[int]] | None = None,
        allow_current: dict[int, list[AllowCurrent]] | None = None,
    ) -> None:
        layers = set(receiver_positions_by_layer)
        if not layers:
            raise ValueError("at least one intervention layer is required")
        missing = layers - set(donor_keys) | layers - set(donor_values)
        if missing:
            raise KeyError(f"donor K/V missing layers: {sorted(missing)}")
        self.donor_keys = donor_keys
        self.donor_values = donor_values
        self.source_positions = tuple(sorted(set(int(value) for value in source_positions)))
        if not self.source_positions:
            raise ValueError("source_positions must be nonempty")
        self.receivers = {
            int(layer): tuple(sorted(set(int(value) for value in positions)))
            for layer, positions in receiver_positions_by_layer.items()
        }
        self.heads = None if heads_by_layer is None else {
            int(layer): tuple(sorted(set(int(value) for value in heads)))
            for layer, heads in heads_by_layer.items()
        }
        self.allow_current = allow_current or {}

    def __call__(self, **call: Any) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        layer = int(call["layer_index"])
        if layer not in self.receivers:
            return None
        key, value = call["key"], call["value"]
        donor_key = self.donor_keys[layer].to(device=key.device, dtype=key.dtype)
        donor_value = self.donor_values[layer].to(device=value.device, dtype=value.dtype)
        same_kv_axes = (
            donor_key.ndim == key.ndim == 4
            and donor_value.ndim == value.ndim == 4
            and donor_key.shape[:2] == key.shape[:2]
            and donor_value.shape[:2] == value.shape[:2]
            and donor_key.shape[3] == key.shape[3]
            and donor_value.shape[3] == value.shape[3]
        )
        if not same_kv_axes:
            raise ValueError(
                f"layer {layer} donor/current K/V non-sequence axes mismatch: "
                f"{donor_key.shape}/{key.shape}, {donor_value.shape}/{value.shape}"
            )
        source_index = _positions(
            self.source_positions,
            device=key.device,
            upper=int(key.shape[2]),
            name="source_positions",
        )
        if int(source_index.max()) >= int(donor_key.shape[2]) or int(source_index.max()) >= int(donor_value.shape[2]):
            raise IndexError(
                f"layer {layer} donor K/V does not contain every source position: "
                f"max={int(source_index.max())}, donor lengths={donor_key.shape[2]}/{donor_value.shape[2]}"
            )
        alternate_key = key.clone()
        alternate_value = value.clone()
        alternate_key[:, :, source_index, :] = donor_key[:, :, source_index, :]
        alternate_value[:, :, source_index, :] = donor_value[:, :, source_index, :]
        alternate_output, alternate_weights = call["original_forward"](
            call["module"],
            call["query"],
            alternate_key,
            alternate_value,
            call["attention_mask"],
            call["scaling"],
            dropout=call["dropout"],
            **call["forward_kwargs"],
        )
        n_heads = int(call["query"].shape[1])
        heads = range(n_heads) if self.heads is None or layer not in self.heads else self.heads[layer]
        patched_output = replace_query_head_block(
            call["output"],
            alternate_output,
            call["query"],
            self.receivers[layer],
            heads,
        )
        patched_weights = replace_weight_block(
            call["weights"],
            alternate_weights,
            self.receivers[layer],
            heads,
        )
        for exception in self.allow_current.get(layer, []):
            patched_output = replace_query_head_block(
                patched_output,
                call["output"],
                call["query"],
                exception.receiver_positions,
                exception.heads,
            )
            patched_weights = replace_weight_block(
                patched_weights,
                call["weights"],
                exception.receiver_positions,
                exception.heads,
            )
        return patched_output, patched_weights


class SourceEffectCapture:
    """Measure per-head output change from source K/V substitution without intervening.

    This is used for behavior-blind reader selection. Scores accumulate the
    squared output difference over batches, receiver queries, head dimensions,
    and repeated denoising steps. Returning ``None`` leaves the live forward
    pass completely unchanged.
    """

    def __init__(
        self,
        *,
        donor_keys: dict[int, torch.Tensor],
        donor_values: dict[int, torch.Tensor],
        source_positions: Iterable[int],
        layers: Iterable[int],
    ) -> None:
        self.layers = tuple(sorted(set(int(layer) for layer in layers)))
        if not self.layers:
            raise ValueError("at least one measurement layer is required")
        missing = set(self.layers) - set(donor_keys) | set(self.layers) - set(donor_values)
        if missing:
            raise KeyError(f"donor K/V missing layers: {sorted(missing)}")
        self.donor_keys = donor_keys
        self.donor_values = donor_values
        self.source_positions = tuple(sorted(set(int(value) for value in source_positions)))
        if not self.source_positions:
            raise ValueError("source_positions must be nonempty")
        self.squared_effect: dict[int, torch.Tensor] = {}
        self.calls: dict[int, int] = {}

    def __call__(self, **call: Any) -> None:
        layer = int(call["layer_index"])
        if layer not in self.layers:
            return None
        key, value = call["key"], call["value"]
        donor_key = self.donor_keys[layer].to(device=key.device, dtype=key.dtype)
        donor_value = self.donor_values[layer].to(device=value.device, dtype=value.dtype)
        if (
            donor_key.ndim != 4
            or donor_value.ndim != 4
            or donor_key.shape[:2] != key.shape[:2]
            or donor_value.shape[:2] != value.shape[:2]
            or donor_key.shape[3] != key.shape[3]
            or donor_value.shape[3] != value.shape[3]
        ):
            raise ValueError(f"layer {layer} donor/current K/V axes do not match")
        source_index = _positions(
            self.source_positions,
            device=key.device,
            upper=int(key.shape[2]),
            name="source_positions",
        )
        if int(source_index.max()) >= min(int(donor_key.shape[2]), int(donor_value.shape[2])):
            raise IndexError(f"layer {layer} donor K/V does not contain every source position")
        alternate_key = key.clone()
        alternate_value = value.clone()
        alternate_key[:, :, source_index, :] = donor_key[:, :, source_index, :]
        alternate_value[:, :, source_index, :] = donor_value[:, :, source_index, :]
        alternate_output, _ = call["original_forward"](
            call["module"],
            call["query"],
            alternate_key,
            alternate_value,
            call["attention_mask"],
            call["scaling"],
            dropout=call["dropout"],
            **call["forward_kwargs"],
        )
        delta = alternate_output.float() - call["output"].float()
        layout = output_layout(delta, call["query"])
        axes = (0, 1, 3) if layout == "BQHD" else (0, 2, 3)
        score = delta.square().sum(dim=axes).detach().cpu()
        if layer not in self.squared_effect:
            self.squared_effect[layer] = torch.zeros_like(score)
            self.calls[layer] = 0
        self.squared_effect[layer] += score
        self.calls[layer] += 1
        return None


def ranked_edges_by_effect(captures: Iterable[SourceEffectCapture]) -> list[dict[str, float | int]]:
    """Aggregate source-effect captures and rank layer/head reader edges."""
    total_by_edge: dict[tuple[int, int], float] = {}
    calls_by_layer: dict[int, int] = {}
    for capture in captures:
        for layer, scores in capture.squared_effect.items():
            calls_by_layer[layer] = calls_by_layer.get(layer, 0) + capture.calls.get(layer, 0)
            for head, score in enumerate(scores.tolist()):
                total_by_edge[(int(layer), int(head))] = total_by_edge.get((int(layer), int(head)), 0.0) + float(score)
    total = sum(total_by_edge.values())
    cumulative = 0.0
    rows: list[dict[str, float | int]] = []
    for rank, ((layer, head), score) in enumerate(
        sorted(total_by_edge.items(), key=lambda item: (-item[1], item[0])), start=1
    ):
        cumulative += score
        rows.append(
            {
                "rank": rank,
                "layer": layer,
                "head": head,
                "squared_effect": score,
                "effect_norm": score**0.5,
                "cumulative_squared_fraction": 0.0 if total == 0.0 else cumulative / total,
                "layer_calls": calls_by_layer.get(layer, 0),
            }
        )
    return rows


def ranked_heads_by_write(
    output_a: torch.Tensor,
    output_b: torch.Tensor,
    query_shape: tuple[int, ...],
    receiver_positions: Iterable[int],
) -> list[dict[str, float | int]]:
    """Rank block-0 heads by their prompt-induced write magnitude at receiver positions."""
    if output_a.shape != output_b.shape:
        raise ValueError("A/B attention outputs must have matching shapes")
    fake_query = torch.empty(query_shape)
    layout = output_layout(output_a, fake_query)
    q_index = _positions(
        receiver_positions,
        device=output_a.device,
        upper=int(query_shape[2]),
        name="receiver_positions",
    )
    delta = output_b.float() - output_a.float()
    if layout == "BQHD":
        norms = delta[:, q_index, :, :].square().sum(dim=(0, 1, 3)).sqrt()
    else:
        norms = delta[:, :, q_index, :].square().sum(dim=(0, 2, 3)).sqrt()
    total_sq = float(norms.square().sum().item())
    order = torch.argsort(norms, descending=True).tolist()
    cumulative = 0.0
    ranked: list[dict[str, float | int]] = []
    for rank, head in enumerate(order, start=1):
        magnitude = float(norms[head].item())
        cumulative += magnitude * magnitude
        ranked.append(
            {
                "rank": rank,
                "head": int(head),
                "write_norm": magnitude,
                "cumulative_squared_fraction": 0.0 if total_sq == 0.0 else cumulative / total_sq,
            }
        )
    return ranked
