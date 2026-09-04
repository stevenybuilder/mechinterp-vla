"""Pure geometry helpers for the sealed pi0.5 reusable-mediator experiment.

This module deliberately has no LIBERO or LeRobot imports so the numerical
contract can be tested without loading the policy or simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class StableSVD:
    token_basis: torch.Tensor
    singular_values: torch.Tensor
    feature_basis: torch.Tensor
    reliability: torch.Tensor

    @property
    def max_rank(self) -> int:
        return int(self.singular_values.numel())


def parse_id_spec(spec: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges without duplicates."""
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                raise ValueError(f"descending range is not allowed: {part}")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("empty ID specification")
    return list(dict.fromkeys(values))


def _low_rank_svd(matrix: torch.Tensor, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got {tuple(matrix.shape)}")
    rank = min(int(rank), *matrix.shape)
    if rank < 1:
        raise ValueError("rank must be positive")

    # Exact SVD is deterministic and preferable for unit-sized matrices. The
    # production 512 x hidden matrices use a seeded rank-q decomposition to
    # avoid materializing the unused 512-dimensional right factor.
    if rank == min(matrix.shape) or matrix.numel() <= 262_144:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u[:, :rank], s[:rank], vh[:rank].T

    devices = [matrix.device] if matrix.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        u, s, v = torch.pca_lowrank(matrix, q=rank, center=False, niter=4)
    return u, s, v


def fit_stable_svd(
    deltas: torch.Tensor,
    max_rank: int = 32,
    seed: int = 0,
    compute_device: str | torch.device | None = None,
) -> StableSVD:
    """Fit a paired mean SVD and shrink components unstable across scenes.

    ``deltas`` has shape [calibration_scene, image_token, hidden_feature].
    Reliability is signal^2 / (signal^2 + across-scene coefficient variance).
    """
    if deltas.ndim != 3 or deltas.shape[0] < 2:
        raise ValueError(f"expected [n>=2, token, hidden], got {tuple(deltas.shape)}")
    source_device = deltas.device
    work_device = torch.device(compute_device) if compute_device is not None else source_device
    work = deltas.float().to(work_device)
    mean = work.mean(dim=0)
    u, s, v = _low_rank_svd(mean, max_rank, seed)
    left = torch.einsum("tr,nth->nrh", u, work)
    coefficients = (left * v.T.unsqueeze(0)).sum(dim=-1)
    signal = coefficients.mean(dim=0).square()
    nuisance = coefficients.var(dim=0, unbiased=False)
    scale = torch.finfo(signal.dtype).eps * signal.mean().clamp_min(1.0)
    reliability = signal / (signal + nuisance + scale)
    return StableSVD(
        token_basis=u.detach().to(source_device),
        singular_values=s.detach().to(source_device),
        feature_basis=v.detach().to(source_device),
        reliability=reliability.detach().to(source_device),
    )


def stable_coefficients(factors: StableSVD, rank: int) -> torch.Tensor:
    """Reliability-shrink the selected modes at the original rank-k norm."""
    rank = int(rank)
    if not 1 <= rank <= factors.max_rank:
        raise ValueError(f"rank {rank} outside [1, {factors.max_rank}]")
    raw = factors.singular_values[:rank].float()
    weighted = raw * factors.reliability[:rank].float().clamp(0.0, 1.0)
    raw_norm = raw.norm()
    weighted_norm = weighted.norm()
    if weighted_norm <= torch.finfo(weighted.dtype).eps:
        return torch.zeros_like(weighted)
    return weighted * (raw_norm / weighted_norm)


def materialize(factors: StableSVD, rank: int) -> torch.Tensor:
    coefficients = stable_coefficients(factors, rank)
    u = factors.token_basis[:, :rank].float()
    v = factors.feature_basis[:, :rank].float()
    return (u * coefficients.unsqueeze(0)) @ v.T


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    a, b = left.float().reshape(-1), right.float().reshape(-1)
    denom = a.norm() * b.norm()
    if denom <= torch.finfo(a.dtype).eps:
        return 0.0
    return float(torch.dot(a, b) / denom)


def rescale_like(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    candidate = candidate.float()
    reference = reference.float()
    norm = candidate.norm()
    if norm <= torch.finfo(candidate.dtype).eps:
        return torch.zeros_like(candidate)
    return candidate * (reference.norm() / norm)


def matched_spectrum_random(factors: StableSVD, rank: int, seed: int) -> torch.Tensor:
    """Return a deterministic random-basis matrix with the fitted spectrum."""
    coefficients = stable_coefficients(factors, rank)
    n_token = factors.token_basis.shape[0]
    n_feature = factors.feature_basis.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    token_random = torch.randn(n_token, rank, generator=generator)
    feature_random = torch.randn(n_feature, rank, generator=generator)
    token_q, _ = torch.linalg.qr(token_random, mode="reduced")
    feature_q, _ = torch.linalg.qr(feature_random, mode="reduced")
    return (token_q * coefficients.cpu().unsqueeze(0)) @ feature_q.T


def median(values: Iterable[float]) -> float:
    tensor = torch.as_tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        raise ValueError("median of empty sequence")
    return float(tensor.median()) if tensor.numel() % 2 else float(tensor.sort().values[tensor.numel() // 2 - 1: tensor.numel() // 2 + 1].mean())

