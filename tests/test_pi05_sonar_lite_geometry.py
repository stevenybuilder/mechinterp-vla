from __future__ import annotations

from pathlib import Path
import sys

import pytest
torch = pytest.importorskip("torch")


MODULE_DIR = Path(__file__).parents[1] / "scripts" / "vla"
sys.path.insert(0, str(MODULE_DIR))
import sonar_lite_geometry as geometry


def test_parse_id_spec_supports_ranges_and_deduplicates() -> None:
    assert geometry.parse_id_spec("0-2,5,2,7-8") == [0, 1, 2, 5, 7, 8]
    with pytest.raises(ValueError):
        geometry.parse_id_spec("3-1")


def test_stable_svd_recovers_repeated_low_rank_signal() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(8, 2, generator=generator)
    right = torch.randn(6, 2, generator=generator)
    signal = left @ right.T
    deltas = torch.stack(
        [signal + 0.01 * torch.randn(8, 6, generator=generator) for _ in range(6)]
    )
    factors = geometry.fit_stable_svd(deltas, max_rank=4, seed=11)
    estimate = geometry.materialize(factors, rank=2)

    assert geometry.cosine(estimate, signal) > 0.999
    assert torch.all((0.0 <= factors.reliability) & (factors.reliability <= 1.0))
    assert torch.isclose(
        geometry.stable_coefficients(factors, 2).norm(),
        factors.singular_values[:2].norm(),
    )


def test_matched_random_preserves_singular_spectrum() -> None:
    generator = torch.Generator().manual_seed(3)
    deltas = torch.randn(5, 10, 12, generator=generator)
    factors = geometry.fit_stable_svd(deltas, max_rank=4, seed=5)
    fitted = geometry.materialize(factors, rank=4)
    random = geometry.matched_spectrum_random(factors, rank=4, seed=19)

    fitted_s = torch.linalg.svdvals(fitted)[:4]
    random_s = torch.linalg.svdvals(random)[:4]
    assert torch.allclose(fitted_s, random_s, rtol=1e-4, atol=1e-5)
    assert geometry.cosine(fitted, random) < 0.5


def test_rescale_like_matches_norm() -> None:
    candidate = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    reference = torch.ones(2, 2)
    scaled = geometry.rescale_like(candidate, reference)
    assert torch.isclose(scaled.norm(), reference.norm())
