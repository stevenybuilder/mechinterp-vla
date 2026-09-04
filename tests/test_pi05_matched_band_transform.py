from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/vla/pi05_matched_band_transform.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("pi05_matched_band_transform", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_axis_metrics_has_clean_endpoints() -> None:
    a = np.asarray([[0.0, 0.0]])
    b = np.asarray([[2.0, 0.0]])
    assert MODULE.axis_metrics(a, a, b) == {"progress_to_B": 0.0, "D_A": 0.0, "D_B": 1.0}
    assert MODULE.axis_metrics(b, a, b) == {"progress_to_B": 1.0, "D_A": 1.0, "D_B": 0.0}


def test_norm_match_preserves_target_norm() -> None:
    source = torch.tensor([[3.0, 4.0]])
    target = torch.tensor([[0.0, 12.0]])
    actual = MODULE.norm_match(source, target)
    assert torch.allclose(torch.linalg.vector_norm(actual), torch.linalg.vector_norm(target))


def test_sign_test_is_two_sided() -> None:
    result = MODULE.exact_two_sided_sign_test([1.0] * 12)
    assert result["positive"] == 12
    assert np.isclose(result["p_two_sided"], 2.0 / (2 ** 12))


def test_bootstrap_interval_is_deterministic() -> None:
    first = MODULE.bootstrap_median_interval(range(12), resamples=1000, seed=7)
    second = MODULE.bootstrap_median_interval(range(12), resamples=1000, seed=7)
    assert first == second
