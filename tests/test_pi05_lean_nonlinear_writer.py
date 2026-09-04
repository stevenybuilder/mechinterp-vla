from __future__ import annotations

import numpy as np


def test_curvature_ratio_is_zero_for_affine_midpoint() -> None:
    from pi05_lean_nonlinear_writer import curvature_ratio

    left = np.asarray([0.0, 2.0])
    right = np.asarray([2.0, 4.0])
    assert curvature_ratio(left, 0.5 * (left + right), right) == 0.0


def test_curvature_ratio_is_relative_to_endpoint_chord() -> None:
    from pi05_lean_nonlinear_writer import curvature_ratio

    left = np.asarray([0.0])
    right = np.asarray([2.0])
    midpoint = np.asarray([1.5])
    assert curvature_ratio(left, midpoint, right) == 0.25
