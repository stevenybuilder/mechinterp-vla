from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_reader_controls_are_nested_and_disjoint() -> None:
    from pi05_attention_resolution import validate_config

    config = json.loads((ROOT / "configs/pi05_attention_resolution.json").read_text())
    validate_config(config)


def test_writer_curve_classifier_separates_localized_and_distributed() -> None:
    from pi05_attention_resolution import writer_curve_analysis

    gate = json.loads((ROOT / "configs/pi05_attention_resolution.json").read_text())[
        "development_gates"
    ]["writer"]
    localized_prefix = [-1.0] * 9 + [1.0] * 9
    localized_suffix = [1.0] * 10 + [-1.0] * 8
    localized = writer_curve_analysis(localized_prefix, localized_suffix, -1.0, 1.0, gate)
    assert localized["classification"] == "localized"

    distributed_prefix = [-1.0 + 2.0 * (index + 1) / 18 for index in range(18)]
    distributed_suffix = [1.0 - 2.0 * index / 17 for index in range(18)]
    distributed = writer_curve_analysis(distributed_prefix, distributed_suffix, -1.0, 1.0, gate)
    assert distributed["classification"] == "distributed"


def test_reader_intervention_rejects_empty_explicit_edge_set() -> None:
    from pi05_attention_resolution import reader_intervention

    with pytest.raises(ValueError, match="at least one intervention layer"):
        reader_intervention({}, {}, [0], 2, [12], active_edges=[])
