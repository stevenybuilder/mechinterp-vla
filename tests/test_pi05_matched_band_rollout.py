import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/vla"))

from pi05_matched_band_rollout import episode_seed


def test_episode_seed_separates_cells_and_initial_states():
    values = {
        episode_seed(2_600_000, task_a, task_b, init_id)
        for task_a, task_b in ((0, 6), (6, 0), (2, 5))
        for init_id in (33, 34, 35)
    }
    assert len(values) == 9


def test_unknown_intervention_condition_fails():
    from pi05_matched_band_rollout import intervention_prefix

    with pytest.raises(KeyError):
        intervention_prefix(None, {}, {}, {}, "not_a_condition", random_seed=0)
