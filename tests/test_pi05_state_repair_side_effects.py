import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/vla"))

from pi05_state_repair_side_effects import condition_spec, exact_mcnemar


def test_condition_spec_uses_fixed_host_donor_and_layer_roles():
    task = {"task_id": 1, "conflict_task_id": 5, "wrong_donor_task_id": 0}
    late = tuple(range(12, 18))
    early = tuple(range(6))
    assert condition_spec("clean_correct", task, late, early) == (1, None, None)
    assert condition_spec("preserve_correct", task, late, early) == (1, 1, late)
    assert condition_spec("repair_live", task, late, early) == (5, 1, late)
    assert condition_spec("repair_early", task, late, early) == (5, 1, early)
    assert condition_spec("wrong_donor_live", task, late, early) == (5, 0, late)


def test_exact_mcnemar_handles_unanimous_and_tied_pairs():
    assert exact_mcnemar([False] * 5, [True] * 5) == {
        "positive": 5,
        "negative": 0,
        "discordant": 5,
        "p_two_sided": 0.0625,
    }
    assert exact_mcnemar([False, True], [False, True])["p_two_sided"] == 1.0
