from __future__ import annotations

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")


MODULE_DIR = Path(__file__).parents[1] / "scripts" / "vla"
sys.path.insert(0, str(MODULE_DIR))

from attention_pathway import (  # noqa: E402
    AllowCurrent,
    SourceEffectCapture,
    SourceKVSubstitution,
    output_layout,
    ranked_heads_by_write,
    ranked_edges_by_effect,
    replace_query_head_block,
)


def fake_eager(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    del module, dropout, kwargs
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


def make_call():
    generator = torch.Generator().manual_seed(9)
    query = torch.randn(1, 2, 3, 4, generator=generator)
    key = torch.randn(1, 2, 5, 4, generator=generator)
    value = torch.randn(1, 2, 5, 4, generator=generator)
    output, weights = fake_eager(None, query, key, value, None, 0.5)
    return {
        "layer_index": 0,
        "step_index": None,
        "module": None,
        "query": query,
        "key": key,
        "value": value,
        "attention_mask": None,
        "scaling": 0.5,
        "dropout": 0.0,
        "output": output,
        "weights": weights,
        "original_forward": fake_eager,
        "forward_kwargs": {},
    }


def test_output_layout_and_narrow_block_replacement() -> None:
    call = make_call()
    receiver = call["output"]
    donor = receiver + 10.0
    patched = replace_query_head_block(receiver, donor, call["query"], [1], [0])
    assert output_layout(receiver, call["query"]) == "BQHD"
    assert torch.equal(patched[:, 1, 0], donor[:, 1, 0])
    assert torch.equal(patched[:, 0, 0], receiver[:, 0, 0])
    assert torch.equal(patched[:, 1, 1], receiver[:, 1, 1])


def test_source_substitution_changes_only_selected_query_head_block() -> None:
    call = make_call()
    donor_key = call["key"].clone()
    donor_value = call["value"].clone()
    donor_key[:, :, 2, :] += 3.0
    donor_value[:, :, 2, :] -= 2.0
    intervention = SourceKVSubstitution(
        donor_keys={0: donor_key},
        donor_values={0: donor_value},
        source_positions=[2],
        receiver_positions_by_layer={0: [1]},
        heads_by_layer={0: [0]},
    )
    patched_output, patched_weights = intervention(**call)

    alternate_key = call["key"].clone()
    alternate_value = call["value"].clone()
    alternate_key[:, :, 2, :] = donor_key[:, :, 2, :]
    alternate_value[:, :, 2, :] = donor_value[:, :, 2, :]
    expected_output, expected_weights = fake_eager(
        None, call["query"], alternate_key, alternate_value, None, 0.5
    )
    assert torch.allclose(patched_output[:, 1, 0], expected_output[:, 1, 0])
    assert torch.allclose(patched_weights[:, 0, 1], expected_weights[:, 0, 1])
    assert torch.equal(patched_output[:, 0, 0], call["output"][:, 0, 0])
    assert torch.equal(patched_output[:, 1, 1], call["output"][:, 1, 1])


def test_allow_current_restores_rescue_block() -> None:
    call = make_call()
    donor_key = call["key"].clone() + 1.0
    donor_value = call["value"].clone() - 1.0
    intervention = SourceKVSubstitution(
        donor_keys={0: donor_key},
        donor_values={0: donor_value},
        source_positions=[2],
        receiver_positions_by_layer={0: [0, 1]},
        allow_current={0: [AllowCurrent(receiver_positions=(1,), heads=(0,))]},
    )
    patched_output, patched_weights = intervention(**call)
    assert torch.equal(patched_output[:, 1, 0], call["output"][:, 1, 0])
    assert torch.equal(patched_weights[:, 0, 1], call["weights"][:, 0, 1])
    assert not torch.equal(patched_output[:, 0, 0], call["output"][:, 0, 0])


def test_identity_substitution_is_exact() -> None:
    call = make_call()
    intervention = SourceKVSubstitution(
        donor_keys={0: call["key"].clone()},
        donor_values={0: call["value"].clone()},
        source_positions=[2],
        receiver_positions_by_layer={0: [0, 1, 2]},
    )
    output, weights = intervention(**call)
    assert torch.equal(output, call["output"])
    assert torch.equal(weights, call["weights"])


def test_prefix_only_donor_can_patch_prefix_sources_in_longer_reader_kv() -> None:
    call = make_call()
    call["key"] = torch.cat([call["key"], call["key"][:, :, :2, :]], dim=2)
    call["value"] = torch.cat([call["value"], call["value"][:, :, :2, :]], dim=2)
    call["output"], call["weights"] = fake_eager(
        None, call["query"], call["key"], call["value"], None, 0.5
    )
    donor_key = call["key"][:, :, :5, :].clone()
    donor_value = call["value"][:, :, :5, :].clone()
    donor_value[:, :, 2, :] -= 2.0
    intervention = SourceKVSubstitution(
        donor_keys={0: donor_key},
        donor_values={0: donor_value},
        source_positions=[2],
        receiver_positions_by_layer={0: [1]},
    )
    output, _ = intervention(**call)
    assert not torch.equal(output[:, 1], call["output"][:, 1])
    assert torch.equal(output[:, 0], call["output"][:, 0])


def test_head_ranking_uses_prompt_induced_receiver_write() -> None:
    output_a = torch.zeros(1, 3, 2, 4)
    output_b = output_a.clone()
    output_b[:, [0, 1], 1, :] = 2.0
    output_b[:, [0, 1], 0, :] = 0.5
    ranked = ranked_heads_by_write(output_a, output_b, (1, 2, 3, 4), [0, 1])
    assert [row["head"] for row in ranked] == [1, 0]
    assert ranked[-1]["cumulative_squared_fraction"] == pytest.approx(1.0)


def test_source_effect_capture_measures_without_changing_forward() -> None:
    call = make_call()
    donor_key = call["key"].clone()
    donor_value = call["value"].clone()
    donor_value[:, :, 2, :] += 4.0
    capture = SourceEffectCapture(
        donor_keys={0: donor_key},
        donor_values={0: donor_value},
        source_positions=[2],
        layers=[0],
    )
    assert capture(**call) is None
    ranked = ranked_edges_by_effect([capture])
    assert len(ranked) == call["query"].shape[1]
    assert ranked[-1]["cumulative_squared_fraction"] == pytest.approx(1.0)
    assert capture.calls == {0: 1}
