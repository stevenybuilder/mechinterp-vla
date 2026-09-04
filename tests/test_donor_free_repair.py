from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "vla"))

from donor_free_repair import (  # noqa: E402
    match_early_to_live,
    orthogonal_matched,
    predict_tensor_delta,
    random_matched,
    rescale_like,
    signed_task_code,
)


class DonorFreeRepairTests(unittest.TestCase):
    def test_frozen_edge_split_is_disjoint_and_has_five_pairs(self) -> None:
        config = json.loads((ROOT / "configs" / "pi05_donor_free_repair.json").read_text())
        training = {
            (int(target), int(source))
            for target, sources in config["training_edges"].items()
            for source in sources
        }
        heldout = {
            (int(item["target_task_id"]), int(item["source_task_id"]))
            for item in config["heldout_pairs"]
        }
        self.assertEqual(len(heldout), 5)
        self.assertFalse(training & heldout)
        self.assertFalse(set(config["calibration_init_ids"]) & set(config["confirmation_init_ids"]))

    def test_signed_task_code_composes_target_and_source(self) -> None:
        code = signed_task_code(2, 4, 6, device="cpu", dtype=torch.float32)
        self.assertTrue(torch.equal(code, torch.tensor([0.0, 0.0, 1.0, 0.0, -1.0, 0.0])))
        self.assertTrue(torch.equal(signed_task_code(2, 2, 6, device="cpu", dtype=torch.float32), torch.zeros(6)))

    def test_low_rank_predictor_has_expected_shape(self) -> None:
        host = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) / 10
        feature_dim = 3 + 5
        generator = torch.Generator().manual_seed(7)
        record = {
            "feature_dim": feature_dim,
            "max_rank": 2,
            "mean_feature": torch.zeros(feature_dim),
            "scale_feature": torch.ones(feature_dim),
            "mean_delta": torch.zeros(3),
            "u": torch.randn(feature_dim, 2, generator=generator),
            "s": torch.tensor([1.0, 0.5]),
            "vh": torch.randn(2, 3, generator=generator),
        }
        delta = predict_tensor_delta(host, record, 1, 4, 5, rank=2)
        self.assertEqual(delta.shape, host.shape)
        self.assertTrue(torch.isfinite(delta).all())

    def test_random_and_orthogonal_controls_are_norm_matched(self) -> None:
        generator = torch.Generator().manual_seed(11)
        target = {12: {"k": torch.randn(1, 2, 4, 3, generator=generator), "v": torch.randn(1, 2, 4, 3, generator=generator)}}
        random = random_matched(target, 21)
        orthogonal = orthogonal_matched(target, 22)
        for name in ("k", "v"):
            target_norm = target[12][name].norm()
            self.assertTrue(torch.allclose(random[12][name].norm(), target_norm, rtol=1e-6, atol=1e-6))
            self.assertTrue(torch.allclose(orthogonal[12][name].norm(), target_norm, rtol=1e-6, atol=1e-6))
            cosine = torch.nn.functional.cosine_similarity(
                orthogonal[12][name].flatten(), target[12][name].flatten(), dim=0
            )
            self.assertLess(abs(float(cosine)), 1e-6)

    def test_wrong_and_early_controls_are_norm_matched(self) -> None:
        target = {12: {"k": torch.ones(1, 1, 2, 2), "v": torch.ones(1, 1, 2, 2) * 2}}
        wrong = {12: {"k": torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2), "v": torch.tensor([[[[1.0, -1.0], [2.0, -2.0]]]])}}
        scaled = rescale_like(wrong, target)
        for name in ("k", "v"):
            self.assertTrue(torch.allclose(scaled[12][name].norm(), target[12][name].norm()))
        early = {0: {"k": wrong[12]["k"].clone(), "v": wrong[12]["v"].clone()}}
        early_scaled = match_early_to_live(early, target, [0], [12])
        early_total = torch.sqrt(sum(early_scaled[0][name].square().sum() for name in ("k", "v")))
        target_total = torch.sqrt(sum(target[12][name].square().sum() for name in ("k", "v")))
        self.assertTrue(torch.allclose(early_total, target_total))


if __name__ == "__main__":
    unittest.main()
