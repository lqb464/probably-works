from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostic.global_hidden_recovery.feature_extractor import GlobalHiddenExtractor
from diagnostic.global_hidden_recovery.evaluator import build_global_decisions
from diagnostic.global_hidden_recovery.hidden_scorer import HiddenScorer
from diagnostic.global_hidden_recovery.run_diagnostic import make_repo_args


class HiddenScorerTests(unittest.TestCase):
    def test_maxsim_manual_example(self):
        scorer = HiddenScorer(torch.device("cpu"))
        text = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        mask = torch.tensor([[True, True]])
        image = torch.tensor([[[1.0, 0.0], [0.0, 0.5], [0.0, 1.0]]])
        score = scorer.score_batch(text, mask, image)
        self.assertTrue(torch.allclose(score, torch.tensor([1.0])))

    def test_masked_tokens_do_not_contribute(self):
        scorer = HiddenScorer(torch.device("cpu"))
        text = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        mask = torch.tensor([[True, False]])
        image = torch.tensor([[[1.0, 0.0]]])
        score = scorer.score_batch(text, mask, image)
        self.assertTrue(torch.allclose(score, torch.tensor([1.0])))

    def test_score_pairs(self):
        scorer = HiddenScorer(torch.device("cpu"))
        text = torch.eye(2).view(2, 1, 2)
        mask = torch.ones(2, 1, dtype=torch.bool)
        image = torch.eye(2).view(2, 1, 2)
        scores = scorer.score_pairs(text, mask, image, [0, 1], [0, 1], batch_size=1)
        self.assertTrue(torch.allclose(scores, torch.ones(2)))


class EvaluatorTests(unittest.TestCase):
    def test_pair_fixation_and_multiple_positive_images(self):
        sim = torch.tensor([[0.90, 0.50, 0.80]])
        qids = torch.tensor([1])
        gids = torch.tensor([2, 1, 1])
        decisions = build_global_decisions(sim, qids, gids, [1, 2, 3])
        self.assertEqual(int(decisions["best_positive_index"][0]), 2)
        self.assertEqual(int(decisions["best_negative_index"][0]), 0)
        self.assertAlmostEqual(float(decisions["delta_global"][0]), -0.10, places=6)
        self.assertFalse(bool(decisions["positive_in_topk"][1][0]))
        self.assertTrue(bool(decisions["positive_in_topk"][2][0]))


class ExtractorTests(unittest.TestCase):
    def test_hidden_projection_shapes_and_normalization(self):
        model = DummyITSELF()
        extractor = GlobalHiddenExtractor(
            model,
            device=torch.device("cpu"),
            feature_dtype="float32",
        )
        caption = torch.tensor([[49406, 10, 11, 49407, 0]])
        image = torch.zeros(1, 3, 8, 8)
        hidden = extractor.extract_hidden_features(
            txt_loader=[(torch.tensor([7]), caption)],
            img_loader=[(torch.tensor([7]), image)],
        )

        self.assertEqual(tuple(hidden.text_features.shape), (1, 2, 512))
        self.assertEqual(tuple(hidden.image_features.shape), (1, 3, 512))
        self.assertTrue(torch.equal(hidden.token_ids[0], torch.tensor([10, 11])))
        self.assertTrue(torch.allclose(hidden.text_features.norm(dim=-1), torch.ones(1, 2)))
        self.assertTrue(torch.allclose(hidden.image_features.norm(dim=-1), torch.ones(1, 3)))


class PresetTests(unittest.TestCase):
    def test_clip_preset_flags(self):
        args = make_repo_args({"dataset": "CUHK-PEDES", "model_preset": "clip"})
        self.assertTrue(args.only_global)
        self.assertFalse(args.return_all)
        self.assertEqual(args.topk_type, "mean")
        self.assertFalse(args.modify_k)

    def test_itself_preset_flags(self):
        args = make_repo_args({"dataset": "RSTPReid", "model_preset": "itself"})
        self.assertFalse(args.only_global)
        self.assertTrue(args.return_all)
        self.assertEqual(args.topk_type, "custom")
        self.assertTrue(args.modify_k)

    def test_root_dir_override(self):
        args = make_repo_args({
            "dataset": "CUHK-PEDES",
            "model_preset": "clip",
            "root_dir": "/tmp/itself-data",
        })
        self.assertEqual(args.root_dir, "/tmp/itself-data")


class DummyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = type("Visual", (), {"num_y": 1, "num_x": 3})()

    def encode_text(self, caption):
        batch, length = caption.shape
        seq = torch.zeros(batch, length, 512)
        for pos in range(length):
            seq[:, pos, pos] = 2.0
        return seq, None

    def encode_image(self, image):
        batch = image.shape[0]
        seq = torch.zeros(batch, 4, 512)
        for pos in range(4):
            seq[:, pos, pos] = 3.0
        return seq, None


class DummyITSELF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = DummyBase()

    def encode_text(self, caption):
        seq, _ = self.base_model.encode_text(caption)
        return seq[:, 0]

    def encode_image(self, image):
        seq, _ = self.base_model.encode_image(image)
        return seq[:, 0]


if __name__ == "__main__":
    unittest.main()
