from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluation import metrics_from_rankings, rerank_prefix
from scorers import build_specs, score_pairs_all, score_similarity_tensor


class ScorerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.text = F.normalize(torch.randn(3, 4, 8), dim=-1)
        self.image = F.normalize(torch.randn(3, 5, 8), dim=-1)
        self.mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ])
        self.qglobal = F.normalize(torch.randn(3, 8), dim=-1)
        self.gglobal = F.normalize(torch.randn(3, 8), dim=-1)

    def test_maxsim_matches_manual_formula(self):
        similarities = torch.bmm(self.text, self.image.transpose(1, 2))
        spec = build_specs([{"name": "max", "kind": "maxsim"}])
        actual = score_similarity_tensor(similarities, self.mask, spec)["max"]
        token_max = similarities.max(2).values
        expected = (token_max * self.mask).sum(1) / self.mask.sum(1)
        torch.testing.assert_close(actual, expected)

    def test_all_scorers_are_finite(self):
        specs = build_specs([
            {"name": "max", "kind": "maxsim"},
            {"name": "topm", "kind": "topm", "m": 3},
            {"name": "scan", "kind": "softmax_t2i", "temperature": 0.05},
            {"name": "chamfer", "kind": "smooth_chamfer", "alpha": 10},
            {"name": "cfine", "kind": "cfine_cross_grained"},
            {"name": "flow", "kind": "tokenflow_stable"},
            {"name": "ot", "kind": "sinkhorn_ot", "iterations": 5},
            {"name": "mean", "kind": "uniform_mean"},
        ])
        similarities = torch.bmm(self.text, self.image.transpose(1, 2))
        values = score_similarity_tensor(
            similarities,
            self.mask,
            specs,
            self.text,
            self.image,
            self.qglobal,
            self.gglobal,
        )
        self.assertEqual(set(values), {spec.name for spec in specs})
        for scores in values.values():
            self.assertEqual(tuple(scores.shape), (3,))
            self.assertTrue(torch.isfinite(scores).all())

    def test_pair_batching_fans_out(self):
        specs = build_specs([
            {"name": "max", "kind": "maxsim"},
            {"name": "top2", "kind": "topm", "m": 2},
        ])
        queries = torch.tensor([0, 1, 2, 0])
        galleries = torch.tensor([2, 1, 0, 1])
        result = score_pairs_all(
            specs,
            self.text,
            self.mask,
            self.image,
            queries,
            galleries,
            torch.device("cpu"),
            batch_size=2,
            query_global=self.qglobal,
            gallery_global=self.gglobal,
        )
        self.assertEqual(tuple(result["max"].shape), (4,))
        self.assertEqual(tuple(result["top2"].shape), (4,))


class EvaluationTests(unittest.TestCase):
    def test_rerank_and_metrics(self):
        global_rankings = torch.tensor([[1, 0, 2], [0, 1, 2]])
        local_scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        reranked = rerank_prefix(global_rankings, local_scores, 2)
        qids = torch.tensor([10, 20])
        gids = torch.tensor([10, 20, 30])
        metrics = metrics_from_rankings(reranked, qids, gids)
        self.assertEqual(metrics["r1"], 50.0)
        self.assertEqual(metrics["r5"], 100.0)


if __name__ == "__main__":
    unittest.main()
