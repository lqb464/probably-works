from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from uncertainty import IdentityBootstrap, decision_intervals
from protocol_v2 import (PairHead, permute_oriented, prepare_splits,
                         calibration_bound, run_suite_v2, query_metrics)
from validated import SplitScores, evaluate_reranking
from diagnostic.global_hidden_recovery.feature_extractor import GlobalFeatureSet, HiddenFeatureSet
from scorers import build_specs

torch.set_num_threads(1)


def fixture(first_id=0, identities=24, seed=1):
    rng = np.random.default_rng(seed)
    pids = np.arange(first_id, first_id+identities)
    qids = torch.tensor(np.repeat(pids, 2))
    gids = torch.tensor(pids)
    similarity = torch.tensor(rng.normal(0, .1, (2*identities, identities)), dtype=torch.float32)
    for i in range(len(qids)):
        similarity[i, i//2] += .15
    rankings = similarity.argsort(dim=1, descending=True, stable=True)
    top = rankings[:, :4]
    g = GlobalFeatureSet(torch.randn(len(qids), 8), torch.randn(len(gids), 8), qids, gids,
                         [str(i) for i in range(len(qids))])
    h = HiddenFeatureSet(torch.nn.functional.normalize(torch.randn(len(qids), 3, 8), dim=-1),
                         torch.ones(len(qids), 3, dtype=torch.bool), torch.zeros(len(qids), 3, dtype=torch.long),
                         torch.nn.functional.normalize(torch.randn(len(gids), 4, 8), dim=-1), (2, 2), "float32")
    local = torch.tensor(rng.normal(0, .1, top.shape), dtype=torch.float32)
    local += gids[top].eq(qids[:, None])*.2
    return SplitScores("source", g, h, similarity, rankings, top, similarity.gather(1, top), {"maxsim": local})


class BootstrapTests(unittest.TestCase):
    def test_paired_identical_has_zero_delta(self):
        base = np.array([1, 1, 0, 0, 1, 0], bool)
        row = decision_intervals([1, 1, 2, 2, 3, 3], base, base, 2000)
        self.assertEqual(row["delta_r1_ci_low"], 0)
        self.assertEqual(row["delta_r1_ci_high"], 0)

    def test_cluster_resampling_keeps_siblings_and_nan_denominators(self):
        b = IdentityBootstrap([1, 1, 2, 2], 2000)
        point, draws = b.ratio([0, 0, 1, 1], np.ones(4))
        self.assertEqual(point, .5)
        self.assertTrue(set(draws) <= {0., .5, 1.})
        row = b.report("undefined", np.zeros(4), np.zeros(4))
        self.assertTrue(np.isnan(row["undefined_ci_high"]))

    def test_auc_matches_weighted_sklearn_with_ties(self):
        labels = [0, 1, 0, 1, 0, 1]
        scores = [.1, .4, .4, .7, .2, .7]
        weights = np.array([1, 1, 2, 2, 1, 1])
        b = IdentityBootstrap([1, 1, 2, 2, 3, 3], 50)
        actual, draws = b.auc(labels, scores, weights)
        self.assertAlmostEqual(actual, roc_auc_score(labels, scores, sample_weight=weights))
        for i in range(5):
            self.assertAlmostEqual(draws[i], roc_auc_score(labels, scores, sample_weight=weights*b.counts[i,b.inverse]))


class ProtocolTests(unittest.TestCase):
    def test_permutation_breaks_label_orientation_leak(self):
        rng = np.random.default_rng(15)
        orientation = rng.choice([-1., 1.], size=20000)
        delta = np.ones(len(orientation))
        control = permute_oriented(delta, orientation, 8)
        self.assertLess(abs(np.corrcoef(control, orientation)[0,1]), .03)
        # Regress old bug: permutation of raw positive deltas then orientation gives perfect labels.
        self.assertEqual(np.corrcoef(rng.permutation(delta)*orientation, orientation)[0,1], 1)

    def test_pair_head_swap_antisymmetry(self):
        rng = np.random.default_rng(1)
        x = rng.normal(.4, 1, (100, 2))
        model = PairHead().fit(x, np.ones(len(x)))
        np.testing.assert_allclose(model.probability(x)+model.probability(-x), 1, atol=1e-12)
        a, b = x[:20], x[20:40]
        np.testing.assert_allclose(model.score(a-b), model.score(a)-model.score(b), atol=1e-12)

    def test_icfg_holdout_and_official_validation_are_disjoint(self):
        torch.manual_seed(2)
        data = fixture(identities=40)
        states, source = prepare_splits(None, data, {"missing_validation": "test_identity_holdout"}, 42)
        sets = [set(s.global_features.qids.tolist()) for s in states.values()]
        self.assertEqual(len(set.union(*sets)), 40)
        for i,a in enumerate(sets):
            for b in sets[i+1:]:
                self.assertFalse(a & b)
        for state in states.values():
            self.assertEqual(len(state.global_features.gids), 40)
        with self.assertRaises(ValueError):
            prepare_splits(None, data, {}, 42)

    def test_noop_wins_over_harmful_candidate(self):
        state = fixture()
        # Force local scores to prefer last in global top-K.
        state.local_topk_scores["maxsim"] = torch.arange(4).float().expand(len(state.rankings), -1)
        specs = build_specs([{"name":"maxsim", "kind":"maxsim"}])
        # Ensure all original top1 are correct.
        state.global_features.qids = state.global_features.gids[state.rankings[:,0]]
        _, selected, results = evaluate_reranking(state, state, specs, [2,4], [.5], [])
        self.assertEqual(selected[-1]["mode"], "noop")
        self.assertEqual(results[-1]["test_r1"], 100.)
        choice = {"mode":"noop", "scorer":"__global__", "k":1}
        self.assertEqual(query_metrics(state, choice)["r1"].mean(), 1.)

    def test_no_errors_does_not_certify_zero_risk(self):
        bound,n,k = calibration_bound(np.arange(50), np.ones(50,bool), np.zeros(50,bool), .05/3)
        self.assertGreater(bound, .05)
        self.assertEqual((n,k), (50,0))

    def test_stable_ties_and_per_query_metrics_match_global(self):
        state = fixture()
        choice = {"scorer":"maxsim", "mode":"learned", "k":4}
        state.local_topk_scores["maxsim"].zero_()
        new = query_metrics(state, choice)
        baseline = query_metrics(state, {"mode":"noop","k":1})
        for key in new:
            np.testing.assert_allclose(new[key], baseline[key])

    def test_test_labels_do_not_select_reranker(self):
        selection, test = fixture(), fixture(100,12,7)
        specs = build_specs([{"name":"maxsim","kind":"maxsim"}])
        _, choices, _ = evaluate_reranking(selection,test,specs,[2,4],[.1,.5],[])
        test.global_features.qids = test.global_features.qids.roll(1)
        _, changed, _ = evaluate_reranking(selection,test,specs,[2,4],[.1,.5],[])
        self.assertEqual(choices,changed)

    def test_entire_suite_writes_all_experiments_and_intervals(self):
        torch.manual_seed(3)
        validation, test = fixture(), fixture(100, 12, 7)
        specs = build_specs([{"name":"maxsim", "kind":"maxsim"}])
        config = {"seed":42, "topk":[2,4], "fusion_weights":[.1,.5], "validated":{
            "bootstrap_repetitions":30, "bootstrap_confidence":.95, "probe_negatives":2,
            "setwise_negative_counts":[1,2], "probe_c_grid":[1.], "gating_threshold_steps":3}}
        with tempfile.TemporaryDirectory() as folder:
            result = run_suite_v2(config, Path(folder), specs, validation, test, torch.device("cpu"), 64)
            out = Path(folder)/"validated_v2"
            for name in ["setwise_summary.csv", "probe_results.csv", "probe_primary_comparisons.csv",
                         "reranking_selected_test.csv", "gating_summary.csv", "split_assignments.csv",
                         "probe_predictions.npz", "reranking_query_metrics.npz", "gating_query_metrics.npz"]:
                self.assertTrue((out/name).stat().st_size > 0, name)
            self.assertIn("delta_r1_ci_low", result["reranking_test"][1])
            self.assertEqual(len(result["gating"]),6)
            self.assertEqual(result["protocol"]["bootstrap_repetitions"],30)
            self.assertEqual(sum(bool(r["selected_on_validation"]) for r in result["probe"]),1)
            # Reuse must fail instead of overwrite previous results.
            with self.assertRaises(FileExistsError):
                run_suite_v2(config, Path(folder), specs, validation, test, torch.device("cpu"),64)


if __name__ == "__main__":
    unittest.main()
