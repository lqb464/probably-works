"""Frozen retrieval protocol v2: RankNet linear heads + held-out risk screening.

See METHODS_V2.md for sources, adaptations, units and limits of inference.
No scorer, threshold or regularization is selected using evaluation labels.
"""
from __future__ import annotations

import logging
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from uncertainty import IdentityBootstrap, decision_intervals
from scorers import ScorerSpec
from validated import (build_oracle_set_scores, evaluate_setwise, evaluate_coverage,
                       iter_variants, get_variant_scores, top1_variant_summary,
                       probe_local_delta, build_probe_base,
                       write_csv, write_json)


def partition_ids(pids, fractions, seed):
    ids = np.unique(np.asarray(pids))
    if len(ids) < 8 or not np.isclose(sum(fractions), 1):
        raise ValueError("Need >=8 identities and partition fractions summing to one")
    ids = np.random.default_rng(seed).permutation(ids)
    sizes = [max(2, int(len(ids) * f)) for f in fractions[:-1]]
    if sum(sizes) > len(ids) - 2:
        raise ValueError("Too few identities for the requested split")
    return np.split(ids, np.cumsum(sizes))


def subset_split(split, ids, name):
    """Subset query identities only. All partitions retain their source gallery."""
    index = torch.from_numpy(np.flatnonzero(np.isin(split.global_features.qids.numpy(), ids)))
    gf, hf = split.global_features, split.hidden_features
    return replace(split, name=name,
        global_features=replace(gf, qids=gf.qids[index], qfeats=gf.qfeats[index],
                                captions=[gf.captions[i] for i in index.tolist()]),
        hidden_features=replace(hf, text_features=hf.text_features[index],
                                text_mask=hf.text_mask[index], token_ids=hf.token_ids[index]),
        similarity=split.similarity[index], rankings=split.rankings[index],
        topk_indices=split.topk_indices[index], global_topk_scores=split.global_topk_scores[index],
        local_topk_scores={k: v[index] for k, v in split.local_topk_scores.items()})


def prepare_splits(validation, test, cfg, seed):
    if validation is None:
        if cfg.get("missing_validation") != "test_identity_holdout":
            raise ValueError("No official validation split; set missing_validation=test_identity_holdout explicitly")
        dev_ids, eval_ids = partition_ids(test.global_features.qids.numpy(),
                                        [float(cfg.get("holdout_development_fraction", 0.2)),
                                         1-float(cfg.get("holdout_development_fraction", 0.2))], seed)
        validation = subset_split(test, dev_ids, "development")
        test = subset_split(test, eval_ids, "heldout_test")
        source = "official_test_identity_holdout; fixed official test gallery; exploratory"
    else:
        if np.intersect1d(validation.global_features.qids.numpy(), test.global_features.qids.numpy()).size:
            raise ValueError("Validation and test query IDs overlap; verify dataset ID namespaces")
        source = "official_validation; official_test; fixed source galleries"
    parts = partition_ids(validation.global_features.qids.numpy(),
                          cfg.get("development_fractions", [0.5, 0.25, 0.25]), seed + 10)
    states = {name: subset_split(validation, ids, name)
              for name, ids in zip(["fit", "selection", "calibration"], parts)}
    states["test"] = test
    return states, source


def permute_oriented(delta, orientation, seed):
    # Shuffle AFTER orientation. Multiplying by label-derived orientation again leaks labels.
    return np.random.default_rng(seed).permutation(delta * orientation)


class PairHead:
    """Linear RankNet pairwise logistic objective with L2 penalty and no intercept.

    Mirrored training pairs impose swap antisymmetry. Frozen item score w*x/scale
    can then be used in real top-K ranking without supplying positive labels.
    """
    def __init__(self, c=1.0):
        self.c = c

    def fit(self, raw_deltas, weights):
        xx = np.concatenate([raw_deltas, -raw_deltas])
        yy = np.r_[np.ones(len(raw_deltas)), np.zeros(len(raw_deltas))]
        ww = np.tile(weights, 2) / 2
        self.scaler = StandardScaler(with_mean=False).fit(xx, sample_weight=ww)
        self.model = LogisticRegression(C=self.c, fit_intercept=False, max_iter=1000,
                                         solver="lbfgs")
        self.model.fit(self.scaler.transform(xx), yy, sample_weight=ww)
        return self

    def probability(self, oriented_deltas):
        return self.model.predict_proba(self.scaler.transform(oriented_deltas))[:, 1]

    def score(self, items):
        return self.scaler.transform(items) @ self.model.coef_[0]

    def record(self):
        return {"C": self.c, "scale": self.scaler.scale_.tolist(),
                "coefficients": self.model.coef_[0].tolist(), "intercept": 0}


def pair_data(oracle, specs, negatives, seed):
    base = build_probe_base(oracle, negatives, seed)
    cols = {"global": base["raw_global_delta"]}
    cols.update({s.name: probe_local_delta(oracle, s.name, negatives) for s in specs})
    return base, cols


def fit_probes(oracles, specs, cfg, seed, output):
    negatives = int(cfg.get("probe_negatives", 10))
    data = {key: pair_data(value, specs, negatives, seed + i)
            for i, (key, value) in enumerate(oracles.items()) if key != "calibration"}
    fit, select, test = [data[key] for key in ["fit", "selection", "test"]]
    groups = [("__global__", "global_only", ["global"])]
    for spec in specs:
        groups += [(spec.name, "hidden_only", [spec.name]),
                   (spec.name, "global_plus_hidden", ["global", spec.name]),
                   (spec.name, "permuted_hidden_control", ["global", spec.name])]
    for kind in ["hidden_only", "global_plus_hidden", "permuted_hidden_control"]:
        columns = [s.name for s in specs]
        groups.append(("all_hidden", kind, columns if kind == "hidden_only" else ["global"] + columns))
    rows, predictions, fitted, choices, models = [], {}, {}, [], {}
    boot = IdentityBootstrap(test[0]["pids"], cfg["bootstrap_repetitions"],
                              cfg["bootstrap_confidence"], seed)
    weights, y = test[0]["weights"], test[0]["labels"]
    for gi, (scorer, kind, columns) in enumerate(groups):
        matrices = []
        for si, (base, cols) in enumerate([fit, select, test]):
            oriented = np.column_stack([cols[c] * base["orientation"] for c in columns])
            if kind == "permuted_hidden_control":
                # Same permutation for all hidden columns preserves their joint structure.
                oriented[:, 1:] = np.random.default_rng(seed + 1000 + gi * 10 + si).permutation(oriented[:, 1:])
            matrices.append(oriented)
        # Fit mirrored pairs from oriented data and labels; same result for non-controls.
        fit_raw = matrices[0] * fit[0]["orientation"][:, None]
        candidates = []
        for c in cfg.get("probe_c_grid", [0.1, 1.0, 10.0]):
            head = PairHead(float(c)).fit(fit_raw, fit[0]["weights"])
            loss = log_loss(select[0]["labels"], head.probability(matrices[1]),
                            sample_weight=select[0]["weights"], labels=[0, 1])
            candidates.append((loss, float(c), head))
        val_loss, c, head = min(candidates, key=lambda t: (t[0], t[1]))
        key = scorer + "/" + kind
        prob = head.probability(matrices[2])
        predictions[key] = prob
        fitted[key] = (head, columns)
        models[key] = {**head.record(), "columns": columns, "selection_log_loss": val_loss}
        row = {"scorer": scorer, "probe": kind, "selected_C": c,
               "selection_log_loss": val_loss, "train_split": "fit", "selection_split": "selection",
               "test_split": "test", "num_train_pairs": len(fit_raw), "num_test_pairs": len(y),
               "roc_auc": roc_auc_score(y, prob, sample_weight=weights)}
        correct = (prob >= 0.5) == y
        loss_vector = -(y * np.log(np.clip(prob, 1e-15, 1)) +
                        (1-y) * np.log(np.clip(1-prob, 1e-15, 1)))
        row.update(boot.report("pair_accuracy", weights * correct, weights))
        row.update(boot.report("log_loss", weights * loss_vector, weights))
        row.update(boot.report("recovery_rate_on_global_fail_pairs", weights * correct * (test[0]["raw_global_delta"] < 0),
                               weights * (test[0]["raw_global_delta"] < 0)))
        row.update(boot.report("harm_rate_on_global_correct_pairs", weights * ~correct * (test[0]["raw_global_delta"] > 0),
                               weights * (test[0]["raw_global_delta"] > 0)))
        rows.append(row)
        if kind == "global_plus_hidden":
            choices.append((val_loss, key))
    # Paper-established nonlinear calibration control; same raw global delta only.
    iso = IsotonicRegression(out_of_bounds="clip")
    x = fit[1]["global"]
    iso.fit(np.r_[x, -x], np.r_[np.ones(len(x)), np.zeros(len(x))],
            sample_weight=np.tile(fit[0]["weights"], 2)/2)
    iso_test = iso.predict(test[1]["global"] * test[0]["orientation"])
    iso_select = iso.predict(select[1]["global"] * select[0]["orientation"])
    iso_key = "__global__/global_isotonic"
    predictions[iso_key] = np.clip(iso_test, 1e-15, 1-1e-15)
    rows.append({"scorer": "__global__", "probe": "global_isotonic", "train_split": "fit",
                 "selection_log_loss": log_loss(select[0]["labels"], iso_select, sample_weight=select[0]["weights"]),
                 "roc_auc": roc_auc_score(y, iso_test, sample_weight=weights)})
    models[iso_key] = {"x_thresholds": iso.X_thresholds_.tolist(), "y_thresholds": iso.y_thresholds_.tolist()}
    best_key = min(choices)[1]
    global_key = min([r for r in rows if r["scorer"] == "__global__"],
                     key=lambda r: r["selection_log_loss"])
    global_key = "__global__/" + global_key["probe"]
    comparisons = []
    for row in rows:
        key = row["scorer"] + "/" + row["probe"]
        prob = predictions[key]
        losses = -(y * np.log(np.clip(prob, 1e-15, 1)) + (1-y)*np.log(np.clip(1-prob, 1e-15, 1)))
        row.update(boot.report("log_loss", weights * losses, weights))
        row.update(boot.report("pair_accuracy", weights * ((prob >= .5) == y), weights))
        ref = predictions[global_key]
        ref_loss = -(y*np.log(np.clip(ref, 1e-15, 1)) + (1-y)*np.log(np.clip(1-ref, 1e-15, 1)))
        row.update(boot.report("delta_log_loss_vs_global", weights * (losses-ref_loss), weights))
        row["comparison_global"] = global_key
        if row["probe"] == "global_plus_hidden":
            control = predictions[row["scorer"] + "/permuted_hidden_control"]
            control_loss = -(y*np.log(np.clip(control, 1e-15, 1)) + (1-y)*np.log(np.clip(1-control, 1e-15, 1)))
            row.update(boot.report("delta_log_loss_vs_permuted", weights*(losses-control_loss), weights))
        row["selected_on_validation"] = key == best_key
        row["bootstrap_repetitions"] = boot.repetitions
        row["bootstrap_confidence"] = boot.confidence
    # Exact AUC bootstrap for the preselected primary comparison and matched control.
    # Avoid thousands of sorts for every exploratory scorer: order is cached in auc().
    auc_keys = [best_key, global_key]
    best_scorer = best_key.split("/")[0]
    matched = best_scorer + "/permuted_hidden_control"
    if matched in predictions:
        auc_keys.append(matched)
    auc_results = {key: boot.auc(y, predictions[key], weights) for key in auc_keys}
    for ref_key in auc_keys[1:]:
        value, draws = auc_results[best_key]
        ref_value, ref_draws = auc_results[ref_key]
        low, high = boot.interval(draws-ref_draws)
        comparisons.append({"selected_probe": best_key, "reference": ref_key,
                            "delta_auc": value-ref_value, "delta_auc_ci_low": low,
                            "delta_auc_ci_high": high, "bootstrap_repetitions": boot.repetitions})
    for row in rows:
        key = row["scorer"] + "/" + row["probe"]
        if key in auc_results:
            row["roc_auc_ci_low"], row["roc_auc_ci_high"] = boot.interval(auc_results[key][1])
    write_csv(output / "probe_results.csv", rows)
    write_csv(output / "probe_primary_comparisons.csv", comparisons)
    write_json(output / "probe_models.json", models)
    np.savez_compressed(output / "probe_predictions.npz", pid=test[0]["pids"], labels=y,
                        weights=weights, raw_global_delta=test[0]["raw_global_delta"], **predictions)
    return fitted, rows, best_key


def evaluate_learned_setwise(oracle, fitted, primary, specs, counts, cfg, seed, output):
    # Score oracle candidates using heads learned on FIT, selected on SELECTION.
    # Keep this trained diagnostic separate from the parameter-free scorer table.
    selected = ["__global__/global_only", primary]
    selected.append(primary.split("/")[0] + "/hidden_only")
    positive, negative = dict(oracle.positive_local_scores), dict(oracle.negative_local_scores)
    extra_specs = []
    for key in selected:
        head, columns = fitted[key]
        pos = np.column_stack([(oracle.positive_global_scores if c == "global" else positive[c]).numpy() for c in columns])
        neg = np.stack([(oracle.negative_global_scores if c == "global" else negative[c]).numpy() for c in columns], axis=-1)
        name = "ranknet:" + key
        positive[name] = torch.from_numpy(head.score(pos)).float()
        negative[name] = torch.from_numpy(head.score(neg.reshape(-1,len(columns))).reshape(neg.shape[:2])).float()
        extra_specs.append(ScorerSpec(name, "linear_ranknet"))
    baseline_specs = [s for s in specs if s.kind == "maxsim"][:1]
    scored = replace(oracle, positive_local_scores=positive, negative_local_scores=negative)
    summary, queries = evaluate_setwise(scored, baseline_specs + extra_specs, counts, max(counts),
                                       cfg["bootstrap_repetitions"], cfg["bootstrap_confidence"], seed)
    boot = IdentityBootstrap(oracle.qids.numpy(), cfg["bootstrap_repetitions"], cfg["bootstrap_confidence"], seed)
    decisions = {}
    for row in summary:
        pos = positive[row["scorer"]]
        best = np.array([float(pos[a:b].max()) for a,b in zip(oracle.positive_offsets[:-1], oracle.positive_offsets[1:])])
        correct = best > negative[row["scorer"]][:,:row["num_hard_negatives"]].max(1).values.numpy()
        decisions[(row["scorer"], row["num_hard_negatives"])] = correct
        row.update(boot.report("setwise_accuracy", correct))
    for row in summary:
        row["trained_head"] = row["scorer"].startswith("ranknet:")
        row["selected_primary_probe"] = primary
        row["bootstrap_repetitions"] = cfg["bootstrap_repetitions"]
        key = (row["scorer"], row["num_hard_negatives"])
        ref = (row.get("comparison_scorer"), row["num_hard_negatives"])
        if ref in decisions:
            row.update(boot.report("delta_setwise_accuracy_vs_maxsim", decisions[key].astype(float)-decisions[ref]))
    write_csv(output / "setwise_learned_summary.csv", summary)
    write_csv(output / "setwise_learned_queries.csv", queries)
    np.savez_compressed(output / "setwise_learned_decisions.npz", pid=oracle.qids.numpy(),
                        global_correct=oracle.global_correct.numpy(),
                        **{name+"/"+str(m):v for (name,m),v in decisions.items()})
    return summary


def add_learned_scores(states, fitted):
    """Apply the same frozen linear head to each candidate; no candidate labels."""
    learned = []
    for key, (head, columns) in fitted.items():
        if key.endswith("/permuted_hidden_control"):
            continue
        name = "ranknet:" + key
        for state in states.values():
            values = [state.global_topk_scores if c == "global" else state.local_topk_scores[c]
                      for c in columns]
            items = torch.stack(values, dim=-1).numpy().astype(np.float64)
            state.local_topk_scores[name] = torch.from_numpy(head.score(items.reshape(-1, len(columns)))
                                                             .reshape(items.shape[:2])).float()
        learned.append(name)
    return learned


def variants(state, specs, ks, weights, learned):
    yield {"scorer": "__global__", "kind": "global", "k": 1,
           "mode": "noop", "fusion_weight": ""}, state.global_topk_scores[:, :1]
    for spec, k, mode, weight, scores in iter_variants(state, specs, ks, weights):
        yield {"scorer": spec.name, "kind": spec.kind, "k": k, "mode": mode,
               "fusion_weight": "" if weight is None else weight}, scores
    for name in learned:
        for k in ks:
            k = min(k, state.topk_indices.shape[1])
            yield {"scorer": name, "kind": "linear_ranknet", "k": k,
                   "mode": "learned", "fusion_weight": ""}, state.local_topk_scores[name][:, :k]


def scores_for(state, choice):
    if choice["mode"] == "noop":
        return state.global_topk_scores[:, :1]
    if choice["mode"] == "learned":
        return state.local_topk_scores[choice["scorer"]][:, :choice["k"]]
    return get_variant_scores(state, choice)


def predictions_for(state, choice):
    scores = scores_for(state, choice)
    pos = scores.argmax(1)
    pred = state.topk_indices.gather(1, pos[:, None]).flatten()
    return pred, state.global_features.gids[pred].eq(state.global_features.qids).numpy()


def query_metrics(state, choice, gate=None):
    """Exact per-query CMC/AP/INP, stable ties preserve the global prefix order."""
    scores = scores_for(state, choice)
    k = choice["k"]
    n = len(state.rankings)
    gate = torch.ones(n, dtype=torch.bool) if gate is None else torch.as_tensor(gate)
    result = {key: np.empty(n) for key in ["r1", "r5", "r10", "map", "minp"]}
    for start in range(0, n, 128):
        stop = min(n, start+128)
        ranks = state.rankings[start:stop].clone()
        order = torch.argsort(scores[start:stop], descending=True, stable=True, dim=1)
        prefix = ranks[:, :k].gather(1, order)
        ranks[:, :k] = torch.where(gate[start:stop, None], prefix, ranks[:, :k])
        matches = state.global_features.gids[ranks].eq(state.global_features.qids[start:stop, None])
        cumulative = matches.cumsum(1).double()
        positions = torch.arange(1, ranks.shape[1]+1).double()[None]
        relevant = matches.sum(1)
        for cutoff in [1, 5, 10]:
            result[f"r{cutoff}"][start:stop] = matches[:, :cutoff].any(1).numpy()
        result["map"][start:stop] = ((cumulative/positions*matches).sum(1)/relevant.clamp_min(1)).numpy()
        last = (positions * matches).max(1).values
        result["minp"][start:stop] = (relevant/last.clamp_min(1)).numpy()
    return result


def evaluated_row(state, choice, base_metrics, cfg, seed, gate=None):
    metrics = query_metrics(state, choice, gate)
    pids = state.global_features.qids.numpy()
    row = decision_intervals(pids, base_metrics["r1"], metrics["r1"],
                             cfg["bootstrap_repetitions"], cfg["bootstrap_confidence"], seed)
    boot = IdentityBootstrap(pids, cfg["bootstrap_repetitions"], cfg["bootstrap_confidence"], seed)
    for name in ["map", "r5", "r10", "minp"]:
        row.update(boot.report(name, metrics[name], scale=100))
        row.update(boot.report("delta_"+name, metrics[name]-base_metrics[name], scale=100))
    return {**choice, **row}, metrics


def selection_key(row):
    return (row["net_rescued"], -row["harmed"], row["mode"] == "noop", -row["k"])


def evaluate_retrieval(states, specs, ks, weights, learned, cfg, seed, output):
    selection, test = states["selection"], states["test"]
    grid = [{**choice, **top1_variant_summary(selection, scores, choice["k"])}
            for choice, scores in variants(selection, specs, ks, weights, learned)]
    winner = max(grid, key=selection_key)
    baseline = next(row for row in grid if row["mode"] == "noop")
    # Predeclare a MaxSim reference whose own settings are selected without test labels.
    mx = [s.name for s in specs if s.kind == "maxsim"]
    maxsim_candidates = [r for r in grid if r["scorer"] in mx or r["mode"] == "noop"]
    selections = [("global_baseline", baseline), ("overall", winner),
                  ("maxsim_reference", max(maxsim_candidates, key=selection_key))]
    base_metrics = query_metrics(test, baseline)
    records, arrays, metrics_by_role = [], {}, {}
    for role, chosen in selections:
        row, metrics = evaluated_row(test, chosen, base_metrics, cfg, seed)
        row["selected_as"] = role
        row["selection_r1"] = chosen["r1"]
        records.append(row)
        metrics_by_role[role] = metrics
        for key, value in metrics.items():
            arrays[role + "/" + key] = value
    boot = IdentityBootstrap(test.global_features.qids.numpy(), cfg["bootstrap_repetitions"], cfg["bootstrap_confidence"], seed)
    for key in ["r1", "map"]:
        records[1].update(boot.report("delta_"+key+"_vs_maxsim",
                         metrics_by_role["overall"][key]-metrics_by_role["maxsim_reference"][key], scale=100))
    write_csv(output / "reranking_validation_grid.csv", grid)
    write_csv(output / "reranking_selected_test.csv", records)
    np.savez_compressed(output / "reranking_query_metrics.npz", pid=test.global_features.qids.numpy(), **arrays)
    return grid, winner, baseline, base_metrics, records


def gate_features(state, choice):
    scores = scores_for(state, choice).numpy()
    glob = state.global_topk_scores.numpy()
    pos = scores.argmax(1)
    rows = np.arange(len(scores))
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1]-sorted_scores[:, -2] if scores.shape[1] > 1 else np.zeros(len(scores))
    return np.column_stack([glob[:, 0]-glob[:, 1], margin,
                            scores[rows, pos]-scores[:, 0], glob[:, 0]-glob[rows, pos]])


def calibration_bound(pids, base_correct, harmed, delta):
    """Conservative cluster bound: any harmed query / identity with any correct query.

    This upper-bounds identity-macro conditional harm, NOT query-micro harm.
    Clopper-Pearson is valid under independent/exchangeable identity clusters
    conditional on the fitted system and fixed gallery. Zero errors != zero risk.
    """
    ids = np.unique(pids)
    eligible = [pid for pid in ids if np.any(base_correct[pids == pid])]
    failures = sum(np.any(harmed[pids == pid]) for pid in eligible)
    n = len(eligible)
    bound = 1.0 if n == 0 or failures == n else float(beta.ppf(1-delta, failures+1, n-failures))
    return bound, n, failures


def evaluate_gates(states, grid, baseline, base_metrics, cfg, seed, output):
    # One candidate per budget preselected on SELECTION, then a single independent
    # CALIBRATION test per budget; Bonferroni across budgets. Never retry on calibration.
    budgets = cfg.get("gating_harm_budgets", [.01, .02, .05])
    # Restrict gate training to one deployable candidate chosen on selection.
    nontrivial = [r for r in grid if r["mode"] != "noop"]
    choice = max(nontrivial, key=selection_key) if nontrivial else baseline
    predictions = {name: predictions_for(state, choice) for name, state in states.items()}
    base = {name: state.global_features.gids[state.rankings[:, 0]].eq(state.global_features.qids).numpy()
            for name, state in states.items()}
    disagreements = {name: pred[0].ne(states[name].rankings[:, 0]).numpy()
                     for name, pred in predictions.items()}
    features = {name: gate_features(state, choice) for name, state in states.items()}
    fit_mask = disagreements["fit"]
    y = predictions["fit"][1][fit_mask].astype(int)
    gate_model = None
    if len(np.unique(y)) == 2:
        scaler = StandardScaler().fit(features["fit"][fit_mask])
        gate_model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(features["fit"][fit_mask]), y)
        learned_conf = {name: gate_model.predict_proba(scaler.transform(x))[:, 1]
                        for name, x in features.items()}
        write_json(output / "gate_model.json", {"feature_names": ["global_margin", "reranker_margin", "local_advantage", "global_cost"],
                   "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                   "coefficients": gate_model.coef_.tolist(), "intercept": gate_model.intercept_.tolist()})
    else:
        learned_conf = None
    sources = {"global_uncertainty": {name: -x[:, 0] for name, x in features.items()}}
    if learned_conf is not None:
        sources["learned_correctness"] = learned_conf
    proposals = []
    for source, scores in sources.items():
        values = scores["selection"][disagreements["selection"]]
        thresholds = np.unique(np.quantile(values, np.linspace(0, 1, int(cfg.get("gating_threshold_steps", 41))))) if len(values) else []
        for threshold in thresholds:
            gate = disagreements["selection"] & (scores["selection"] >= threshold)
            final = np.where(gate, predictions["selection"][1], base["selection"])
            rescued, harmed = ~base["selection"] & final, base["selection"] & ~final
            proposals.append({"source": source, "threshold": float(threshold),
                               "rescued": int(rescued.sum()), "harmed": int(harmed.sum()),
                               "net": int(rescued.sum())-int(harmed.sum()),
                               "harm": float(harmed.sum()/max(1, base["selection"].sum()))})
    rows, arrays = [], {}
    delta = float(cfg.get("calibration_delta", .05)) / max(1, len(budgets))
    for budget in budgets:
        viable = [p for p in proposals if p["harm"] <= budget and p["net"] > 0]
        proposal = max(viable, key=lambda p: (p["net"], -p["harmed"], p["rescued"])) if viable else None
        if proposal:
            conf = sources[proposal["source"]]
            gates = {name: disagreements[name] & (conf[name] >= proposal["threshold"]) for name in states}
            harmed = base["calibration"] & ~np.where(gates["calibration"], predictions["calibration"][1], base["calibration"])
            bound, n, failures = calibration_bound(states["calibration"].global_features.qids.numpy(), base["calibration"], harmed, delta)
            accepted = bound <= budget
        else:
            gates = {name: np.zeros(len(state.rankings), bool) for name, state in states.items()}
            bound, n, failures, accepted = 0., 0, 0, True
        for policy in ["empirical", "risk_screened"]:
            gate = gates["test"] if policy == "empirical" or accepted else np.zeros(len(states["test"].rankings), bool)
            row, metrics = evaluated_row(states["test"], choice, base_metrics, cfg, seed, gate)
            pids = states["test"].global_features.qids.numpy()
            identity_rates = [np.mean((~metrics["r1"].astype(bool))[ (pids==pid) & base["test"] ])
                              for pid in np.unique(pids) if np.any((pids==pid) & base["test"])]
            row.update(split="test", policy=policy, harm_budget=budget,
                       gate_source=proposal["source"] if proposal else "noop",
                       threshold=proposal["threshold"] if proposal else None,
                       calibration_accepted=accepted, calibration_upper_bound=bound,
                       calibration_eligible_ids=n, calibration_harmed_ids=failures,
                       calibration_delta_per_budget=delta, controlled_risk="identity_macro_conditional_harm",
                       test_identity_macro_harm=float(np.mean(identity_rates)) if identity_rates else None,
                       test_micro_budget_exceeded=(row["harm_rate"] > budget) if np.isfinite(row["harm_rate"]) else None,
                       selection_net=proposal["net"] if proposal else 0,
                       gate_coverage=float(gate.mean()), gated_queries=int(gate.sum()),
                       fallback_noop=not bool(gate.any()))
            rows.append(row)
            key = policy + "/" + str(budget)
            arrays[key + "/gate"] = gate
            for name, values in metrics.items():
                arrays[key + "/" + name] = values
    write_csv(output / "gating_summary.csv", rows)
    write_csv(output / "gating_selection_grid.csv", proposals)
    np.savez_compressed(output / "gating_query_metrics.npz", pid=states["test"].global_features.qids.numpy(), **arrays)
    return rows


def run_suite_v2(config, run_dir, specs, validation, test, device, pair_batch_size, existing_test_rows=()):
    cfg = {"bootstrap_repetitions": 2000, "bootstrap_confidence": .95, **dict(config.get("validated") or {})}
    seed = int(config.get("seed", 42))
    output = Path(run_dir) / "validated_v2"
    output.mkdir(parents=True, exist_ok=False)
    log = logging.getLogger("ex-of-ex.v2")
    states, source = prepare_splits(validation, test, cfg, seed)
    assignments = [{"split": name, "pid": int(pid), "queries": int((state.global_features.qids==int(pid)).sum())}
                   for name, state in states.items() for pid in torch.unique(state.global_features.qids).tolist()]
    write_csv(output / "split_assignments.csv", assignments)
    protocol = {"version": 2, "source": source, "seed": seed, "encoder": "frozen",
                "split_unit": "person_identity", "fit_selection_calibration_disjoint": True,
                "bootstrap_repetitions": cfg["bootstrap_repetitions"], "bootstrap_confidence": cfg["bootstrap_confidence"],
                "bootstrap_unit": "identity_cluster", "bootstrap_scope": "fixed model/selection/gallery; query uncertainty only",
                "setwise": "oracle all positives; no positive injection in retrieval",
                "test_status": "exploratory follow-up; test results from v1 have been inspected",
                "risk_screen": "LTT-inspired split testing, Clopper-Pearson bound on any-harm identity events; Bonferroni across budgets",
                "risk_limit": "identity-macro harm only; not a guarantee on test micro harm or distribution shift"}
    protocol["method_sources"] = {
        "linear_ranknet": "https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/",
        "isotonic_calibration": "https://doi.org/10.1145/775047.775151",
        "selective_classification": "https://arxiv.org/abs/1705.08500",
        "learn_then_test_inspiration": "https://arxiv.org/abs/2110.01052",
        "bootstrap": "https://blogs.helsinki.fi/bk-club/files/2012/05/Efron_Bootstrap_AS1979.pdf",
    }
    protocol["implementation_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ["protocol_v2.py", "uncertainty.py", "validated.py", "scorers.py", "evaluation.py"]
    }
    write_json(output / "protocol.json", protocol)
    ks = sorted(set(int(k) for k in config.get("topk", [10, 20, 50])))
    counts = cfg.get("setwise_negative_counts", [1, 5, 10])
    max_neg = max(max(counts), int(cfg.get("probe_negatives", 10)))
    log.info("v2: scoring oracle packs; protocol=%s", source)
    oracles = {name: build_oracle_set_scores(state, specs, max_neg, device, pair_batch_size)
               for name, state in states.items() if name != "calibration"}
    log.info("v2 experiment 1: setwise + %d identity bootstrap draws", cfg["bootstrap_repetitions"])
    sw, queries = evaluate_setwise(oracles["test"], specs, counts, max(counts),
                                  cfg["bootstrap_repetitions"], cfg["bootstrap_confidence"], seed)
    # Extra CI: setwise accuracy and its paired difference from MaxSim at every M.
    oracle = oracles["test"]
    boot = IdentityBootstrap(oracle.qids.numpy(), cfg["bootstrap_repetitions"], cfg["bootstrap_confidence"], seed)
    correct_by_key = {}
    for row in sw:
        pos = oracle.positive_local_scores[row["scorer"]]
        best = np.array([float(pos[a:b].max()) for a,b in zip(oracle.positive_offsets[:-1], oracle.positive_offsets[1:])])
        correct = best > oracle.negative_local_scores[row["scorer"]][:, :row["num_hard_negatives"]].max(1).values.numpy()
        correct_by_key[(row["scorer"], row["num_hard_negatives"])] = correct
        row.update(boot.report("setwise_accuracy", correct))
        row.update(bootstrap_repetitions=cfg["bootstrap_repetitions"], bootstrap_confidence=cfg["bootstrap_confidence"])
    for row in sw:
        key = (row["scorer"], row["num_hard_negatives"])
        reference = (row.get("comparison_scorer"), row["num_hard_negatives"])
        if reference in correct_by_key:
            row.update(boot.report("delta_setwise_accuracy_vs_maxsim", correct_by_key[key].astype(float)-correct_by_key[reference]))
    write_csv(output / "setwise_summary.csv", sw)
    write_csv(output / "setwise_query_results.csv", queries)
    np.savez_compressed(output / "setwise_decisions.npz", pid=oracle.qids.numpy(), global_correct=oracle.global_correct.numpy(),
                        **{name+"/"+str(m): v for (name,m),v in correct_by_key.items()})
    coverage = [r for state in states.values() for r in evaluate_coverage(state, ks)]
    write_csv(output / "coverage.csv", coverage)
    log.info("v2 experiment 2: antisymmetric probes, isotonic global control, selected AUC bootstrap")
    fitted, probes, primary = fit_probes(oracles, specs, cfg, seed, output)
    learned_setwise = evaluate_learned_setwise(oracles["test"], fitted, primary, specs, counts, cfg, seed, output)
    learned = add_learned_scores(states, fitted)
    log.info("v2 experiment 3: selecting frozen RankNet/fusion heads including no-op")
    grid, winner, baseline, base_metrics, rerank = evaluate_retrieval(states, specs, ks,
        config.get("fusion_weights", [.1,.2,.3,.5]), learned, cfg, seed, output)
    log.info("v2 experiment 4: independent calibration screen and empirical gate ablation")
    gates = evaluate_gates(states, grid, baseline, base_metrics, cfg, seed, output)
    summary = {"protocol": protocol, "setwise": sw, "probe": probes, "primary_probe": primary,
               "setwise_learned": learned_setwise,
               "reranking_test": rerank, "gating": gates, "coverage": coverage,
               "outputs": {"validated_dir": str(output)}}
    write_json(output / "summary.json", summary)
    log.info("v2 complete: %s", output)
    return summary
