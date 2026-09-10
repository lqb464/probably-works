"""Paired identity-cluster bootstrap, conditional on a fixed gallery/model.

Percentile intervals are descriptive, not simultaneous or post-selection bounds.
Undefined ratios stay NaN; they are never silently reported as zero uncertainty.
"""
from __future__ import annotations

import numpy as np


def grouped_bootstrap_ci(metric_fn, query_pids, repetitions=2000, confidence=.95, seed=42):
    pids = np.asarray(query_pids)
    ids = np.unique(pids)
    if len(ids) < 2 or repetitions < 2:
        return [float("nan"), float("nan")]
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    groups = [np.flatnonzero(pids == pid) for pid in ids]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        indices = np.concatenate([groups[i] for i in rng.integers(len(ids), size=len(ids))])
        draws.append(metric_fn(indices))
    valid = np.asarray(draws)[np.isfinite(draws)]
    if len(valid) < .9 * repetitions:
        return [float("nan"), float("nan")]
    alpha = (1-confidence)/2
    return np.quantile(valid, [alpha, 1-alpha]).tolist()


class IdentityBootstrap:
    def __init__(self, pids, repetitions=2000, confidence=0.95, seed=42):
        self.pids = np.asarray(pids)
        self.ids, self.inverse = np.unique(self.pids, return_inverse=True)
        if len(self.ids) < 2 or repetitions < 2 or not 0 < confidence < 1:
            raise ValueError("Bootstrap needs >=2 identities, >=2 repetitions and 0<CI<1")
        self.confidence = confidence
        self.repetitions = int(repetitions)
        rng = np.random.default_rng(seed)
        self.counts = rng.multinomial(len(self.ids), np.full(len(self.ids), 1 / len(self.ids)),
                                      size=self.repetitions).astype(np.float64)

    def sums(self, values):
        return np.bincount(self.inverse, weights=np.asarray(values, dtype=float),
                           minlength=len(self.ids))

    def ratio(self, numerator, denominator):
        a, b = self.sums(numerator), self.sums(denominator)
        boot_den = self.counts @ b
        samples = np.divide(self.counts @ a, boot_den,
                            out=np.full(self.repetitions, np.nan), where=boot_den > 0)
        return (float(a.sum() / b.sum()) if b.sum() else float("nan")), samples

    def interval(self, samples):
        valid = np.asarray(samples)[np.isfinite(samples)]
        if len(valid) < 0.9 * self.repetitions:
            return float("nan"), float("nan")
        alpha = (1 - self.confidence) / 2
        return tuple(np.quantile(valid, [alpha, 1 - alpha]).tolist())

    def report(self, name, numerator, denominator=None, scale=1.0):
        if denominator is None:
            denominator = np.ones(len(self.pids))
        point, samples = self.ratio(numerator, denominator)
        low, high = self.interval(samples * scale)
        return {name: point * scale, name + "_ci_low": low, name + "_ci_high": high,
                name + "_bootstrap_valid": int(np.isfinite(samples).sum())}

    def auc(self, labels, scores, weights=None):
        """Exact weighted AUC per draw; sorting/tie groups computed only once."""
        labels, scores = np.asarray(labels), np.asarray(scores)
        weights = np.ones(len(labels)) if weights is None else np.asarray(weights)
        order = np.argsort(scores, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
        inv = self.inverse[order]
        pos = weights[order] * labels[order]
        neg = weights[order] * (1 - labels[order])

        def value(multiplicity):
            pp = np.add.reduceat(pos * multiplicity, starts)
            nn = np.add.reduceat(neg * multiplicity, starts)
            den = pp.sum() * nn.sum()
            return float(np.sum(pp * (np.cumsum(nn) - 0.5 * nn)) / den) if den else np.nan

        point = value(np.ones(len(order)))
        draws = np.array([value(count[inv]) for count in self.counts])
        return point, draws


def decision_intervals(pids, baseline_correct, method_correct, repetitions=2000,
                       confidence=0.95, seed=42):
    boot = IdentityBootstrap(pids, repetitions, confidence, seed)
    base, new = np.asarray(baseline_correct, bool), np.asarray(method_correct, bool)
    result = {}
    for name, numerator, denominator, scale in [
        ("r1", new, np.ones(len(base)), 100),
        ("delta_r1", new.astype(float) - base, np.ones(len(base)), 100),
        ("recovery_rate", ~base & new, ~base, 1),
        ("harm_rate", base & ~new, base, 1),
    ]:
        result.update(boot.report(name, numerator, denominator, scale))
    result.update(rescued=int((~base & new).sum()), harmed=int((base & ~new).sum()),
                  net_rescued=int(new.sum()) - int(base.sum()),
                  num_queries=len(base), num_identities=len(boot.ids),
                  bootstrap_repetitions=repetitions, bootstrap_confidence=confidence)
    return result
