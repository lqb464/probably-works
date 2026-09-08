"""Ranking and recovery metrics for the scorer sweep."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch


def metrics_from_rankings(
    rankings: torch.Tensor,
    qids: torch.Tensor,
    gids: torch.Tensor,
) -> Dict[str, float]:
    """Compute text-to-image R@1/5/10, mAP and mINP from ranked indices."""
    labels = gids.cpu().long().index_select(0, rankings.cpu().reshape(-1)).view_as(rankings)
    matches = labels.eq(qids.cpu().long().view(-1, 1))
    num_gallery = rankings.shape[1]

    cmc = matches.cumsum(dim=1).clamp_max(1).float().mean(dim=0) * 100.0
    precisions = matches.cumsum(dim=1).float()
    ranks = torch.arange(1, num_gallery + 1, dtype=torch.float32).view(1, -1)
    average_precision = ((precisions / ranks) * matches).sum(dim=1) / matches.sum(dim=1).clamp_min(1)

    inp_values = []
    for row, cumulative in zip(matches, precisions):
        hits = row.nonzero(as_tuple=False).flatten()
        if hits.numel():
            last = int(hits[-1].item())
            inp_values.append(cumulative[last] / float(last + 1))
    minp = torch.stack(inp_values).mean() * 100.0 if inp_values else torch.tensor(0.0)

    def recall_at(k: int) -> float:
        return float(cmc[min(k, num_gallery) - 1].item())

    return {
        "r1": recall_at(1),
        "r5": recall_at(5),
        "r10": recall_at(10),
        "map": float(average_precision.mean().item() * 100.0),
        "minp": float(minp.item()),
    }


def rerank_prefix(
    global_rankings: torch.Tensor,
    candidate_scores: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Reorder only the global top-k prefix and leave its tail unchanged."""
    kk = min(int(k), global_rankings.shape[1], candidate_scores.shape[1])
    order = torch.argsort(candidate_scores[:, :kk], dim=1, descending=True)
    reranked = global_rankings.clone()
    reranked[:, :kk] = global_rankings[:, :kk].gather(1, order)
    return reranked


def querywise_zscore(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (values - mean) / std


def candidate_score_variants(
    local_scores: torch.Tensor,
    global_scores: torch.Tensor,
    fusion_weights: Iterable[float],
) -> list[Tuple[str, float | None, torch.Tensor]]:
    """Return local-only and query-normalized global/local fusion variants."""
    variants: list[Tuple[str, float | None, torch.Tensor]] = [
        ("local", None, local_scores)
    ]
    local_z = querywise_zscore(local_scores)
    global_z = querywise_zscore(global_scores)
    for weight in fusion_weights:
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"fusion weight must be in [0, 1], got {weight}")
        variants.append(
            (f"fusion_w{weight:g}", weight, (1.0 - weight) * global_z + weight * local_z)
        )
    return variants


def decision_summary(
    rankings: torch.Tensor,
    global_rankings: torch.Tensor,
    qids: torch.Tensor,
    gids: torch.Tensor,
    positive_in_topk: torch.Tensor,
) -> Tuple[Dict[str, object], torch.Tensor]:
    qids = qids.cpu().long()
    gids = gids.cpu().long()
    predicted = rankings[:, 0].cpu().long()
    baseline = global_rankings[:, 0].cpu().long()
    correct = gids.index_select(0, predicted).eq(qids)
    global_correct = gids.index_select(0, baseline).eq(qids)
    rescued = (~global_correct) & correct
    harmed = global_correct & (~correct)
    eligible = (~global_correct) & positive_in_topk.cpu().bool()

    metrics = metrics_from_rankings(rankings, qids, gids)
    metrics.update({
        "rescued": int(rescued.sum().item()),
        "harmed": int(harmed.sum().item()),
        "net_rescued": int(rescued.sum().item() - harmed.sum().item()),
        "rescue_rate_among_eligible": _rate(rescued & eligible, eligible),
        "harm_rate_among_global_correct": _rate(harmed, global_correct),
        "proper_oracle_r1": float((global_correct | correct).float().mean().item() * 100.0),
    })
    return metrics, correct


def fixed_pair_summary(
    delta_global: torch.Tensor,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> Dict[str, object]:
    delta_global = delta_global.cpu().float()
    delta_local = positive_scores.cpu().float() - negative_scores.cpu().float()
    global_correct = delta_global > 0
    local_correct = delta_local > 0
    recovered = (~global_correct) & local_correct
    harmed = global_correct & (~local_correct)
    return {
        "global_failures": int((~global_correct).sum().item()),
        "recovered": int(recovered.sum().item()),
        "harmed": int(harmed.sum().item()),
        "recovery_rate": _rate(recovered, ~global_correct),
        "harm_rate": _rate(harmed, global_correct),
        "local_pair_accuracy": float(local_correct.float().mean().item() * 100.0),
        "delta_local_mean": float(delta_local.mean().item()),
        "delta_local_median": float(delta_local.median().item()),
    }


def positive_in_prefix(
    global_rankings: torch.Tensor,
    qids: torch.Tensor,
    gids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    kk = min(int(k), global_rankings.shape[1])
    prefix = global_rankings[:, :kk].cpu().long()
    labels = gids.cpu().long().index_select(0, prefix.reshape(-1)).view_as(prefix)
    return labels.eq(qids.cpu().long().view(-1, 1)).any(dim=1)


def _rate(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    den = int(denominator.sum().item())
    return 0.0 if den == 0 else float(numerator.sum().item()) / float(den)
