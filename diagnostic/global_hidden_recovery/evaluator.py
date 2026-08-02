from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def compute_global_similarity(qfeats: torch.Tensor, gfeats: torch.Tensor) -> torch.Tensor:
    qnorm = F.normalize(qfeats.float(), p=2, dim=1)
    gnorm = F.normalize(gfeats.float(), p=2, dim=1)
    return qnorm @ gnorm.t()


def compute_retrieval_metrics(
    similarity: torch.Tensor,
    qids: torch.Tensor,
    gids: torch.Tensor,
) -> Dict[str, float]:
    cmc, m_ap, m_inp, _ = rank_like_official(
        similarity=similarity,
        q_pids=qids.cpu(),
        g_pids=gids.cpu(),
        max_rank=10,
        get_mAP=True,
    )
    return {
        "r1": float(cmc[0].item()),
        "r5": float(cmc[4].item()),
        "r10": float(cmc[9].item()),
        "map": float(m_ap.item()),
        "minp": float(m_inp.item()),
    }


def rank_like_official(
    similarity: torch.Tensor,
    q_pids: torch.Tensor,
    g_pids: torch.Tensor,
    max_rank: int = 10,
    get_mAP: bool = True,
):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )

    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)

    inp = []
    for i, match_row in enumerate(matches):
        hits = match_row.nonzero(as_tuple=False).view(-1)
        if hits.numel() == 0:
            continue
        last_hit = hits[-1]
        inp.append(tmp_cmc[i][last_hit] / (last_hit + 1.0))
    mINP = torch.stack(inp).mean() * 100 if inp else torch.tensor(0.0)

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel.clamp_min(1)
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def compare_metrics(
    official: Mapping[str, float],
    diagnostic: Mapping[str, float],
    tolerance: float,
) -> Dict[str, float]:
    diffs = {}
    for key in ("r1", "r5", "r10", "map"):
        diffs[key] = abs(float(official[key]) - float(diagnostic[key]))
    diffs["max_abs"] = max(diffs.values()) if diffs else 0.0
    diffs["passed"] = bool(diffs["max_abs"] <= tolerance)
    return diffs


def build_global_decisions(
    similarity: torch.Tensor,
    qids: torch.Tensor,
    gids: torch.Tensor,
    topk_values: Sequence[int],
) -> Dict[str, torch.Tensor]:
    max_topk = int(max(topk_values))
    if max_topk > similarity.shape[1]:
        max_topk = int(similarity.shape[1])

    full_rankings = torch.argsort(similarity, dim=1, descending=True)
    topk_indices = full_rankings[:, :max_topk].contiguous()

    qids = qids.cpu().long()
    gids = gids.cpu().long()
    sim = similarity.cpu().float()
    n_query = sim.shape[0]

    best_pos = torch.full((n_query,), -1, dtype=torch.long)
    best_neg = torch.full((n_query,), -1, dtype=torch.long)
    best_pos_score = torch.full((n_query,), float("nan"), dtype=torch.float32)
    best_neg_score = torch.full((n_query,), float("nan"), dtype=torch.float32)
    best_pos_rank = torch.full((n_query,), -1, dtype=torch.long)
    global_top1_pid = torch.full((n_query,), -1, dtype=torch.long)
    positive_in_topk = {int(k): torch.zeros(n_query, dtype=torch.bool) for k in topk_values}

    for qi in range(n_query):
        pid = qids[qi]
        pos_mask = gids.eq(pid)
        neg_mask = ~pos_mask
        row = sim[qi]

        pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
        neg_idx = torch.nonzero(neg_mask, as_tuple=False).view(-1)
        if pos_idx.numel() > 0:
            local = torch.argmax(row.index_select(0, pos_idx))
            best_pos[qi] = pos_idx[local]
            best_pos_score[qi] = row[best_pos[qi]]
        if neg_idx.numel() > 0:
            local = torch.argmax(row.index_select(0, neg_idx))
            best_neg[qi] = neg_idx[local]
            best_neg_score[qi] = row[best_neg[qi]]

        ranked = full_rankings[qi]
        global_top1_pid[qi] = gids[ranked[0]]
        pos_rank_hits = torch.nonzero(gids.index_select(0, ranked).eq(pid), as_tuple=False).view(-1)
        if pos_rank_hits.numel() > 0:
            best_pos_rank[qi] = pos_rank_hits[0] + 1

        for k in topk_values:
            kk = min(int(k), ranked.numel())
            positive_in_topk[int(k)][qi] = gids.index_select(0, ranked[:kk]).eq(pid).any()

    return {
        "topk_indices": topk_indices,
        "best_positive_index": best_pos,
        "best_negative_index": best_neg,
        "global_best_positive_score": best_pos_score,
        "global_best_negative_score": best_neg_score,
        "delta_global": best_pos_score - best_neg_score,
        "global_rank_best_positive": best_pos_rank,
        "global_top1_pid": global_top1_pid,
        "positive_in_topk": positive_in_topk,
    }


def assign_categories(delta_global: torch.Tensor, delta_hidden: torch.Tensor) -> List[str]:
    categories = []
    for dg, dh in zip(delta_global.tolist(), delta_hidden.tolist()):
        if dg > 0 and dh > 0:
            categories.append("A")
        elif dg < 0 and dh > 0:
            categories.append("B_hidden_recoverable_global_failure")
        elif dg < 0 and dh < 0:
            categories.append("C")
        elif dg > 0 and dh < 0:
            categories.append("D_harm")
        else:
            categories.append("tie")
    return categories


def summarize_fixed_pair(delta_global: np.ndarray, delta_hidden: np.ndarray) -> Dict[str, object]:
    global_fail = delta_global < 0
    global_correct = delta_global > 0
    hidden_correct = delta_hidden > 0
    hidden_wrong = delta_hidden < 0

    category_a = global_correct & hidden_correct
    category_b = global_fail & hidden_correct
    category_c = global_fail & hidden_wrong
    category_d = global_correct & hidden_wrong

    recovery_den = int(global_fail.sum())
    harm_den = int(global_correct.sum())

    return {
        "global_failures": recovery_den,
        "category_A": int(category_a.sum()),
        "category_B_recoverable": int(category_b.sum()),
        "category_C": int(category_c.sum()),
        "category_D_harm": int(category_d.sum()),
        "category_tie": int(
            len(delta_global)
            - category_a.sum()
            - category_b.sum()
            - category_c.sum()
            - category_d.sum()
        ),
        "recovery_rate": safe_rate(category_b.sum(), recovery_den),
        "harm_rate": safe_rate(category_d.sum(), harm_den),
        "margin_stats": margin_stats(delta_global, delta_hidden, {
            "A": category_a,
            "B_hidden_recoverable_global_failure": category_b,
            "C": category_c,
            "D_harm": category_d,
        }),
    }


def summarize_topk(
    topk_values: Sequence[int],
    topk_indices: torch.Tensor,
    hidden_topk_scores: torch.Tensor,
    qids: torch.Tensor,
    gids: torch.Tensor,
    delta_global: np.ndarray,
    positive_in_topk: Mapping[int, torch.Tensor],
) -> Dict[str, object]:
    out = {}
    qids = qids.cpu().long()
    gids = gids.cpu().long()
    global_wrong = torch.as_tensor(delta_global < 0)

    for k in topk_values:
        k = int(k)
        kk = min(k, topk_indices.shape[1])
        scores_k = hidden_topk_scores[:, :kk]
        best_local = torch.argmax(scores_k, dim=1)
        best_gallery = topk_indices[:, :kk].gather(1, best_local.view(-1, 1)).view(-1)
        hidden_correct = gids.index_select(0, best_gallery).eq(qids)

        pos_in = positive_in_topk[k].cpu().bool()
        denom_mask = global_wrong & pos_in
        rescue_mask = denom_mask & hidden_correct

        out[str(k)] = {
            "num_global_failures": int(global_wrong.sum().item()),
            "positive_in_topk_given_global_failure": safe_rate(
                int((global_wrong & pos_in).sum().item()),
                int(global_wrong.sum().item()),
            ),
            "num_global_failures_with_positive_in_topk": int(denom_mask.sum().item()),
            "num_rescued": int(rescue_mask.sum().item()),
            "rescue_at_k": safe_rate(int(rescue_mask.sum().item()), int(denom_mask.sum().item())),
            "hidden_rerank_topk_correct_rate_all_queries": safe_rate(
                int(hidden_correct.sum().item()),
                int(hidden_correct.numel()),
            ),
            "hidden_correct": hidden_correct.cpu(),
        }

    return out


def summarize_oracle(delta_global: np.ndarray, delta_hidden: np.ndarray) -> Dict[str, float]:
    global_correct = delta_global > 0
    hidden_correct = delta_hidden > 0
    oracle = global_correct | hidden_correct
    global_rate = float(global_correct.mean()) if len(delta_global) else 0.0
    hidden_rate = float(hidden_correct.mean()) if len(delta_hidden) else 0.0
    oracle_rate = float(oracle.mean()) if len(delta_global) else 0.0
    return {
        "global_correct_rate": global_rate,
        "hidden_correct_rate": hidden_rate,
        "oracle_correct_rate": oracle_rate,
        "complementarity_gap": oracle_rate - max(global_rate, hidden_rate),
    }


def margin_stats(
    delta_global: np.ndarray,
    delta_hidden: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    stats = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        stats[name] = {
            "count": int(mask.sum()),
            "delta_global": robust_stats(delta_global[mask]),
            "delta_hidden": robust_stats(delta_hidden[mask]),
        }
    return stats


def robust_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"median": float("nan"), "iqr_low": float("nan"), "iqr_high": float("nan")}
    return {
        "median": float(np.median(values)),
        "iqr_low": float(np.percentile(values, 25)),
        "iqr_high": float(np.percentile(values, 75)),
    }


def safe_rate(num, den) -> float:
    den = int(den)
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def write_query_csv(
    path: Path,
    qids: torch.Tensor,
    captions: Sequence[str],
    decisions: Mapping[str, object],
    fixed_positive_hidden: torch.Tensor,
    fixed_negative_hidden: torch.Tensor,
    categories: Sequence[str],
    topk_summary: Mapping[str, object],
    topk_values: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    delta_hidden = fixed_positive_hidden - fixed_negative_hidden
    fieldnames = [
        "query_id",
        "pid",
        "caption",
        "global_rank_best_positive",
        "global_top1_pid",
        "global_best_positive_score",
        "global_best_negative_score",
        "delta_global",
        "hidden_fixed_positive_score",
        "hidden_fixed_negative_score",
        "delta_hidden",
        "global_pair_correct",
        "hidden_pair_correct",
        "category",
    ]
    for k in topk_values:
        fieldnames.append(f"positive_in_top{k}")
    for k in topk_values:
        fieldnames.append(f"hidden_rerank_top{k}_correct")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(qids)):
            row = {
                "query_id": i,
                "pid": int(qids[i].item()),
                "caption": captions[i] if i < len(captions) else "",
                "global_rank_best_positive": int(decisions["global_rank_best_positive"][i].item()),
                "global_top1_pid": int(decisions["global_top1_pid"][i].item()),
                "global_best_positive_score": float(decisions["global_best_positive_score"][i].item()),
                "global_best_negative_score": float(decisions["global_best_negative_score"][i].item()),
                "delta_global": float(decisions["delta_global"][i].item()),
                "hidden_fixed_positive_score": float(fixed_positive_hidden[i].item()),
                "hidden_fixed_negative_score": float(fixed_negative_hidden[i].item()),
                "delta_hidden": float(delta_hidden[i].item()),
                "global_pair_correct": bool(decisions["delta_global"][i].item() > 0),
                "hidden_pair_correct": bool(delta_hidden[i].item() > 0),
                "category": categories[i],
            }
            for k in topk_values:
                row[f"positive_in_top{k}"] = bool(decisions["positive_in_topk"][int(k)][i].item())
            for k in topk_values:
                hidden_correct = topk_summary[str(int(k))]["hidden_correct"]
                row[f"hidden_rerank_top{k}_correct"] = bool(hidden_correct[i].item())
            writer.writerow(row)


def write_summary_json(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
