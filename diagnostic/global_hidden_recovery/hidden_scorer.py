from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class HiddenScorer:
    """Parameter-free text-to-image MaxSim scorer."""

    def __init__(self, device: torch.device):
        self.device = device

    def score_batch(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        image_features: torch.Tensor,
        return_details: bool = False,
    ):
        text = text_features.to(self.device, non_blocking=True).float()
        mask = text_mask.to(self.device, non_blocking=True).bool()
        image = image_features.to(self.device, non_blocking=True).float()

        sims = torch.bmm(text, image.transpose(1, 2))
        token_max, argmax_patch = sims.max(dim=-1)
        token_max = token_max.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=-1).clamp_min(1).float()
        score = token_max.sum(dim=-1) / denom

        if not return_details:
            return score

        return {
            "score": score,
            "token_max": token_max,
            "argmax_patch": argmax_patch.masked_fill(~mask, -1),
        }

    def score_pairs(
        self,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
        image_features: torch.Tensor,
        query_indices,
        gallery_indices,
        batch_size: int = 512,
        return_details: bool = False,
    ):
        qidx = torch.as_tensor(query_indices, dtype=torch.long)
        gidx = torch.as_tensor(gallery_indices, dtype=torch.long)
        if qidx.numel() != gidx.numel():
            raise ValueError("query_indices and gallery_indices must have the same length")

        scores = []
        token_max = []
        argmax_patch = []

        for start in range(0, qidx.numel(), batch_size):
            end = min(start + batch_size, qidx.numel())
            tq = text_features.index_select(0, qidx[start:end])
            tm = text_mask.index_select(0, qidx[start:end])
            ig = image_features.index_select(0, gidx[start:end])

            out = self.score_batch(tq, tm, ig, return_details=return_details)
            if return_details:
                scores.append(out["score"].detach().cpu())
                token_max.append(out["token_max"].detach().cpu())
                argmax_patch.append(out["argmax_patch"].detach().cpu())
            else:
                scores.append(out.detach().cpu())

        if not return_details:
            return torch.cat(scores, dim=0)

        return {
            "score": torch.cat(scores, dim=0),
            "token_max": torch.cat(token_max, dim=0),
            "argmax_patch": torch.cat(argmax_patch, dim=0),
        }


def maxsim_score(
    text_features: torch.Tensor,
    text_mask: torch.Tensor,
    image_features: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    scorer = HiddenScorer(device or torch.device("cpu"))
    return scorer.score_batch(text_features, text_mask, image_features)

