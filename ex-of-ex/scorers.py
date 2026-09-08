"""Parameter-free scorers for the frozen final-layer CLIP/ITSELF tokens.

Every scorer consumes the same cosine-similarity tensor ``[batch, text, patch]``.
That is deliberate: the expensive pairwise matmul is done once per batch and all
experiments fan out from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ScorerSpec:
    name: str
    kind: str
    params: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ScorerSpec":
        if "name" not in value or "kind" not in value:
            raise ValueError(f"Each scorer needs 'name' and 'kind': {value}")
        params = {k: v for k, v in value.items() if k not in {"name", "kind"}}
        return cls(name=str(value["name"]), kind=str(value["kind"]), params=params)


SUPPORTED_KINDS = {
    "maxsim",
    "topm",
    "softmax_t2i",
    "smooth_chamfer",
    "cfine_cross_grained",
    "tokenflow_stable",
    "sinkhorn_ot",
    "uniform_mean",
}


def build_specs(values: Iterable[Mapping[str, object]]) -> list[ScorerSpec]:
    specs = [ScorerSpec.from_mapping(value) for value in values]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Scorer names must be unique: {names}")
    unsupported = sorted({spec.kind for spec in specs} - SUPPORTED_KINDS)
    if unsupported:
        raise ValueError(f"Unsupported scorer kinds: {unsupported}")
    if not specs:
        raise ValueError("At least one scorer is required")
    return specs


def score_similarity_tensor(
    similarities: torch.Tensor,
    text_mask: torch.Tensor,
    specs: Iterable[ScorerSpec],
    text_features: Optional[torch.Tensor] = None,
    image_features: Optional[torch.Tensor] = None,
    text_global: Optional[torch.Tensor] = None,
    image_global: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Score one batch after its local cosine tensor has been computed once."""
    if similarities.ndim != 3:
        raise ValueError("similarities must have shape [batch, text_tokens, patches]")
    if text_mask.shape != similarities.shape[:2]:
        raise ValueError("text_mask must match similarities[:2]")

    mask = text_mask.bool()
    results: Dict[str, torch.Tensor] = {}
    for spec in specs:
        params = spec.params
        if spec.kind == "maxsim":
            token_scores = similarities.max(dim=2).values
            score = _masked_mean(token_scores, mask, dim=1)
        elif spec.kind == "topm":
            m = min(int(params.get("m", 3)), similarities.shape[2])
            if m <= 0:
                raise ValueError(f"{spec.name}: m must be positive")
            token_scores = similarities.topk(m, dim=2).values.mean(dim=2)
            score = _masked_mean(token_scores, mask, dim=1)
        elif spec.kind == "softmax_t2i":
            temperature = _positive_float(params, "temperature", 0.05, spec.name)
            weights = torch.softmax(similarities / temperature, dim=2)
            token_scores = (weights * similarities).sum(dim=2)
            score = _masked_mean(token_scores, mask, dim=1)
        elif spec.kind == "smooth_chamfer":
            alpha = _positive_float(params, "alpha", 10.0, spec.name)
            score = _smooth_chamfer(similarities, mask, alpha)
        elif spec.kind == "uniform_mean":
            token_scores = similarities.mean(dim=2)
            score = _masked_mean(token_scores, mask, dim=1)
        elif spec.kind == "cfine_cross_grained":
            _require_globals(spec.name, text_features, image_features, text_global, image_global)
            temperature = _positive_float(params, "temperature", 0.05, spec.name)
            score = _cfine_cross_grained(
                text_features,
                image_features,
                mask,
                text_global,
                image_global,
                temperature,
            )
        elif spec.kind == "tokenflow_stable":
            _require_globals(spec.name, text_features, image_features, text_global, image_global)
            flow_lambda = _positive_float(params, "lambda", 10.0, spec.name)
            floor = float(params.get("importance_floor", 0.05))
            if floor < 0:
                raise ValueError(f"{spec.name}: importance_floor cannot be negative")
            score = _tokenflow_stable(
                similarities,
                text_features,
                image_features,
                mask,
                text_global,
                image_global,
                flow_lambda,
                floor,
            )
        elif spec.kind == "sinkhorn_ot":
            epsilon = _positive_float(params, "epsilon", 0.05, spec.name)
            iterations = int(params.get("iterations", 10))
            if iterations <= 0:
                raise ValueError(f"{spec.name}: iterations must be positive")
            score = _sinkhorn_similarity(similarities, mask, epsilon, iterations)
        else:  # guarded by build_specs, retained for direct callers
            raise ValueError(f"Unsupported scorer kind: {spec.kind}")

        results[spec.name] = score.float()
    return results


def score_pairs_all(
    specs: Iterable[ScorerSpec],
    text_features: torch.Tensor,
    text_mask: torch.Tensor,
    image_features: torch.Tensor,
    query_indices: torch.Tensor,
    gallery_indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
    query_global: Optional[torch.Tensor] = None,
    gallery_global: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Score aligned query-gallery pairs for every configured scorer."""
    specs = list(specs)
    if query_indices.numel() != gallery_indices.numel():
        raise ValueError("query_indices and gallery_indices must have equal length")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunks: Dict[str, list[torch.Tensor]] = {spec.name: [] for spec in specs}
    with torch.inference_mode():
        for start in range(0, query_indices.numel(), batch_size):
            stop = min(start + batch_size, query_indices.numel())
            qidx = query_indices[start:stop].long().cpu()
            gidx = gallery_indices[start:stop].long().cpu()
            text = text_features.index_select(0, qidx).to(device=device, dtype=torch.float32)
            mask = text_mask.index_select(0, qidx).to(device=device)
            image = image_features.index_select(0, gidx).to(device=device, dtype=torch.float32)
            qglobal = None
            gglobal = None
            if query_global is not None:
                qglobal = F.normalize(
                    query_global.index_select(0, qidx).to(device=device, dtype=torch.float32),
                    p=2,
                    dim=1,
                )
            if gallery_global is not None:
                gglobal = F.normalize(
                    gallery_global.index_select(0, gidx).to(device=device, dtype=torch.float32),
                    p=2,
                    dim=1,
                )

            similarities = torch.bmm(text, image.transpose(1, 2))
            batch_scores = score_similarity_tensor(
                similarities,
                mask,
                specs,
                text_features=text,
                image_features=image,
                text_global=qglobal,
                image_global=gglobal,
            )
            for name, values in batch_scores.items():
                chunks[name].append(values.detach().cpu())

    return {
        name: torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.float32)
        for name, parts in chunks.items()
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = values.masked_fill(~mask, torch.finfo(values.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def _smooth_chamfer(
    similarities: torch.Tensor,
    text_mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    # Normalized log-mean-exp keeps score scales comparable across token counts.
    num_patches = similarities.shape[2]
    t2i = torch.logsumexp(alpha * similarities, dim=2) / alpha
    t2i = t2i - torch.log(torch.tensor(float(num_patches), device=similarities.device)) / alpha
    t2i = _masked_mean(t2i, text_mask, dim=1)

    valid = text_mask.unsqueeze(2)
    masked = (alpha * similarities).masked_fill(~valid, torch.finfo(similarities.dtype).min)
    i2t = torch.logsumexp(masked, dim=1) / alpha
    lengths = text_mask.sum(dim=1).clamp_min(1).to(similarities.dtype)
    i2t = i2t - torch.log(lengths).unsqueeze(1) / alpha
    i2t = i2t.mean(dim=1)
    return 0.5 * (t2i + i2t)


def _cfine_cross_grained(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    text_mask: torch.Tensor,
    text_global: torch.Tensor,
    image_global: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    token_logits = torch.einsum("btd,bd->bt", text_features, image_global) / temperature
    token_weights = _masked_softmax(token_logits, text_mask, dim=1)
    image_word = (token_weights * torch.einsum("btd,bd->bt", text_features, image_global)).sum(1)

    patch_logits = torch.einsum("bpd,bd->bp", image_features, text_global) / temperature
    patch_weights = torch.softmax(patch_logits, dim=1)
    patch_sentence = (patch_weights * torch.einsum("bpd,bd->bp", image_features, text_global)).sum(1)
    return 0.5 * (image_word + patch_sentence)


def _tokenflow_stable(
    similarities: torch.Tensor,
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    text_mask: torch.Tensor,
    text_global: torch.Tensor,
    image_global: torch.Tensor,
    flow_lambda: float,
    floor: float,
) -> torch.Tensor:
    # TokenFlow uses global-local correlations as importance marginals.  The
    # affine shift makes frozen vanilla-CLIP cosine values non-negative while
    # preserving their order; cardinality normalization gives each flow mass 1.
    token_importance = (1.0 + torch.einsum("btd,bd->bt", text_features, image_global)) * 0.5
    patch_importance = (1.0 + torch.einsum("bpd,bd->bp", image_features, text_global)) * 0.5
    token_importance = (token_importance + floor) * text_mask.to(similarities.dtype)
    patch_importance = patch_importance + floor

    lengths = text_mask.sum(dim=1, keepdim=True).clamp_min(1).to(similarities.dtype)
    token_importance = token_importance / token_importance.sum(1, keepdim=True).clamp_min(1e-12)
    token_importance = token_importance * lengths
    patch_importance = patch_importance / patch_importance.sum(1, keepdim=True).clamp_min(1e-12)
    patch_importance = patch_importance * similarities.shape[2]

    # [B, P, T] follows the paper's patch-to-token notation.
    sim_pt = similarities.transpose(1, 2)
    token_mask_pt = text_mask.unsqueeze(1).expand_as(sim_pt)
    flow_v_logits = flow_lambda * sim_pt * token_importance.unsqueeze(1)
    flow_v = _masked_softmax(flow_v_logits, token_mask_pt, dim=2)
    flow_v = flow_v * patch_importance.unsqueeze(2) / similarities.shape[2]

    flow_t_logits = flow_lambda * sim_pt * patch_importance.unsqueeze(2)
    flow_t = torch.softmax(flow_t_logits, dim=1)
    flow_t = flow_t * token_importance.unsqueeze(1) / lengths.unsqueeze(1)
    flow_t = flow_t * token_mask_pt.to(flow_t.dtype)
    return 0.5 * ((flow_v * sim_pt).sum((1, 2)) + (flow_t * sim_pt).sum((1, 2)))


def _sinkhorn_similarity(
    similarities: torch.Tensor,
    text_mask: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    batch, _, patches = similarities.shape
    mask = text_mask.to(similarities.dtype)
    a = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    b = torch.full(
        (batch, patches),
        1.0 / float(patches),
        dtype=similarities.dtype,
        device=similarities.device,
    )
    valid = text_mask.unsqueeze(2)
    shifted = similarities - similarities.masked_fill(~valid, -torch.inf).amax((1, 2), keepdim=True)
    kernel = torch.exp(shifted / epsilon) * valid.to(similarities.dtype)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iterations):
        u = a / torch.bmm(kernel, v.unsqueeze(2)).squeeze(2).clamp_min(1e-12)
        v = b / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(2)).squeeze(2).clamp_min(1e-12)
    plan = u.unsqueeze(2) * kernel * v.unsqueeze(1)
    return (plan * similarities).sum((1, 2))


def _positive_float(
    params: Mapping[str, object], key: str, default: float, scorer_name: str
) -> float:
    value = float(params.get(key, default))
    if value <= 0:
        raise ValueError(f"{scorer_name}: {key} must be positive")
    return value


def _require_globals(name, text_features, image_features, text_global, image_global) -> None:
    if any(value is None for value in (text_features, image_features, text_global, image_global)):
        raise ValueError(f"{name} needs local and global features")
