"""Small, independently testable numerical components. Metrics are fractions."""
import numpy as np
import torch
import torch.nn.functional as F


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def pool_tokens(x, modality, tokens=None):
    """Raw block outputs [batch, tokens, width]; exclude specials by position."""
    if modality == "image":
        return x[:, 0], x[:, 1:].mean(1)
    eos = tokens.argmax(-1)
    pos = torch.arange(tokens.shape[1], device=tokens.device)[None]
    mask = (pos > 0) & (pos < eos[:, None])
    if not mask.any(1).all():
        raise ValueError("Empty caption: content-token pooling is undefined")
    # Token ID zero can be a real content token; do NOT mask by token ID.
    return x[torch.arange(len(x), device=x.device), eos], (x * mask[..., None]).sum(1) / mask.sum(1)[:, None]


def partition(ids, fractions, seed):
    unique = np.unique(ids)
    if len(unique) < 12:
        raise ValueError("Need at least 12 held-out identities")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cuts = (np.cumsum(fractions)[:-1] * len(unique)).astype(int)
    return [a.tolist() for a in np.split(unique, cuts)]


def targets(ids, opposite, opposite_ids):
    result = []
    for pid in ids.unique():
        mask = ids == pid
        other = opposite[opposite_ids == pid]
        if not len(other):
            raise ValueError(f"No opposite-modality positive for identity {pid}")
        result.append((mask, normalize(other.mean(0))))
    y = torch.empty(len(ids), opposite.shape[-1])
    for mask, value in result:
        y[mask] = value
    return y


def ridge(x, y, ids, penalty):
    """Identity-balanced ridge, objective mean weighted MSE + lambda ||W||²."""
    x, y = x.double(), y.double()
    _, inverse, counts = ids.unique(return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse].double()
    weights /= weights.sum()
    matrix = x.T @ (weights[:, None] * x)
    matrix += penalty * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    return torch.linalg.solve(matrix, x.T @ (weights[:, None] * y)).float()


def retrieval(q, g, qids, gids, device="cpu", chunk=64):
    """Full gallery, stable ties, memory bounded by query chunk × gallery size."""
    g = normalize(g).to(device)
    gids = gids.to(device)
    rows = []
    for start in range(0, len(q), chunk):
        score = normalize(q[start:start + chunk]).to(device) @ g.T
        order = score.argsort(dim=1, descending=True, stable=True)
        hit = gids[order] == qids[start:start + chunk].to(device)[:, None]
        counts = hit.sum(1)
        if not (counts > 0).all():
            raise ValueError("A retrieval query has no gallery positive")
        ranks = torch.arange(1, len(g) + 1, device=device)[None]
        ap = ((hit.cumsum(1) / ranks) * hit).sum(1) / counts
        last = (hit * ranks).max(1).values
        rows.append(torch.stack([hit[:, :k].any(1).float() for k in (1, 5, 10)] + [ap, counts / last], 1).cpu())
    return torch.cat(rows)


def metrics(rows):
    return dict(zip(("r1", "r5", "r10", "map", "minp"), rows.mean(0).tolist()))


def paired_ci(rows, baseline, ids, seed=42, repetitions=2000):
    """Resample identities, preserve query-weighted metric within each resample."""
    unique = ids.unique()
    sums = np.array([(rows[ids == i] - baseline[ids == i]).sum(0).numpy() for i in unique])
    counts = np.array([(ids == i).sum().item() for i in unique])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sample = rng.integers(len(unique), size=len(unique))
        values.append(sums[sample].sum(0) / counts[sample].sum())
    bounds = np.quantile(values, [.025, .975], axis=0)
    return {k: [float(bounds[0, j]), float(bounds[1, j])] for j, k in enumerate(("r1", "r5", "r10", "map", "minp"))}
