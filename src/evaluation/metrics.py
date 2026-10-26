"""Shared document completion, document-level NPMI and topic diversity."""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy import sparse
from src.models.author_prior_lda import token_arrays


def split_completion(matrix, fraction: float = .5, seed: int = 1001):
    if not 0 < fraction < 1:
        raise ValueError("Observation fraction must lie in (0, 1)")
    words, offsets, v_count = token_arrays(matrix)
    rng = np.random.default_rng(seed)
    observed, held = [], []
    for d in range(len(offsets) - 1):
        tokens = rng.permutation(words[offsets[d]:offsets[d + 1]])
        if len(tokens) < 2:
            raise ValueError("Completion needs at least 2 tokens per document")
        n = max(1, min(len(tokens) - 1, int(len(tokens) * fraction)))
        observed.append(np.bincount(tokens[:n], minlength=v_count))
        held.append(np.bincount(tokens[n:], minlength=v_count))
    return sparse.csr_matrix(observed), sparse.csr_matrix(held)


def completion_score(model, observed, held, authors=None, seed=1001) -> dict:
    theta = model.transform(observed, authors, random_state=seed)
    phi = model.get_topic_word_distribution()
    probabilities = theta @ phi
    per_doc = np.asarray(held.multiply(np.log(np.maximum(probabilities, np.finfo(float).tiny))).sum(1)).ravel()
    counts = np.asarray(held.sum(1)).ravel()
    log_probability = float(per_doc.sum() / counts.sum())
    return {"log_probability": log_probability, "perplexity": float(np.exp(-log_probability)),
        "heldout_tokens": int(counts.sum()), "per_document_log_sum": per_doc,
        "per_document_token_count": counts}


def topic_metrics(phi: np.ndarray, reference, top_m: int = 10) -> dict:
    """C_NPMI using documents as co-occurrence windows, no test text."""
    m = min(top_m, phi.shape[1])
    top = np.argsort(phi, axis=1)[:, -m:]
    if m < 2:
        raise ValueError("NPMI needs at least 2 top words")
    binary = reference.astype(np.int64).copy()
    binary.data[:] = 1
    # Restrict dense co-occurrence table to words actually needed.
    selected = np.unique(top)
    counts = (binary[:, selected].T @ binary[:, selected]).toarray()
    lookup = {int(w): i for i, w in enumerate(selected)}
    d_count = reference.shape[0]
    scores = []
    for words in top:
        pairs = []
        for wi, wj in combinations(words, 2):
            i, j = lookup[int(wi)], lookup[int(wj)]
            pij = counts[i, j] / d_count
            if pij == 0:
                pairs.append(-1.0)
            elif pij >= 1:
                pairs.append(0.0)  # two ubiquitous words carry no association information
            else:
                pi, pj = counts[i, i] / d_count, counts[j, j] / d_count
                pairs.append(float(np.log(pij / (pi * pj)) / -np.log(pij)))
        scores.append(float(np.mean(pairs)))
    return {"npmi": float(np.mean(scores)), "diversity": len(selected) / top.size,
        "per_topic_npmi": scores}
