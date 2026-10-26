"""Numerical kernels. CRT augmentation is derived in mathematical_model.md."""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.special import gammaln


@njit(cache=True)
def seed_kernel(seed: int) -> None:
    np.random.seed(seed)


@njit(cache=True)
def gibbs_sweep(words, doc_ids, z, ndk, nkv, nk, priors, beta):
    """A complete systematic scan conditional on author probabilities."""
    k_count, v_count = nkv.shape
    probs = np.empty(k_count)
    for i in range(len(words)):
        d, w, old = doc_ids[i], words[i], z[i]
        ndk[d, old] -= 1
        nkv[old, w] -= 1
        nk[old] -= 1
        total = 0.0
        for k in range(k_count):
            total += (ndk[d, k] + priors[d, k]) * (nkv[k, w] + beta) / (nk[k] + v_count * beta)
            probs[k] = total
        u = np.random.random() * total
        new = 0
        while new < k_count - 1 and u > probs[new]:
            new += 1
        z[i] = new
        ndk[d, new] += 1
        nkv[new, w] += 1
        nk[new] += 1


@njit(cache=True)
def crt_counts(ndk, author_index, pi, concentration):
    totals = np.zeros(pi.shape, dtype=np.int64)
    for d in range(ndk.shape[0]):
        a = author_index[d]
        for k in range(ndk.shape[1]):
            x = concentration * pi[a, k]
            for i in range(ndk[d, k]):
                if np.random.random() < x / (x + i):
                    totals[a, k] += 1
    return totals


@njit(cache=True)
def fold_in(words, offsets, author_index, pi, phi, concentration, gamma,
            n_iter, seed, hierarchical):
    """Fixed global parameters; unseen authors are inferred locally per document."""
    np.random.seed(seed)
    d_count, k_count = len(offsets) - 1, len(phi)
    result = np.zeros((d_count, k_count))
    for d in range(d_count):
        a = author_index[d]
        unknown = hierarchical and a < 0
        local_pi = np.empty(k_count)
        if a >= 0:
            local_pi[:] = pi[a]
        elif unknown:
            total = 0.0
            for k in range(k_count):
                local_pi[k] = np.random.gamma(gamma)
                total += local_pi[k]
            local_pi /= total
        else:
            local_pi[:] = 1.0 / k_count
        start, end = offsets[d], offsets[d + 1]
        if start == end:
            if a >= 0:
                result[d] = pi[a]
            else:
                result[d] = 1.0 / k_count
            continue
        counts = np.zeros(k_count, dtype=np.int64)
        labels = np.empty(end - start, dtype=np.int64)
        for i in range(end - start):
            labels[i] = np.random.randint(k_count)
            counts[labels[i]] += 1
        cumulative = np.empty(k_count)
        samples = 0
        for iteration in range(n_iter):
            for i in range(end - start):
                counts[labels[i]] -= 1
                total = 0.0
                for k in range(k_count):
                    total += (counts[k] + concentration * local_pi[k]) * phi[k, words[start + i]]
                    cumulative[k] = total
                u = np.random.random() * total
                new = 0
                while new < k_count - 1 and u > cumulative[new]:
                    new += 1
                labels[i] = new
                counts[new] += 1
            if unknown:
                total = 0.0
                for k in range(k_count):
                    tables = 0
                    x = concentration * local_pi[k]
                    for i in range(counts[k]):
                        if np.random.random() < x / (x + i):
                            tables += 1
                    local_pi[k] = np.random.gamma(gamma + tables)
                    total += local_pi[k]
                local_pi /= total
            if iteration >= n_iter // 2:
                result[d] += (counts + concentration * local_pi) / (end - start + concentration)
                samples += 1
        result[d] /= samples
    return result


def collapsed_log_joint(ndk: np.ndarray, nkv: np.ndarray, pi: np.ndarray,
                        author_index: np.ndarray, beta: float,
                        gamma: float, concentration: float, hierarchical: bool = True) -> float:
    """Density of (Z,W,pi), NOT a marginal likelihood or ELBO."""
    k_count, v_count = nkv.shape
    prior = concentration * pi[author_index] if hierarchical else np.full(ndk.shape, concentration / k_count)
    value = (k_count * gammaln(v_count * beta) - gammaln(nkv.sum(1) + v_count * beta).sum()
             + (gammaln(nkv + beta) - gammaln(beta)).sum())
    value += (len(ndk) * gammaln(concentration) - gammaln(ndk.sum(1) + concentration).sum()
              + (gammaln(ndk + prior) - gammaln(prior)).sum())
    if hierarchical:
        value += len(pi) * (gammaln(k_count * gamma) - k_count * gammaln(gamma))
        value += (gamma - 1) * np.log(pi).sum()
    return float(value)


def effective_sample_size(values: np.ndarray) -> float:
    """Initial positive pair autocorrelation estimate, limited to scalar trace."""
    x = np.asarray(values, float)
    x = x - x.mean()
    if len(x) < 4 or np.dot(x, x) < 1e-20:
        return float(len(x))
    acf = np.correlate(x, x, mode="full")[len(x) - 1:] / np.dot(x, x)
    total = 0.0
    for i in range(1, len(x) - 1, 2):
        pair = acf[i] + acf[i + 1]
        if pair <= 0:
            break
        total += pair
    return float(np.clip(len(x) / (1 + 2 * total), 1, len(x)))


def split_rhat(chains: list[np.ndarray]) -> float:
    """Classical split R-hat for a label-invariant scalar diagnostic."""
    n = min(map(len, chains)) // 2
    if n < 2:
        return float("nan")
    x = np.array([half for c in chains for half in (c[:n], c[-n:])])
    within = x.var(axis=1, ddof=1).mean()
    if within <= 0:
        return float("nan")
    between = n * x.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) * within / n + between / n) / within))
