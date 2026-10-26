"""Clearly labelled generated data for recovery tests, never real-corpus metrics."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment


def generate(seed: int = 123, n_authors: int = 6, docs_per_author: int = 24,
             n_topics: int = 3, n_words: int = 36, length: int = 90,
             gamma: float = 0.5, concentration: float = 12.0, beta: float = 0.05):
    rng = np.random.default_rng(seed)
    phi = rng.dirichlet(np.full(n_words, beta), size=n_topics)
    pi = rng.dirichlet(np.full(n_topics, gamma), size=n_authors)
    rows, authors, thetas = [], [], []
    for a in range(n_authors):
        for _ in range(docs_per_author):
            theta = rng.dirichlet(concentration * pi[a])
            z = rng.choice(n_topics, length, p=theta)
            words = np.array([rng.choice(n_words, p=phi[k]) for k in z])
            rows.append(np.bincount(words, minlength=n_words))
            authors.append(str(a))
            thetas.append(theta)
    return sparse.csr_matrix(rows), authors, pi, phi, np.array(thetas)


def align_topics(fitted_phi, true_phi):
    costs = np.abs(fitted_phi[:, None] - true_phi[None]).sum(-1)
    fitted, true = linear_sum_assignment(costs)
    return fitted[np.argsort(true)]
