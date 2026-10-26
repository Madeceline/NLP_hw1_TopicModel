"""Author-prior LDA, implemented from scratch via collapsed/CRT Gibbs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse

from src.inference.inference import (collapsed_log_joint, crt_counts,
    effective_sample_size, fold_in, gibbs_sweep, seed_kernel)


def token_arrays(documents, n_features: int | None = None):
    """Accept integer token sequences or a nonnegative integer count matrix."""
    if sparse.issparse(documents):
        matrix = documents.tocsr(copy=True)
        matrix.sum_duplicates()
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any() or (matrix.data != np.floor(matrix.data)).any():
            raise ValueError("Count matrix must contain finite nonnegative integers")
        if n_features is not None and matrix.shape[1] != n_features:
            raise ValueError("Vocabulary dimension differs from training")
        docs = [np.repeat(matrix.indices[matrix.indptr[d]:matrix.indptr[d + 1]],
                          matrix.data[matrix.indptr[d]:matrix.indptr[d + 1]].astype(np.int64))
                for d in range(matrix.shape[0])]
        inferred = matrix.shape[1]
    else:
        docs = []
        for doc in documents:
            raw = np.asarray(doc)
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number) and raw.size:
                raise ValueError("Documents must be one-dimensional integer token IDs")
            if raw.size and (not np.isfinite(raw).all() or (raw < 0).any() or (raw != np.floor(raw)).any()):
                raise ValueError("Token IDs must be nonnegative integers")
            docs.append(raw.astype(np.int64))
        inferred = max((int(doc.max()) + 1 for doc in docs if len(doc)), default=0)
        if n_features is not None and inferred > n_features:
            raise ValueError("Token ID outside training vocabulary")
    v_count = n_features if n_features is not None else inferred
    offsets = np.r_[0, np.cumsum([len(d) for d in docs])].astype(np.int64)
    words = np.concatenate(docs).astype(np.int64) if docs else np.empty(0, np.int64)
    return words, offsets, v_count


class AuthorPriorLDA:
    """Fit encoded BOW/tokens; vocabulary preparation is explicit and train-only.

    `author_topics_[i]` corresponds to `author_ids_[i]`. `history_` stores
    density/entropy diagnostics; it is a sampling trace, not a loss to minimize.
    """

    def __init__(self, n_topics: int = 10, gamma: float = 0.5,
                 beta: float = 0.01, lambda_author: float = 10.0,
                 random_state: int = 42, n_iter: int = 240,
                 burn_in: int = 120, thin: int = 6, foldin_iter: int = 100,
                 author_aware: bool = True):
        if not isinstance(n_topics, int) or n_topics < 1:
            raise ValueError("n_topics must be a positive integer")
        if any(not np.isfinite(x) or x <= 0 for x in (gamma, beta, lambda_author)):
            raise ValueError("Dirichlet parameters must be finite and strictly positive")
        if not (isinstance(n_iter, int) and isinstance(burn_in, int) and isinstance(thin, int)
                and 0 <= burn_in < n_iter and thin >= 1 and foldin_iter >= 2):
            raise ValueError("Invalid sampling schedule")
        self.n_topics, self.gamma, self.beta = n_topics, float(gamma), float(beta)
        self.lambda_author, self.random_state = float(lambda_author), int(random_state)
        self.n_iter, self.burn_in, self.thin = n_iter, burn_in, thin
        self.foldin_iter, self.author_aware = foldin_iter, author_aware

    def get_params(self) -> dict:
        return {key: getattr(self, key) for key in (
            "n_topics", "gamma", "beta", "lambda_author", "random_state", "n_iter",
            "burn_in", "thin", "foldin_iter", "author_aware")}

    def fit(self, documents, author_ids: Sequence[str | int] | None = None):
        words, offsets, v_count = token_arrays(documents)
        d_count = len(offsets) - 1
        if d_count == 0 or len(words) == 0 or v_count == 0:
            raise ValueError("Training corpus must contain observed tokens")
        if author_ids is None:
            if self.author_aware:
                raise ValueError("Training author IDs are required for AP-LDA")
            author_ids = ["anonymous"] * d_count
        if len(author_ids) != d_count or any(a is None for a in author_ids):
            raise ValueError("Provide one nonmissing author per training document")
        self.author_ids_ = list(dict.fromkeys(map(str, author_ids)))
        lookup = {a: i for i, a in enumerate(self.author_ids_)}
        author_index = np.array([lookup[str(a)] for a in author_ids], dtype=np.int64)
        rng = np.random.default_rng(self.random_state)
        seed_kernel(self.random_state)
        k_count, a_count = self.n_topics, len(lookup)
        doc_ids = np.repeat(np.arange(d_count), np.diff(offsets)).astype(np.int64)
        z = rng.integers(k_count, size=len(words), dtype=np.int64)
        ndk = np.zeros((d_count, k_count), np.int64)
        nkv = np.zeros((k_count, v_count), np.int64)
        np.add.at(ndk, (doc_ids, z), 1)
        np.add.at(nkv, (z, words), 1)
        nk = nkv.sum(1)
        pi = rng.dirichlet(np.full(k_count, self.gamma), size=a_count)
        if not self.author_aware:
            pi[:] = 1 / k_count
        self.history_, pi_samples, phi_samples, theta_samples = [], [], [], []
        self.entropy_trace_ = []
        for iteration in range(self.n_iter):
            prior = self.lambda_author * pi[author_index]
            gibbs_sweep(words, doc_ids, z, ndk, nkv, nk, prior, self.beta)
            if self.author_aware:
                tables = crt_counts(ndk, author_index, pi, self.lambda_author)
                pi = np.array([rng.dirichlet(self.gamma + row) for row in tables])
            if not np.isfinite(pi).all() or (pi <= 0).any():
                raise FloatingPointError("Dirichlet sampling underflow: use a less extreme hyperparameter")
            entropy = float(-(pi * np.log(pi)).sum(1).mean())
            self.entropy_trace_.append(entropy)
            if iteration % 5 == 0 or iteration == self.n_iter - 1:
                self.history_.append({"iteration": iteration + 1,
                    "log_joint": collapsed_log_joint(ndk, nkv, pi, author_index,
                        self.beta, self.gamma, self.lambda_author, self.author_aware),
                    "author_entropy": entropy})
            if iteration >= self.burn_in and (iteration - self.burn_in) % self.thin == 0:
                pi_samples.append(pi.copy())
                phi_samples.append((nkv + self.beta) / (nk[:, None] + v_count * self.beta))
                theta_samples.append((ndk + self.lambda_author * pi[author_index]) /
                                     (ndk.sum(1, keepdims=True) + self.lambda_author))
        self.author_samples_ = np.asarray(pi_samples)
        self.topic_samples_ = np.asarray(phi_samples)
        self.author_topics_ = self.author_samples_.mean(0)
        self.topic_words_ = self.topic_samples_.mean(0)
        self.document_topics_ = np.mean(theta_samples, axis=0)
        self.n_features_in_ = v_count
        self.diagnostics_ = {
            "retained_samples": len(pi_samples), "n_tokens": len(words),
            "entropy_ess": effective_sample_size(np.asarray(self.entropy_trace_[self.burn_in:])),
            "max_normalization_error": max(float(np.abs(p.sum(-1) - 1).max()) for p in
                (self.author_topics_, self.topic_words_, self.document_topics_)),
            "all_finite": all(np.isfinite(p).all() for p in
                (self.author_topics_, self.topic_words_, self.document_topics_))}
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "topic_words_"):
            raise RuntimeError("Fit or load the model before inference")

    def transform(self, documents, author_ids: Sequence[str | int | None] | None = None,
                  random_state: int | None = None) -> np.ndarray:
        self._check_fitted()
        words, offsets, _ = token_arrays(documents, self.n_features_in_)
        d_count = len(offsets) - 1
        if author_ids is None:
            author_ids = [None] * d_count
        if len(author_ids) != d_count:
            raise ValueError("Provide one author ID per input document")
        lookup = {a: i for i, a in enumerate(self.author_ids_)}
        indices = np.array([lookup.get(str(a), -1) if a is not None else -1 for a in author_ids], np.int64)
        result = fold_in(words, offsets, indices, self.author_topics_, self.topic_words_,
            self.lambda_author, self.gamma, self.foldin_iter,
            self.random_state if random_state is None else int(random_state), self.author_aware)
        if not np.isfinite(result).all() or (result < 0).any():
            raise FloatingPointError("Nonfinite local topic estimate")
        return result

    def get_author_topic_distribution(self) -> np.ndarray:
        self._check_fitted()
        return self.author_topics_.copy()

    def get_topic_word_distribution(self) -> np.ndarray:
        self._check_fitted()
        return self.topic_words_.copy()

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format_version": 1, "params": self.get_params(),
            "author_ids": self.author_ids_, "history": self.history_, "diagnostics": self.diagnostics_}
        # NPZ+JSON does not execute pickle payloads on loading.
        with path.open("wb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(metadata),
                author_topics=self.author_topics_, topic_words=self.topic_words_,
                document_topics=self.document_topics_, author_samples=self.author_samples_,
                topic_samples=self.topic_samples_, entropy_trace=self.entropy_trace_)

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["format_version"] != 1:
                raise ValueError("Unsupported checkpoint version")
            model = cls(**metadata["params"])
            for name in ("author_topics", "topic_words", "document_topics", "author_samples", "topic_samples", "entropy_trace"):
                setattr(model, name + "_", saved[name].copy())
            model.author_ids_, model.history_ = metadata["author_ids"], metadata["history"]
            model.diagnostics_ = metadata["diagnostics"]
            model.n_features_in_ = model.topic_words_.shape[1]
        return model
