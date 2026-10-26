"""Mature sklearn LDA training; shared Gibbs fold-in for comparable prediction."""
from __future__ import annotations

import time
import io
import re
from contextlib import redirect_stdout
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import config_context
from src.models.author_prior_lda import AuthorPriorLDA


class LDABaseline(AuthorPriorLDA):
    def fit(self, documents, author_ids=None):
        self.author_ids_ = ["anonymous"]
        self.author_aware = False
        self.author_topics_ = np.full((1, self.n_topics), 1 / self.n_topics)
        estimator = LatentDirichletAllocation(n_components=self.n_topics,
            doc_topic_prior=self.lambda_author / self.n_topics, topic_word_prior=self.beta,
            learning_method="batch", max_iter=self.n_iter, evaluate_every=5, verbose=1,
            random_state=self.random_state, n_jobs=1, max_doc_update_iter=100,
            # Disable early stopping so all runs receive their specified budget.
            perp_tol=0.0)
        start = time.perf_counter()
        log = io.StringIO()
        # sklearn 1.5's metadata validator restricts alpha to <=1 although its
        # variational updates support any positive Dirichlet parameter. Our
        # constructor validates positivity. A local public config context permits
        # concentration-matched alpha=lambda/K even for the large-lambda grid.
        with redirect_stdout(log), config_context(skip_parameter_validation=True):
            estimator.fit(documents)
        self.topic_words_ = estimator.components_ / estimator.components_.sum(1, keepdims=True)
        self.document_topics_ = estimator.transform(documents)
        self.n_features_in_ = documents.shape[1]
        self.topic_samples_ = self.topic_words_[None]
        self.author_samples_ = self.author_topics_[None]
        self.history_ = [{"iteration": int(i), "train_perplexity": float(p)}
            for i, p in re.findall(r"iteration: (\d+).*?perplexity: ([\d.]+)", log.getvalue())]
        self.entropy_trace_ = []
        self.diagnostics_ = {"sklearn_iterations": int(estimator.n_iter_),
            "training_seconds": time.perf_counter() - start, "all_finite": bool(np.isfinite(self.topic_words_).all())}
        return self
