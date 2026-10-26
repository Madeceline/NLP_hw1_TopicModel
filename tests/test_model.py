import numpy as np
import pytest
from scipy import sparse
from src.models import AuthorPriorLDA, LDABaseline
from src.data.synthetic import generate, align_topics


@pytest.fixture
def tiny():
    return sparse.csr_matrix([[5, 3, 0, 0], [4, 1, 1, 0], [0, 0, 3, 5], [0, 1, 4, 3]]), ["a", "a", "b", "b"]


def fit_tiny(tiny):
    return AuthorPriorLDA(n_topics=2, n_iter=60, burn_in=30, thin=3).fit(*tiny)


def test_simplex_and_finiteness(tiny):
    model = fit_tiny(tiny)
    for p in [model.get_author_topic_distribution(), model.get_topic_word_distribution(), model.document_topics_, model.transform(*tiny)]:
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-12)
    assert model.diagnostics_["all_finite"]
    assert np.isfinite([h["log_joint"] for h in model.history_]).all()


@pytest.mark.parametrize("kwargs", [{"gamma": 0}, {"beta": -1}, {"lambda_author": np.nan}, {"n_topics": 0}, {"burn_in": 300}, {"thin": 0}])
def test_positive_parameters(kwargs):
    with pytest.raises(ValueError):
        AuthorPriorLDA(**kwargs)


def test_seed_and_safe_serialization(tiny, tmp_path):
    first, second = fit_tiny(tiny), fit_tiny(tiny)
    np.testing.assert_array_equal(first.author_samples_, second.author_samples_)
    np.testing.assert_array_equal(first.topic_samples_, second.topic_samples_)
    first.save(tmp_path / "model.npz")
    loaded = AuthorPriorLDA.load(tmp_path / "model.npz")
    np.testing.assert_array_equal(first.transform(*tiny), loaded.transform(*tiny))
    np.testing.assert_array_equal(first.transform(tiny[0]), loaded.transform(tiny[0]))


def test_unseen_author_and_empty_document(tiny):
    model = fit_tiny(tiny)
    x = sparse.csr_matrix([[0, 0, 0, 0], [3, 2, 1, 0]])
    probs = model.transform(x, [None, "unseen"])
    np.testing.assert_allclose(probs[0], [.5, .5])
    np.testing.assert_allclose(probs.sum(1), 1)
    before = model.author_topics_.copy()
    np.testing.assert_array_equal(probs, model.transform(x))
    np.testing.assert_array_equal(before, model.author_topics_)


@pytest.mark.parametrize("documents", [[[-1]], [[0.5]], [[float('nan')]], [["word"]]])
def test_invalid_tokens(documents):
    with pytest.raises(ValueError):
        AuthorPriorLDA(n_topics=2).fit(documents, ["a"])


def test_one_token_training():
    model = AuthorPriorLDA(n_topics=2, n_iter=10, burn_in=5).fit([[0]], [1])
    np.testing.assert_allclose(model.get_topic_word_distribution(), 1)
    with pytest.raises(ValueError):
        model.transform([[1]])
    with pytest.raises(ValueError):
        AuthorPriorLDA().fit([[]], [1])


def test_synthetic_recovery_and_permutation():
    x, authors, pi, phi, _ = generate()
    model = AuthorPriorLDA(n_topics=3, lambda_author=12, beta=.05,
        n_iter=350, burn_in=150, thin=5, random_state=19).fit(x, authors)
    order = align_topics(model.topic_words_, phi)
    recovered = model.author_topics_[:, order]
    error = np.abs(recovered - pi).mean()
    shuffled = np.random.default_rng(19).permutation(authors).tolist()
    control = AuthorPriorLDA(n_topics=3, lambda_author=12, beta=.05,
        n_iter=350, burn_in=150, thin=5, random_state=19).fit(x, shuffled)
    control_order = align_topics(control.topic_words_, phi)
    # Control's author order depends on its first observed label.
    indices = [control.author_ids_.index(str(a)) for a in range(len(pi))]
    control_error = np.abs(control.author_topics_[indices][:, control_order] - pi).mean()
    assert error < .12, error
    assert error < control_error, (error, control_error)


def test_sklearn_baseline_checkpoint(tiny, tmp_path):
    model = LDABaseline(n_topics=2, n_iter=8, burn_in=2).fit(tiny[0])
    model.save(tmp_path / "baseline.npz")
    loaded = LDABaseline.load(tmp_path / "baseline.npz")
    np.testing.assert_array_equal(model.transform(tiny[0]), loaded.transform(tiny[0]))
