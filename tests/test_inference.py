import numpy as np
from itertools import product
from scipy.integrate import quad
from scipy.special import gammaln, digamma
from src.inference.inference import (crt_counts, seed_kernel, collapsed_log_joint, effective_sample_size, split_rhat)
from src.evaluation.metrics import split_completion, completion_score, topic_metrics
from src.models import AuthorPriorLDA
from scipy import sparse


def test_crt_expectation_and_support():
    seed_kernel(9)
    n = np.array([[0, 8]], np.int64)
    pi = np.array([[.3, .7]])
    draws = np.array([crt_counts(n, np.array([0]), pi, 5)[0] for _ in range(10000)])
    assert (draws[:, 0] == 0).all()
    assert ((draws[:, 1] >= 1) & (draws[:, 1] <= 8)).all()
    expected = 3.5 * (digamma(3.5 + 8) - digamma(3.5))
    assert abs(draws[:, 1].mean() - expected) < .05


def test_augmented_pi_kernel_against_numerical_quadrature():
    # Known counts: test actual nonconjugate posterior, not just a simplex check.
    counts = np.array([[9, 1], [7, 2]], np.int64)
    concentration, gamma = 5., .7
    def density(p):
        prior = concentration * np.array([p, 1-p])
        logp = ((gamma - 1) * np.log([p, 1-p])).sum()
        logp += (gammaln(counts + prior) - gammaln(prior)).sum()
        return np.exp(logp - 24)
    mass = quad(density, 0, 1, epsabs=1e-11)[0]
    target = np.array([quad(lambda p: density(p) * p**power, 0, 1, epsabs=1e-11)[0] / mass for power in [1, 2]])
    seed_kernel(172)
    rng = np.random.default_rng(172)
    pi, values = np.array([[.5, .5]]), []
    for i in range(22000):
        tables = crt_counts(counts, np.array([0, 0]), pi, concentration)
        pi[0] = rng.dirichlet(gamma + tables[0])
        if i >= 2000:
            values.append(pi[0, 0])
    estimated = np.array([np.mean(values), np.mean(np.array(values)**2)])
    np.testing.assert_allclose(estimated, target, atol=.012)


def test_z_conditional_matches_collapsed_density():
    ndk = np.array([[2, 1], [0, 2]])
    nkv = np.array([[2, 0], [1, 2]])
    pi, authors = np.array([[.8, .2]]), np.array([0, 0])
    # Compare adding a word v=0 to doc 1, choosing k=0 or k=1.
    logp = []
    for k in [0, 1]:
        nd, nv = ndk.copy(), nkv.copy()
        nd[1, k] += 1
        nv[k, 0] += 1
        logp.append(collapsed_log_joint(nd, nv, pi, authors, .1, .5, 4.))
    log_ratio = (np.log((ndk[1] + 4*pi[0]) * (nkv[:, 0]+.1)/(nkv.sum(1)+.2)))
    np.testing.assert_allclose(logp[0]-logp[1], log_ratio[0]-log_ratio[1], atol=1e-12)


def test_full_small_posterior_against_enumeration_and_integral():
    words, doc_ids = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])
    configurations = []
    for labels in product(range(2), repeat=4):
        ndk, nkv = np.zeros((2,2),dtype=int), np.zeros((2,2),dtype=int)
        np.add.at(ndk,(doc_ids,labels),1)
        np.add.at(nkv,(labels,words),1)
        configurations.append((ndk,nkv))
    def integrand(p, entropy=False):
        pi = np.array([[p,1-p]])
        density = sum(np.exp(collapsed_log_joint(nd,nv,pi,np.array([0,0]),.5,.7,2.7)) for nd,nv in configurations)
        return density * (-(pi*np.log(pi)).sum() if entropy else 1)
    mass = quad(integrand,0,1,epsabs=1e-9)[0]
    target = quad(lambda p: integrand(p,True),0,1,epsabs=1e-9)[0]/mass
    model = AuthorPriorLDA(n_topics=2,gamma=.7,beta=.5,lambda_author=2.7,
        n_iter=14000,burn_in=2000,thin=20,random_state=135).fit([[0,0],[1,1]],['a','a'])
    estimate=np.mean(model.entropy_trace_[2000:])
    assert abs(estimate-target) < .025, (estimate,target)


def test_dirichlet_moments_and_cross_document_correlation():
    rng = np.random.default_rng(123)
    pi = np.array([.2, .3, .5])
    for concentration in [1, 10, 100]:
        draws = rng.dirichlet(concentration*pi, size=50000)
        np.testing.assert_allclose(draws.mean(0), pi, atol=.006)
        np.testing.assert_allclose(draws.var(0), pi*(1-pi)/(concentration+1), atol=.003)
    # Conditional independence gives nonzero marginal correlation for shared author.
    n, gamma, concentration, k = 35000, .5, 10, 3
    authors = rng.dirichlet([gamma]*k, size=n)
    first = np.array([rng.dirichlet(concentration*p)[0] for p in authors])
    second = np.array([rng.dirichlet(concentration*p)[0] for p in authors])
    expected = (concentration+1)/(k*gamma+concentration+1)
    assert abs(np.corrcoef(first, second)[0, 1]-expected) < .02


def test_completion_conserves_tokens_and_no_global_updates():
    x = sparse.csr_matrix([[4, 2, 0], [0, 1, 5]])
    observed, held = split_completion(x, seed=7)
    np.testing.assert_array_equal((observed+held).toarray(), x.toarray())
    model = AuthorPriorLDA(n_topics=2, n_iter=40, burn_in=20).fit(x, ['a','b'])
    before = model.topic_words_.copy()
    score = completion_score(model, observed, held, ['a','b'])
    np.testing.assert_array_equal(before, model.topic_words_)
    assert score['heldout_tokens'] == held.sum()
    assert np.isfinite(score['perplexity'])
    # Changing only scored words does not alter theta, since only O is passed to transform.
    theta = model.transform(observed, ['a','b'])
    completion_score(model, observed, held[:, ::-1], ['a','b'])
    np.testing.assert_array_equal(theta, model.transform(observed, ['a','b']))


def test_npmi_and_diversity_hand_calculation():
    reference = sparse.csr_matrix([[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1]])
    phi = np.array([[.45,.45,.05,.05], [.05,.05,.45,.45]])
    result = topic_metrics(phi, reference, 2)
    np.testing.assert_allclose(result['npmi'], 1)
    assert result['diversity'] == 1


def test_scalar_chain_diagnostics():
    rng = np.random.default_rng(1)
    chains = [rng.normal(size=2000) for _ in range(3)]
    assert split_rhat(chains) < 1.02
    assert effective_sample_size(chains[0]) > 1000
    shifted = [c+i*3 for i,c in enumerate(chains)]
    assert split_rhat(shifted) > 2
