import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import multivariate_normal

from latent_state_lab.estimation import fit_mle, fit_oracle, transform
from latent_state_lab.kalman import kalman_filter, nll_and_score, rts_smoother
from latent_state_lab.model import Parameters, observation_covariance, rescale_latent, simulate


@pytest.mark.parametrize("phi,alpha", [(0.85, 1), (-0.7, 1.5), (0, -2), (0.98, 0.3)])
def test_likelihood_matches_dense_gaussian(phi, alpha):
    p = Parameters(phi, 0.12, 0.4, alpha)
    y = np.random.default_rng(15).normal(size=25)
    expected = multivariate_normal.logpdf(y, cov=observation_covariance(len(y), p))
    assert_allclose(kalman_filter(y, p).loglikelihood, expected, rtol=1e-12)


def test_missing_likelihood_matches_marginal_dense_gaussian():
    p = Parameters()
    y = np.array([0.4, np.nan, np.nan, -0.7, 0.6, np.nan])
    mask = np.isfinite(y)
    covariance = observation_covariance(len(y), p)[np.ix_(mask, mask)]
    assert_allclose(kalman_filter(y, p).loglikelihood,
                    multivariate_normal.logpdf(y[mask], cov=covariance), rtol=1e-12)


@pytest.mark.parametrize("missing", [False, True])
def test_smoother_matches_joint_gaussian_conditioning(missing):
    p = Parameters(-0.5, 0.2, 0.3, 1.7)
    y = np.random.default_rng(6).normal(size=12)
    if missing:
        y[[1, 4, 9]] = np.nan
    mask = np.isfinite(y)
    indices = np.arange(len(y))
    state_cov = p.stationary_variance * p.phi**np.abs(indices[:, None] - indices[None, :])
    cross = p.alpha * state_cov[:, mask]
    obs_cov = observation_covariance(len(y), p)[np.ix_(mask, mask)]
    mean = cross @ np.linalg.solve(obs_cov, y[mask])
    variance = np.diag(state_cov - cross @ np.linalg.solve(obs_cov, cross.T))
    actual = rts_smoother(kalman_filter(y, p), p)
    assert_allclose(actual.mean, mean, atol=1e-12)
    assert_allclose(actual.variance, variance, atol=1e-12)


def test_filter_matches_prefix_gaussian_conditioning():
    p = Parameters()
    y = np.random.default_rng(8).normal(size=12)
    result = kalman_filter(y, p)
    for t in range(len(y)):
        cross = p.alpha * p.stationary_variance * p.phi**np.arange(t, -1, -1)
        covariance = observation_covariance(t + 1, p)
        assert_allclose(result.mean[t], cross @ np.linalg.solve(covariance, y[:t + 1]))
        assert_allclose(result.variance[t], p.stationary_variance
                        - cross @ np.linalg.solve(covariance, cross))


@pytest.mark.parametrize("phi,alpha,missing", [(0.85, 1, False), (-0.75, -1.6, False),
                                              (0.95, 2, True), (0, 1, True)])
def test_analytic_score_matches_finite_differences(phi, alpha, missing):
    p = Parameters(phi, 0.2, 0.45, alpha)
    y = simulate(60, p, np.random.default_rng(31)).y
    if missing:
        y[::4] = np.nan
    z = transform(p)
    value, gradient = nll_and_score(z, y, alpha)
    expected = approx_derivative(lambda theta: -kalman_filter(
        y, Parameters(np.tanh(theta[0]), np.exp(theta[1]), np.exp(theta[2]), alpha)
    ).loglikelihood, z).ravel()
    assert_allclose(value, -kalman_filter(y, p).loglikelihood)
    assert_allclose(gradient, expected, rtol=3e-6, atol=1e-7)


def test_all_missing_filter_preserves_stationary_marginals():
    p = Parameters()
    result = kalman_filter(np.full(25, np.nan), p)
    assert_allclose(result.mean, 0)
    assert_allclose(result.variance, p.stationary_variance)
    assert result.loglikelihood == 0


def test_smoothing_uses_future_information_and_reduces_conditional_variance():
    p = Parameters()
    y = simulate(40, p, np.random.default_rng(88)).y
    filtered = kalman_filter(y, p)
    smoothed = rts_smoother(filtered, p)
    assert_allclose(smoothed.mean[-1], filtered.mean[-1])
    assert np.all(smoothed.variance <= filtered.variance + 1e-12)
    assert not np.allclose(smoothed.mean[:-1], filtered.mean[:-1])


@pytest.mark.parametrize("scale", [0.2, 3.0, -2.0])
def test_observational_equivalence_under_latent_rescaling(scale):
    p = Parameters()
    rescaled = rescale_latent(p, scale)
    y = np.random.default_rng(53).normal(size=80)
    first, second = kalman_filter(y, p), kalman_filter(y, rescaled)
    assert_allclose(first.loglikelihood, second.loglikelihood, atol=1e-11)
    assert_allclose(second.mean, scale * first.mean, atol=1e-12)
    assert_allclose(second.variance, scale**2 * first.variance, atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"phi": 1}, {"phi": -1}, {"q": 0}, {"r": -1},
                                      {"alpha": 0}, {"phi": np.nan}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        Parameters(**kwargs)


def test_seed_reproduces_dataset_and_missing_mask():
    kwargs = dict(n=80, parameters=Parameters(), missing_probability=0.3,
                  outlier_probability=0.1)
    one = simulate(rng=np.random.default_rng(11), **kwargs)
    two = simulate(rng=np.random.default_rng(11), **kwargs)
    assert_allclose(one.x, two.x)
    assert_allclose(one.y, two.y, equal_nan=True)
    assert np.any(one.contaminated) and np.any(~one.observed)


def test_mle_recovers_parameters_on_long_seeded_sample():
    p = Parameters(0.85, 0.16, 0.49)
    y = simulate(2500, p, np.random.default_rng(917)).y
    fit = fit_mle(y)
    assert fit.success and fit.ci_valid
    assert fit.gradient_norm < 0.01
    assert abs(fit.parameters.phi - p.phi) < 0.06
    assert abs(fit.parameters.q - p.q) < 0.065
    assert abs(fit.parameters.r - p.r) < 0.08
    assert fit.nll <= min(item["nll"] for item in fit.starts) + 1e-10


def test_oracle_conditional_mle_matches_regression():
    p = Parameters()
    data = simulate(1000, p, np.random.default_rng(54))
    fit = fit_oracle(data.x, data.y)
    phi = np.dot(data.x[1:], data.x[:-1]) / np.dot(data.x[:-1], data.x[:-1])
    assert_allclose(fit.phi, phi)
    assert_allclose(fit.q, np.mean((data.x[1:] - phi * data.x[:-1])**2))
    assert_allclose(fit.r, np.mean((data.y - data.x)**2))


def test_parameter_estimation_requires_observations():
    with pytest.raises(ValueError):
        fit_mle([1, np.nan, 2])
    with pytest.raises(ValueError):
        fit_oracle([1, 2, 3], [np.nan] * 3)


@pytest.mark.parametrize("invalid", [[], [[1, 2]], [1, np.inf]])
def test_invalid_observations_rejected(invalid):
    with pytest.raises(ValueError):
        kalman_filter(invalid, Parameters())
