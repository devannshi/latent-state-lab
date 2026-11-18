import numpy as np
import pytest
from numpy.testing import assert_allclose

from latent_state_lab.kalman import kalman_filter
from latent_state_lab.model import Parameters, simulate
from latent_state_lab.particle import (
    effective_sample_size,
    normalize_logweights,
    particle_filter,
    systematic_resample,
)


def test_logweight_normalization_is_stable_and_shift_invariant():
    logw = np.array([-10000.0, -10001.0, -11000.0])
    normalized, logz = normalize_logweights(logw)
    shifted, shifted_logz = normalize_logweights(logw + 17000)
    assert_allclose(np.exp(normalized).sum(), 1)
    assert_allclose(normalized, shifted, atol=2e-12)
    assert_allclose(shifted_logz - logz, 17000)


def test_zero_total_weight_is_reported():
    with pytest.raises(FloatingPointError):
        normalize_logweights([-np.inf, -np.inf])


def test_ess_extremes():
    assert_allclose(effective_sample_size(np.ones(64)), 64)
    assert_allclose(effective_sample_size([0, 0, 1, 0]), 1)


def test_systematic_resampling_uniform_and_degenerate():
    rng = np.random.default_rng(12)
    assert_allclose(systematic_resample(np.ones(100), rng), np.arange(100))
    assert_allclose(systematic_resample([0, 0, 1, 0], rng), 2)


def test_systematic_resampling_has_correct_expected_offspring():
    weights = np.array([0.08, 0.19, 0.31, 0.42])
    rng = np.random.default_rng(9)
    counts = np.zeros(4)
    for _ in range(4000):
        counts += np.bincount(systematic_resample(weights, rng), minlength=4)
    assert_allclose(counts / counts.sum(), weights, atol=0.008)


def test_optimal_initial_proposal_has_exact_likelihood_and_equal_weights():
    p = Parameters()
    result = particle_filter([0.73], p, 1000, np.random.default_rng(77), proposal="optimal")
    assert_allclose(result.loglikelihood, kalman_filter([0.73], p).loglikelihood, atol=1e-12)
    assert_allclose(result.final_weights, 0.001)
    assert_allclose(result.ess, 1000)


@pytest.mark.parametrize("proposal", ["bootstrap", "optimal"])
def test_particle_means_converge_to_exact_filter(proposal):
    p = Parameters()
    y = simulate(70, p, np.random.default_rng(81)).y
    exact = kalman_filter(y, p)
    errors = {}
    for count in (32, 4096):
        errors[count] = np.mean([np.mean((particle_filter(
            y, p, count, np.random.default_rng(seed), proposal=proposal
        ).mean - exact.mean)**2) for seed in range(6)])
    assert errors[4096] < errors[32] / 12
    assert errors[4096] < 0.002


@pytest.mark.parametrize("proposal", ["bootstrap", "optimal"])
def test_particle_likelihood_estimator_is_unbiased_within_monte_carlo_error(proposal):
    p = Parameters()
    y = simulate(12, p, np.random.default_rng(101)).y
    exact = kalman_filter(y, p).loglikelihood
    ratios = np.array([np.exp(particle_filter(y, p, 80, np.random.default_rng(s),
                                              proposal=proposal).loglikelihood - exact)
                       for s in range(160)])
    assert abs(ratios.mean() - 1) < 4 * ratios.std(ddof=1) / np.sqrt(len(ratios)) + 0.01


def test_missing_observations_have_zero_likelihood_increment():
    y = np.array([1.0, np.nan, np.nan, 0.3, np.nan])
    result = particle_filter(y, Parameters(), 200, np.random.default_rng(6))
    assert_allclose(result.loglikelihood_terms[np.isnan(y)], 0)
    assert_allclose(result.final_weights.sum(), 1)
    assert np.all(result.variance >= 0)


def test_all_missing_particle_filter_retains_prior_moments():
    p = Parameters()
    result = particle_filter(np.full(8, np.nan), p, 30000, np.random.default_rng(4))
    assert result.loglikelihood == 0
    assert_allclose(result.mean, 0, atol=0.02)
    assert_allclose(result.variance, p.stationary_variance, rtol=0.035)


def test_seeded_particle_filter_is_deterministic():
    y = np.linspace(-1, 2, 40)
    one = particle_filter(y, Parameters(), 250, np.random.default_rng(3))
    two = particle_filter(y, Parameters(), 250, np.random.default_rng(3))
    assert_allclose(one.mean, two.mean)
    assert_allclose(one.loglikelihood, two.loglikelihood)
    assert_allclose(one.resampled, two.resampled)


def test_mixture_observation_model_reduces_response_to_a_large_outlier():
    y = np.zeros(30)
    y[15] = 15
    p = Parameters()
    robust = particle_filter(y, p, 15000, np.random.default_rng(3),
                             outlier_probability=0.05, outlier_scale=10)
    gaussian = kalman_filter(y, p)
    assert abs(robust.mean[15]) < abs(gaussian.mean[15]) / 3


@pytest.mark.parametrize("kwargs", [{"n_particles": 1}, {"proposal": "wrong"},
                                      {"ess_fraction": 1.1},
                                      {"proposal": "optimal", "outlier_probability": 0.1}])
def test_invalid_particle_settings_rejected(kwargs):
    arguments = {"n_particles": 10, **kwargs}
    with pytest.raises(ValueError):
        particle_filter([1.0, 0.2], Parameters(), rng=np.random.default_rng(1), **arguments)
