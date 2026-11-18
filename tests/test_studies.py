
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from latent_state_lab.experiments import StudyConfig, seed_for
from latent_state_lab.metrics import mean_interval, summarize_parameters
from latent_state_lab.stochastic_volatility import (
    SVParameters,
    grid_filter,
    simulate_sv,
    sv_particle_filter,
)


def test_trial_and_component_seeds_are_distinct_and_reproducible():
    seeds = [seed_for(123, study, trial, component)
             for study in range(3) for trial in range(8) for component in range(3)]
    assert len(set(seeds)) == len(seeds)
    assert seed_for(123, 2, 4, 1) == seed_for(123, 2, 4, 1)


def test_full_profile_has_thousand_trials_and_predeclared_particle_grid():
    config = StudyConfig.for_profile("full")
    assert config.trials >= 1000
    assert config.particle_counts == (32, 64, 128, 256, 512, 1024, 2048)


def test_monte_carlo_uncertainty_uses_trial_count():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    interval = mean_interval(values)
    assert interval["n"] == 4
    assert_allclose(interval["mcse"], values.std(ddof=1) / np.sqrt(4))
    assert interval["lower"] < values.mean() < interval["upper"]


def test_invalid_parameter_intervals_are_retained_as_misses():
    frame = pd.DataFrame({"method": ["test"] * 4, "parameter": ["q"] * 4,
                          "estimate": [0.2, 0.3, 0.5, 0.1], "ci_valid": [True, True, False, True],
                          "lower": [0.1, 0.1, 0.1, 0.05], "upper": [0.3, 0.4, 0.3, 0.15]})
    row = summarize_parameters(frame, {"q": 0.2}).iloc[0]
    assert row.n == 4
    assert row.ci_valid_count == 3
    assert_allclose(row.coverage_given_valid_ci, 2 / 3)
    assert_allclose(row.coverage_all_trials_invalid_as_miss, 1 / 2)


def test_sv_grid_is_stable_to_resolution_refinement():
    p = SVParameters()
    _, y = simulate_sv(50, p, np.random.default_rng(8))
    coarse = grid_filter(y, p, grid_size=201)
    fine = grid_filter(y, p, grid_size=401)
    assert_allclose(coarse.mean, fine.mean, atol=2e-5)
    assert_allclose(coarse.loglikelihood, fine.loglikelihood, atol=2e-4)
    assert fine.boundary_mass < 1e-8


def test_sv_particles_approximate_numerical_reference():
    p = SVParameters()
    _, y = simulate_sv(50, p, np.random.default_rng(35))
    grid = grid_filter(y, p)
    particle = sv_particle_filter(y, p, 15000, np.random.default_rng(13))
    assert np.mean((grid.mean - particle.mean)**2) < 0.006
    assert np.all(particle.variance > 0)


def test_sv_missing_data_is_supported():
    p = SVParameters()
    result = grid_filter(np.full(12, np.nan), p)
    assert_allclose(result.mean, p.mu, atol=1e-7)
    assert_allclose(result.variance, p.stationary_variance, atol=1e-7)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        StudyConfig.for_profile("unknown")
