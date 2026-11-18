"""A genuinely non-Gaussian observation stress test with a numerical reference.

    h_t = mu + phi*(h_(t-1)-mu) + eta_t; y_t = exp(h_t/2)*epsilon_t.
    The grid filter is deterministic quadrature on a truncated domain, not exact
    inference. The study checks a second resolution and records boundary mass.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .model import FloatArray, observations
from .particle import normalize_logweights, systematic_resample, weighted_quantile


@dataclass(frozen=True)
class SVParameters:
    mu: float = -1.0
    phi: float = 0.95
    q: float = 0.08

    def __post_init__(self):
        if not np.all(np.isfinite([self.mu, self.phi, self.q])) or abs(self.phi) >= 1 or self.q <= 0:
            raise ValueError("SV requires finite parameters, |phi| < 1, and q > 0.")

    @property
    def stationary_variance(self):
        return self.q / (1 - self.phi**2)


@dataclass(frozen=True)
class SVResult:
    mean: FloatArray
    variance: FloatArray
    lower: FloatArray
    upper: FloatArray
    loglikelihood: float
    boundary_mass: float = 0.0

    def interval(self):
        return self.lower, self.upper


def simulate_sv(n: int, parameters: SVParameters, rng: np.random.Generator):
    if n < 1:
        raise ValueError("n must be positive.")
    h = np.empty(n)
    h[0] = parameters.mu + rng.normal(scale=np.sqrt(parameters.stationary_variance))
    for t in range(1, n):
        h[t] = (parameters.mu + parameters.phi * (h[t - 1] - parameters.mu)
                + rng.normal(scale=np.sqrt(parameters.q)))
    y = np.exp(h / 2) * rng.normal(size=n)
    return h, y


def _log_observation(y, h):
    return -0.5 * (np.log(2 * np.pi) + h + y**2 * np.exp(-h))


def grid_filter(y, parameters: SVParameters, *, grid_size: int = 401,
                limits: tuple[float, float] = (-10.0, 8.0)) -> SVResult:
    y = observations(y)
    if grid_size < 50 or limits[0] >= limits[1]:
        raise ValueError("Use at least 50 grid points and increasing limits.")
    grid = np.linspace(*limits, grid_size)
    spacing = grid[1] - grid[0]
    transition_mean = parameters.mu + parameters.phi * (grid - parameters.mu)
    transition = np.exp(-0.5 * (grid[:, None] - transition_mean[None, :])**2 / parameters.q)
    transition *= spacing / np.sqrt(2 * np.pi * parameters.q)
    weights = np.exp(-0.5 * (grid - parameters.mu)**2 / parameters.stationary_variance)
    weights /= weights.sum()
    means, variances, low, high = (np.empty(len(y)) for _ in range(4))
    loglikelihood, maximum_boundary_mass = 0.0, 0.0
    for t, value in enumerate(y):
        if t:
            weights = transition @ weights
            # Do not normalize away domain leakage: it belongs in the likelihood.
        if not np.isnan(value):
            logweights = np.log(np.maximum(weights, np.finfo(float).tiny)) + _log_observation(value, grid)
            normalizer = logsumexp(logweights)
            weights = np.exp(logweights - normalizer)
            loglikelihood += normalizer
        else:
            weights /= weights.sum()
        means[t] = np.dot(grid, weights)
        variances[t] = np.dot((grid - means[t])**2, weights)
        low[t], high[t] = weighted_quantile(grid, weights, [0.025, 0.975])
        maximum_boundary_mass = max(maximum_boundary_mass, float(weights[:3].sum() + weights[-3:].sum()))
    return SVResult(means, variances, low, high, float(loglikelihood), maximum_boundary_mass)


def sv_particle_filter(y, parameters: SVParameters, n_particles: int,
                       rng: np.random.Generator) -> SVResult:
    y = observations(y)
    if n_particles < 2:
        raise ValueError("At least two particles required.")
    particles = parameters.mu + np.sqrt(parameters.stationary_variance) * rng.normal(size=n_particles)
    logweights = np.full(n_particles, -np.log(n_particles))
    means, variances, low, high = (np.empty(len(y)) for _ in range(4))
    loglikelihood = 0.0
    for t, value in enumerate(y):
        if t:
            particles = (parameters.mu + parameters.phi * (particles - parameters.mu)
                         + np.sqrt(parameters.q) * rng.normal(size=n_particles))
        if not np.isnan(value):
            logweights, increment = normalize_logweights(logweights + _log_observation(value, particles))
            loglikelihood += increment
        weights = np.exp(logweights)
        means[t] = np.dot(weights, particles)
        variances[t] = np.dot(weights, (particles - means[t])**2)
        low[t], high[t] = weighted_quantile(particles, weights, [0.025, 0.975])
        if 1 / np.dot(weights, weights) < n_particles / 2 and t < len(y) - 1:
            particles = particles[systematic_resample(weights, rng)]
            logweights.fill(-np.log(n_particles))
    return SVResult(means, variances, low, high, float(loglikelihood))
