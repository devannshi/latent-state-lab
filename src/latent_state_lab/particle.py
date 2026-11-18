"""Sequential importance sampling with adaptive systematic resampling.

    Weights and likelihood increments are accumulated in log space. The optimal
    proposal is optimal conditional on an ancestor; this is not an auxiliary PF.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from .model import FloatArray, Parameters, observations


def normalize_logweights(logweights: ArrayLike) -> tuple[FloatArray, float]:
    logweights = np.asarray(logweights, dtype=float)
    if logweights.ndim != 1 or len(logweights) == 0 or np.any(np.isnan(logweights)):
        raise ValueError("Log weights must be a nonempty 1-D array without NaNs.")
    normalizer = float(logsumexp(logweights))
    if not np.isfinite(normalizer):
        raise FloatingPointError("All particle weights vanished or overflowed.")
    return logweights - normalizer, normalizer


def systematic_resample(weights: ArrayLike, rng: np.random.Generator) -> NDArray[np.int64]:
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or len(weights) == 0 or np.any(~np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Weights must be nonnegative, finite, and have positive total mass.")
    cdf = np.cumsum(weights / weights.sum())
    cdf[-1] = 1.0
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(cdf, positions, side="right")


def effective_sample_size(weights: ArrayLike) -> float:
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or len(weights) == 0 or np.any(~np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Weights must be nonnegative finite values with positive mass.")
    weights = weights / weights.sum()
    return float(1 / np.dot(weights, weights))


def weighted_quantile(values: FloatArray, weights: FloatArray, probabilities: ArrayLike) -> FloatArray:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative[-1] = 1.0
    positions = np.searchsorted(cumulative, probabilities, side="left")
    return values[order[np.minimum(positions, len(values) - 1)]]


def normal_logpdf(value: ArrayLike, mean: ArrayLike, variance: float) -> FloatArray:
    return -0.5 * (np.log(2 * np.pi * variance) + (np.asarray(value) - mean)**2 / variance)


def observation_logpdf(
    y: float, x: FloatArray, parameters: Parameters,
    outlier_probability: float = 0.0, outlier_scale: float = 8.0,
) -> FloatArray:
    base = normal_logpdf(y, parameters.alpha * x, parameters.r)
    if outlier_probability == 0:
        return base
    wide = normal_logpdf(y, parameters.alpha * x, parameters.r * outlier_scale**2)
    if outlier_probability == 1:
        return wide
    return np.logaddexp(np.log1p(-outlier_probability) + base,
                        np.log(outlier_probability) + wide)


@dataclass(frozen=True)
class ParticleResult:
    mean: FloatArray
    variance: FloatArray
    lower: FloatArray
    upper: FloatArray
    ess: FloatArray
    resampled: NDArray[np.bool_]
    loglikelihood_terms: FloatArray
    final_particles: FloatArray
    final_weights: FloatArray

    @property
    def loglikelihood(self) -> float:
        return float(self.loglikelihood_terms.sum())

    def interval(self) -> tuple[FloatArray, FloatArray]:
        return self.lower, self.upper


def particle_filter(
    y: ArrayLike,
    parameters: Parameters,
    n_particles: int,
    rng: np.random.Generator,
    *,
    proposal: str = "bootstrap",
    ess_fraction: float = 0.5,
    outlier_probability: float = 0.0,
    outlier_scale: float = 8.0,
) -> ParticleResult:
    """Approximate filtered marginal distributions and the data likelihood.

    A Gaussian-mixture observation model is available for the bootstrap proposal.
    Setting mixture parameters to their simulation truth is a known-model stress
    comparison; it does not estimate contamination from observations.
    """
    y = observations(y)
    if not isinstance(n_particles, (int, np.integer)) or n_particles < 2:
        raise ValueError("n_particles must be an integer >= 2.")
    if proposal not in {"bootstrap", "optimal"}:
        raise ValueError("proposal must be 'bootstrap' or 'optimal'.")
    if not 0 <= ess_fraction <= 1 or not 0 <= outlier_probability <= 1:
        raise ValueError("ESS fraction and contamination probability must lie in [0, 1].")
    if not np.isfinite(outlier_scale) or outlier_scale < 1:
        raise ValueError("outlier_scale must be finite and >= 1.")
    if proposal == "optimal" and outlier_probability != 0:
        raise ValueError("The Gaussian optimal proposal is not implemented for mixture noise.")
    n = len(y)
    means, variances, lower, upper, ess = (np.empty(n) for _ in range(5))
    resampled = np.zeros(n, dtype=bool)
    terms = np.zeros(n)
    logweights = np.full(n_particles, -np.log(n_particles))
    particles = np.zeros(n_particles)
    for t, value in enumerate(y):
        prior_variance = parameters.stationary_variance if t == 0 else parameters.q
        prior_mean = np.zeros(n_particles) if t == 0 else parameters.phi * particles
        if np.isnan(value):
            particles = prior_mean + np.sqrt(prior_variance) * rng.normal(size=n_particles)
        elif proposal == "bootstrap":
            particles = prior_mean + np.sqrt(prior_variance) * rng.normal(size=n_particles)
            increments = observation_logpdf(value, particles, parameters,
                                            outlier_probability, outlier_scale)
            logweights, terms[t] = normalize_logweights(logweights + increments)
        else:
            predictive_variance = parameters.alpha**2 * prior_variance + parameters.r
            gain = prior_variance * parameters.alpha / predictive_variance
            proposal_mean = prior_mean + gain * (value - parameters.alpha * prior_mean)
            proposal_variance = prior_variance * parameters.r / predictive_variance
            particles = proposal_mean + np.sqrt(proposal_variance) * rng.normal(size=n_particles)
            increments = normal_logpdf(value, parameters.alpha * prior_mean, predictive_variance)
            logweights, terms[t] = normalize_logweights(logweights + increments)
        weights = np.exp(logweights)
        means[t] = np.dot(weights, particles)
        variances[t] = np.dot(weights, (particles - means[t])**2)
        lower[t], upper[t] = weighted_quantile(particles, weights, [0.025, 0.975])
        ess[t] = 1 / np.dot(weights, weights)
        if ess[t] < ess_fraction * n_particles and t < n - 1:
            particles = particles[systematic_resample(weights, rng)]
            logweights.fill(-np.log(n_particles))
            resampled[t] = True
    return ParticleResult(means, variances, lower, upper, ess, resampled, terms,
                          particles.copy(), np.exp(logweights))
