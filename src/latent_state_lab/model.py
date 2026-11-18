"""A stationary AR(1) latent process, with explicit scale identification.

``q`` and ``r`` denote variances throughout, never standard deviations.
Time starts at zero: x[0] is drawn from the stationary marginal.
"""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Parameters:
    phi: float = 0.85
    q: float = 0.16
    r: float = 0.49
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) for v in (self.phi, self.q, self.r, self.alpha)):
            raise ValueError("All parameters must be finite.")
        if abs(self.phi) >= 1 or self.q <= 0 or self.r <= 0 or self.alpha == 0:
            raise ValueError("Require |phi| < 1, q > 0, r > 0, and nonzero known alpha.")

    @property
    def stationary_variance(self) -> float:
        return self.q / (1 - self.phi**2)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Simulation:
    x: FloatArray
    y: FloatArray
    observed: NDArray[np.bool_]
    contaminated: NDArray[np.bool_]


def observations(y: ArrayLike) -> FloatArray:
    """Validate a 1-D nonempty series; NaNs explicitly represent missing values."""
    array = np.asarray(y, dtype=float)
    if array.ndim != 1 or len(array) == 0 or np.any(np.isinf(array)):
        raise ValueError("y must be a nonempty 1-D array with finite values or NaN.")
    return array


def simulate(
    n: int,
    parameters: Parameters,
    rng: np.random.Generator,
    *,
    missing_probability: float = 0.0,
    outlier_probability: float = 0.0,
    outlier_scale: float = 8.0,
) -> Simulation:
    """Generate independent Gaussian-mixture observation errors when requested.

    Outlier scale multiplies the observation standard deviation. Missingness is
    independent of both state and observation; it is not informative missingness.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    if not 0 <= missing_probability <= 1 or not 0 <= outlier_probability <= 1:
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.isfinite(outlier_scale) or outlier_scale < 1:
        raise ValueError("outlier_scale must be finite and at least one.")
    x = np.empty(n)
    x[0] = rng.normal(scale=np.sqrt(parameters.stationary_variance))
    shocks = rng.normal(scale=np.sqrt(parameters.q), size=n - 1)
    for t in range(1, n):
        x[t] = parameters.phi * x[t - 1] + shocks[t - 1]
    contaminated = rng.random(n) < outlier_probability
    scales = np.where(contaminated, outlier_scale, 1.0) * np.sqrt(parameters.r)
    y = parameters.alpha * x + rng.normal(size=n) * scales
    observed = rng.random(n) >= missing_probability
    y[~observed] = np.nan
    return Simulation(x, y, observed, contaminated)


def observation_covariance(n: int, parameters: Parameters) -> FloatArray:
    """Dense stationary observation covariance, for independent likelihood tests."""
    lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    covariance = parameters.alpha**2 * parameters.stationary_variance * parameters.phi**lag
    return covariance + parameters.r * np.eye(n)


def rescale_latent(parameters: Parameters, scale: float) -> Parameters:
    """The unidentifiable transformation x'=c*x, alpha'=alpha/c, q'=c²q."""
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Latent scale must be finite and nonzero.")
    return Parameters(parameters.phi, parameters.q * scale**2, parameters.r,
                      parameters.alpha / scale)
