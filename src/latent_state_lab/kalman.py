"""Exact scalar Gaussian filtering, marginal likelihood, and RTS smoothing."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .model import FloatArray, Parameters, observations

LOG_2PI = float(np.log(2 * np.pi))


@dataclass(frozen=True)
class FilterResult:
    mean: FloatArray
    variance: FloatArray
    predicted_mean: FloatArray
    predicted_variance: FloatArray
    innovation: FloatArray
    innovation_variance: FloatArray
    loglikelihood_terms: FloatArray

    @property
    def loglikelihood(self) -> float:
        return float(np.sum(self.loglikelihood_terms))

    def interval(self, z: float = 1.959963984540054) -> tuple[FloatArray, FloatArray]:
        radius = z * np.sqrt(self.variance)
        return self.mean - radius, self.mean + radius


@dataclass(frozen=True)
class SmootherResult:
    mean: FloatArray
    variance: FloatArray

    def interval(self, z: float = 1.959963984540054) -> tuple[FloatArray, FloatArray]:
        radius = z * np.sqrt(self.variance)
        return self.mean - radius, self.mean + radius


def kalman_filter(y: ArrayLike, parameters: Parameters) -> FilterResult:
    """Compute p(x_t | y_0:t, theta), skipping updates for missing observations."""
    y = observations(y)
    n = len(y)
    mean, variance, pm, pv = (np.empty(n) for _ in range(4))
    innovation = np.full(n, np.nan)
    innovation_variance = np.full(n, np.nan)
    terms = np.zeros(n)
    phi, q, r, alpha = parameters.phi, parameters.q, parameters.r, parameters.alpha
    for t, value in enumerate(y):
        pm[t] = 0.0 if t == 0 else phi * mean[t - 1]
        pv[t] = parameters.stationary_variance if t == 0 else phi**2 * variance[t - 1] + q
        if np.isnan(value):
            mean[t], variance[t] = pm[t], pv[t]
            continue
        innovation[t] = value - alpha * pm[t]
        innovation_variance[t] = alpha**2 * pv[t] + r
        gain = pv[t] * alpha / innovation_variance[t]
        mean[t] = pm[t] + gain * innovation[t]
        # Algebraically equivalent to (1-K*alpha)*P, but avoids cancellation.
        variance[t] = pv[t] * r / innovation_variance[t]
        terms[t] = -0.5 * (LOG_2PI + np.log(innovation_variance[t])
                           + innovation[t]**2 / innovation_variance[t])
    return FilterResult(mean, variance, pm, pv, innovation, innovation_variance, terms)


def rts_smoother(filtered: FilterResult, parameters: Parameters) -> SmootherResult:
    """Compute p(x_t | y_0:T-1, theta); these estimates use future observations."""
    mean, variance = filtered.mean.copy(), filtered.variance.copy()
    for t in range(len(mean) - 2, -1, -1):
        gain = filtered.variance[t] * parameters.phi / filtered.predicted_variance[t + 1]
        mean[t] += gain * (mean[t + 1] - filtered.predicted_mean[t + 1])
        variance[t] += gain**2 * (variance[t + 1] - filtered.predicted_variance[t + 1])
    return SmootherResult(mean, np.maximum(variance, 0.0))


def nll_and_score(z: ArrayLike, y: ArrayLike, alpha: float = 1.0) -> tuple[float, FloatArray]:
    """Negative exact likelihood and analytic score in (atanh(phi), log(q), log(r)).

    Scalar sensitivity recursions avoid finite-difference objective evaluations
    inside optimization. The stationary initial variance is differentiated too.
    """
    z = np.asarray(z, dtype=float)
    phi, q, r = np.tanh(z[0]), np.exp(z[1]), np.exp(z[2])
    p = q / (1 - phi * phi)
    dp0, dp1, dp2 = 2 * phi * q / (1 - phi * phi)**2, p, 0.0
    # Derivatives are in (phi, log q, log r) until the final chain rule.
    m = dm0 = dm1 = dm2 = 0.0
    g0 = g1 = g2 = total = 0.0
    aa = alpha * alpha
    for t, value in enumerate(y):
        if t:
            dp0, dp1, dp2 = (2 * phi * p + phi * phi * dp0,
                              phi * phi * dp1 + q, phi * phi * dp2)
            p = phi * phi * p + q
            dm0, dm1, dm2 = m + phi * dm0, phi * dm1, phi * dm2
            m = phi * m
        if np.isnan(value):
            continue
        v = value - alpha * m
        dv0, dv1, dv2 = -alpha * dm0, -alpha * dm1, -alpha * dm2
        s = aa * p + r
        ds0, ds1, ds2 = aa * dp0, aa * dp1, aa * dp2 + r
        total += 0.5 * (LOG_2PI + np.log(s) + v * v / s)
        a, b = 0.5 * (1 / s - v * v / (s * s)), v / s
        g0 += a * ds0 + b * dv0
        g1 += a * ds1 + b * dv1
        g2 += a * ds2 + b * dv2
        k = alpha * p / s
        dk0 = alpha * (dp0 * s - p * ds0) / (s * s)
        dk1 = alpha * (dp1 * s - p * ds1) / (s * s)
        dk2 = alpha * (dp2 * s - p * ds2) / (s * s)
        dm0, dm1, dm2 = (dm0 + dk0 * v + k * dv0,
                          dm1 + dk1 * v + k * dv1, dm2 + dk2 * v + k * dv2)
        m += k * v
        dp0, dp1, dp2 = (r * (dp0 * s - p * ds0) / (s * s),
                          r * (dp1 * s - p * ds1) / (s * s),
                          r * (dp2 * s + p * s - p * ds2) / (s * s))
        p = p * r / s
    return float(total), np.array([g0 * (1 - phi * phi), g1, g2])
