"""Identified marginal MLE and a privileged complete-data parameter benchmark."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .kalman import nll_and_score
from .model import FloatArray, Parameters, observations


@dataclass(frozen=True)
class FitResult:
    parameters: Parameters
    lower: FloatArray
    upper: FloatArray
    covariance_transformed: FloatArray
    success: bool
    message: str
    nll: float
    iterations: int
    evaluations: int
    gradient_norm: float
    at_boundary: bool
    ci_valid: bool
    hessian_min_eigenvalue: float
    hessian_condition: float
    starts: tuple[dict, ...]


def transform(parameters: Parameters) -> FloatArray:
    return np.array([np.arctanh(parameters.phi), np.log(parameters.q), np.log(parameters.r)])


def inverse_transform(z: ArrayLike, alpha: float = 1.0) -> Parameters:
    z = np.asarray(z)
    return Parameters(float(np.tanh(z[0])), float(np.exp(z[1])), float(np.exp(z[2])), alpha)


def observed_information(z: ArrayLike, y: ArrayLike, alpha: float = 1.0) -> FloatArray:
    """Central differences of the analytic score (not an optimizer approximation)."""
    z = np.asarray(z, dtype=float)
    columns = []
    for j in range(3):
        step = 1e-4 * max(1.0, abs(z[j]))
        perturb = np.zeros(3)
        perturb[j] = step
        columns.append((nll_and_score(z + perturb, y, alpha)[1]
                        - nll_and_score(z - perturb, y, alpha)[1]) / (2 * step))
    information = np.column_stack(columns)
    return (information + information.T) / 2


def fit_mle(y: ArrayLike, *, alpha: float = 1.0, max_iterations: int = 300) -> FitResult:
    """Three-start exact marginal MLE with known alpha and stationary initialization.

    Wald intervals transform back from the unconstrained coordinates. Singular,
    non-positive, boundary, or failed fits have unavailable intervals, not silently
    regularized information. All starts and the winning fit are retained.
    """
    y = observations(y)
    if np.count_nonzero(np.isfinite(y)) < 5:
        raise ValueError("At least five observations are required for parameter estimation.")
    if not np.isfinite(alpha) or alpha == 0:
        raise ValueError("alpha is a known finite nonzero loading.")
    scale = max(float(np.nanmean(y**2)), 1e-4)
    log_low, log_high = np.log(scale * 1e-6), np.log(scale * 20)
    bounds = [(-np.arctanh(0.995), np.arctanh(0.995)),
              (log_low - np.log(alpha**2), log_high - np.log(alpha**2)),
              (log_low, log_high)]
    adjacent = np.isfinite(y[1:]) & np.isfinite(y[:-1])
    numerator = float(np.dot(y[1:][adjacent], y[:-1][adjacent]))
    denominator = max(float(np.dot(y[:-1][adjacent], y[:-1][adjacent])), 1e-10)
    rho = float(np.clip(numerator / denominator, -0.9, 0.9))
    initial_phis = (rho, 0.85 if rho >= 0 else -0.85, 0.15 if rho >= 0 else -0.15)
    fits = []
    for phi in initial_phis:
        initial = Parameters(phi, 0.5 * scale * (1 - phi**2) / alpha**2, 0.5 * scale, alpha)
        fits.append(minimize(nll_and_score, transform(initial), args=(y, alpha), jac=True,
                             method="L-BFGS-B", bounds=bounds,
                             options={"maxiter": max_iterations, "ftol": 1e-11, "gtol": 1e-6}))
    result = min(fits, key=lambda fit: fit.fun)
    at_boundary = any(min(z - lower, upper - z) < 1e-4
                      for z, (lower, upper) in zip(result.x, bounds))
    information = observed_information(result.x, y, alpha)
    eigenvalues = np.linalg.eigvalsh(information)
    condition = float(np.linalg.cond(information))
    ci_valid = bool(result.success and not at_boundary and eigenvalues[0] > 1e-7
                    and condition < 1e9)
    covariance = np.full((3, 3), np.nan)
    lower = upper = np.full(3, np.nan)
    if ci_valid:
        covariance = np.linalg.inv(information)
        radius = 1.959963984540054 * np.sqrt(np.diag(covariance))
        low, high = result.x - radius, result.x + radius
        lower = np.array([np.tanh(low[0]), np.exp(low[1]), np.exp(low[2])])
        upper = np.array([np.tanh(high[0]), np.exp(high[1]), np.exp(high[2])])
    starts = tuple({"nll": float(fit.fun), "success": bool(fit.success),
                    "iterations": int(fit.nit), "evaluations": int(fit.nfev),
                    "message": str(fit.message)} for fit in fits)
    return FitResult(inverse_transform(result.x, alpha), lower, upper, covariance,
                     bool(result.success), str(result.message), float(result.fun),
                     int(result.nit), sum(int(f.nfev) for f in fits),
                     float(np.max(np.abs(result.jac))), bool(at_boundary), ci_valid,
                     float(eigenvalues[0]), condition, starts)


def fit_oracle(x: ArrayLike, y: ArrayLike, *, alpha: float = 1.0) -> Parameters:
    """Complete-data conditional MLE: observe x and condition on x[0].

    This is privileged parameter estimation, not a realizable observation-only
    estimator. Its use in a subsequent filter does not grant that filter x_t.
    """
    x, y = observations(x), observations(y)
    if len(x) != len(y) or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("Oracle requires complete states and equal series lengths >= 3.")
    if np.count_nonzero(np.isfinite(y)) == 0:
        raise ValueError("Oracle requires at least one observed y.")
    denom = float(np.dot(x[:-1], x[:-1]))
    if denom <= 0:
        raise ValueError("State regression has zero information.")
    phi = float(np.clip(np.dot(x[:-1], x[1:]) / denom, -0.995, 0.995))
    q = max(float(np.mean((x[1:] - phi * x[:-1])**2)), 1e-12)
    r = max(float(np.nanmean((y - alpha * x)**2)), 1e-12)
    return Parameters(phi, q, r, alpha)
