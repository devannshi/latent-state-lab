"""Exact and particle inference for identifiable scalar state-space models."""

from .estimation import fit_mle, fit_oracle
from .kalman import kalman_filter, rts_smoother
from .model import Parameters, simulate
from .particle import particle_filter

__all__ = [
    "Parameters", "simulate", "kalman_filter", "rts_smoother", "fit_mle",
    "fit_oracle", "particle_filter",
]
__version__ = "0.1.0"
