"""Trial-level uncertainty: time points within a series are not IID replicates."""

import numpy as np
import pandas as pd
from scipy import stats

from .model import FloatArray


def state_metrics(x: FloatArray, result) -> dict[str, float]:
    lower, upper = result.interval()
    error = result.mean - x
    return {"mse": float(np.mean(error**2)), "rmse": float(np.sqrt(np.mean(error**2))),
            "coverage": float(np.mean((lower <= x) & (x <= upper))),
            "interval_width": float(np.mean(upper - lower))}


def mean_interval(values) -> dict[str, float]:
    """A Student-t Monte Carlo interval across independent trial-level outcomes."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": np.nan, "mcse": np.nan, "lower": np.nan, "upper": np.nan}
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    radius = stats.t.ppf(0.975, n - 1) * se if n > 1 else np.nan
    return {"n": n, "mean": mean, "mcse": se, "lower": mean - radius, "upper": mean + radius}


def summarize_states(frame: pd.DataFrame, group_columns=("method",)) -> pd.DataFrame:
    records = []
    for keys, group in frame.groupby(list(group_columns), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in ("mse", "rmse", "coverage", "interval_width", "seconds"):
            if metric in group:
                records.append({**base, "metric": metric, **mean_interval(group[metric])})
    return pd.DataFrame(records)


def summarize_parameters(frame: pd.DataFrame, truth: dict) -> pd.DataFrame:
    rows = []
    for (method, parameter), group in frame.groupby(["method", "parameter"], sort=True):
        error = group.estimate.to_numpy() - truth[parameter]
        interval = mean_interval(error)
        valid = group.ci_valid.astype(bool).to_numpy()
        covered = ((group.lower <= truth[parameter]) & (truth[parameter] <= group.upper)).to_numpy()
        test = stats.ttest_1samp(error, 0)
        coverage = mean_interval(covered[valid].astype(float))
        rows.append({"method": method, "parameter": parameter, "truth": truth[parameter],
                     "n": len(group), "mean": float(group.estimate.mean()),
                     "bias": interval["mean"], "bias_mcse": interval["mcse"],
                     "bias_lower": interval["lower"], "bias_upper": interval["upper"],
                     "rmse": float(np.sqrt(np.mean(error**2))),
                     "bias_t": float(test.statistic), "bias_p_unadjusted": float(test.pvalue),
                     "ci_valid_count": int(valid.sum()),
                     "ci_available_fraction": float(valid.mean()),
                     "coverage_given_valid_ci": coverage["mean"],
                     "coverage_mcse": coverage["mcse"],
                     "coverage_all_trials_invalid_as_miss": float((covered & valid).sum() / len(group))})
    result = pd.DataFrame(rows)
    # Holm adjustment across the complete family of reported parameter-bias tests.
    pvalues = result.bias_p_unadjusted.to_numpy()
    order = np.argsort(pvalues)
    adjusted = np.minimum(1.0, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    result["bias_p_holm"] = np.empty(len(order))
    result.loc[order, "bias_p_holm"] = adjusted
    return result
