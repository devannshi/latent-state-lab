"""Generate numerical claims from saved results; no hand-entered performance numbers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _table(headers, rows):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}" if np.isfinite(value) else "unavailable"
        return str(value)
    return ("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
            + "\n".join("| " + " | ".join(cell(value) for value in row) + " |" for row in rows))


def write_report(output: Path):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["config"]
    state = pd.read_csv(output / "state_summary.csv")
    parameters = pd.read_csv(output / "parameter_summary.csv")
    paired = pd.read_csv(output / "paired_risk_differences.csv")
    diagnostics = [json.loads(line) for line in (output / "optimizer_diagnostics.jsonl").read_text().splitlines()]
    holdout = pd.read_csv(output / "holdout_trials.csv")
    stress = pd.read_csv(output / "stress_summary.csv")
    sv = pd.read_csv(output / "sv_summary.csv")
    grid = json.loads((output / "sv_grid_check.json").read_text())
    ridge = pd.read_csv(output / "identifiability_ridge.csv")
    bootstrap = json.loads((output / "bootstrap_summary.json").read_text())
    state_pivot = state.pivot(index="method", columns="metric", values="mean")
    state_rows = [[method, *[row[column] for column in ("mse", "rmse", "coverage", "interval_width")]]
                  for method, row in state_pivot.iterrows()]
    parameter_rows = [[row.method, row.parameter, row.bias, row.bias_mcse, row.rmse,
                       row.coverage_given_valid_ci, row.coverage_all_trials_invalid_as_miss,
                       f"{int(row.ci_valid_count)}/{int(row.n)}", f"{row.bias_p_holm:.3g}"]
                      for row in parameters.itertuples()]
    pair_rows = [[row.comparison, row.mean, row.mcse, row.lower, row.upper]
                 for row in paired.itertuples()]
    holdout_rows = [[method, group.mse.mean(), group.coverage.mean(),
                     group.mean_negative_log_predictive_density.mean()]
                    for method, group in holdout.groupby("method")]
    stress_pivot = stress.pivot(index=["scenario", "method"], columns="metric", values="mean")
    stress_rows = [[scenario, method, row.mse, row.coverage, row.interval_width]
                   for (scenario, method), row in stress_pivot.iterrows()]
    sv_pivot = sv.pivot(index="method", columns="metric", values="mean")
    sv_rows = [[method, row.mse, row.coverage, row.seconds * 1000] for method, row in sv_pivot.iterrows()]
    failed = sum(not row["success"] for row in diagnostics)
    boundary = sum(row["at_boundary"] for row in diagnostics)
    valid = sum(row["ci_valid"] for row in diagnostics)
    kf_risk = state_pivot.loc["KF_known", "mse"]
    smoother_risk = state_pivot.loc["RTS_known", "mse"]
    text = f"""# How Reliably Can We Recover Hidden States from Noisy Data?

## Reproducible experiment report

This report is generated from the CSV and JSON files in this directory. It describes
the **{config['profile']}** profile: **{config['trials']:,} independent core trials**, each
of length **{config['length']}**, with **{config['particles']} particles** per core PF.
The master seed is `{config['seed']}`. Every data and particle stream is independently
derived from an explicit trial identifier. All trials remain in the analysis.

## What this experiment establishes

The observation loading is fixed at α=1, so latent-state scale is identified. Under
the correctly specified linear Gaussian model, the exact Kalman conditional mean
is the minimum expected squared-error estimator using the same observations and
known parameters. A particle filter approximates this solution; it cannot improve
its population risk merely by adding simulation. Smoothing and oracle parameter
estimation use different information and are identified separately below.

The measured known-parameter filtering MSE is **{kf_risk:.4f}**, versus
**{smoother_risk:.4f}** for smoothing with future observations. These are synthetic
inference results, not trading returns or evidence of market predictability.

![Known-parameter state recovery](figures/01_state_recovery.png)

## State recovery and conditional interval coverage

{_table(['Method', 'Mean MSE', 'Mean trial RMSE', '95% state coverage', 'Mean interval width'], state_rows)}

MSE is averaged over time within a trial and then across independent trials. Mean
trial RMSE is a different estimand from the square root of pooled MSE. State coverage
is the fraction of latent states inside pointwise intervals, averaged across trials;
these are not simultaneous confidence bands. Gaussian methods use exact conditional
Gaussian intervals for the supplied parameters. PF intervals are weighted empirical
quantiles. For full-sample MLE, those intervals **ignore parameter-estimation
uncertainty** and their empirical coverage is the diagnostic of that omission.

`KF_full_sample_MLE` performs forward state updates after estimating parameters on
the entire series. It is retrospective empirical Bayes, not a real-time trading
filter. `KF_oracle_parameters` estimates parameters using the latent states, then
filters using observations only. It is a privileged benchmark, not an achievable
strategy or a guaranteed lower bound. RTS explicitly uses future observations.
An oracle parameter fit can even beat the known-parameter KF on this experiment:
its fitted parameters carry information from the latent path, so the two estimators
do not have the same information set.

![State risk with Monte Carlo intervals](figures/02_state_risk.png)

### Paired risk differences

{_table(['Comparison', 'Mean MSE difference', 'MCSE', '95% MC lower', '95% MC upper'], pair_rows)}

Pairing uses the same simulated dataset for each method. Monte Carlo standard errors
and Student-t intervals use **independent trial-level differences**, never individual
time points as if they were independent samples. Finite experiments can show a
negative PF-minus-KF difference by chance; expected optimality is a mathematical
statement, and approximation error to the exact KF is measured separately.

## Parameter estimation: bias, RMSE, and interval reliability

{_table(['Estimator', 'Parameter', 'Bias', 'Bias MCSE', 'RMSE', 'Coverage if available', 'Coverage all trials', 'Valid intervals', 'Holm bias p'], parameter_rows)}

Marginal MLE uses a stationary initial distribution, three deterministic initial
values, an analytic likelihood score, and an independently differenced observed
information matrix. Parameters are optimized as `(atanh(φ), log(q), log(r))`, with
explicit wide bounds. Intervals are transformed Wald intervals, not exact finite
sample confidence sets. Oracle intervals are conditional-regression asymptotic
intervals and must not be conflated with observation-only information.

Optimizer audit: **{failed} failed selected fits**, **{boundary} boundary fits**,
**{valid}/{config['trials']} usable observed-information interval sets**. The raw
diagnostics retain every start, objective, evaluation count, gradient norm,
information eigenvalue, and boundary indicator. Failed or boundary trials are not
removed from point-estimate or state-risk summaries. Invalid intervals count as
misses in the all-trial coverage column; available-only coverage is also shown.

Bias tests are two-sided one-sample t-tests across trials. Holm adjustment covers
the six reported estimator/parameter combinations. Bias magnitude, Monte Carlo
precision, and RMSE should drive interpretation; a small p-value is not by itself
an economically meaningful failure.

![Parameter recovery](figures/03_parameter_recovery.png)

## Why the loading normalization matters

For any nonzero c, replacing `(x, α, q)` by `(cx, α/c, c²q)` leaves the entire
observation distribution unchanged, including the stationary initial distribution.
Without a normalization or external scale information, α and q cannot be separately
identified. Comparing states in inconsistent units can produce arbitrary RMSE
differences with exactly the same observational fit.

The saved ridge evaluates 101 scales. Its maximum absolute log-likelihood difference
is **{ridge.likelihood_difference.abs().max():.3g}**; after rescaling states back to
the original units, the RMSE range is **{np.ptp(ridge.latent_rmse_after_unit_correction):.3g}**.
This is an exact observational equivalence, not an optimizer defect.

![Scale identification](figures/05_identifiability.png)

## Particle approximation, weight degeneracy, and compute

The convergence experiment uses **{config['convergence_trials']} independent series**
and particle counts **{', '.join(map(str, config['particle_counts']))}**, with separate
streams for each proposal and particle count. Both proposals retain previous weights,
normalize in log space, and use systematic resampling when ESS falls below N/2.
The initial optimal proposal is the exact one-observation Gaussian posterior.

The log likelihood estimate is generally downward biased even though the likelihood
estimate is unbiased under standard SMC conditions. We therefore record log-error
and approximation error without claiming an unbiased log estimator. Exact KF-based
MLE is used for this tractable model; noisy PF likelihood optimization would add
unnecessary simulation error and obscure the comparison.

![Particle count and runtime study](figures/04_particle_convergence.png)

All timings are measured wall time on the execution machine, not universal latency
claims. Core fitted-method times include parameter fitting. Particle counts and
the chronological split were fixed in the profile before running results.

## Missing observations and contaminated noise

Each scenario contains **{config['stress_trials']} independent trials**. Random
missingness is independent with probability 0.3; the block removes the middle third
of the series. Contaminated observations have probability 0.05 and eight times the
normal observation standard deviation. The Gaussian model retains its original
variance and is deliberately misspecified under contamination.

{_table(['Scenario', 'Method', 'MSE', '95% state coverage', 'Interval width'], stress_rows)}

The mixture PF is supplied the true contamination probability and scale. Its
advantage, if present, reflects a more appropriate observation model plus numerical
approximation; it does not prove PF superiority under identical Gaussian assumptions.
This experiment does not estimate unknown contamination parameters or handle
informative missingness.

![Missingness and outlier studies](figures/06_stress_tests.png)

## Chronological holdout: avoiding parameter look-ahead

Across **{config['holdout_trials']} trials**, parameters are estimated on the first
**{config['train_length']} observations** and then frozen. The subsequent
**{config['test_length']} observations** are processed sequentially. The predictive
score is computed before each held-out observation is assimilated. The full-sample
fit is shown as an explicitly contaminated look-ahead comparison, not a valid
forecasting baseline.

{_table(['Parameter source', 'Test state MSE', 'Test state coverage', 'Mean test negative log predictive density'], holdout_rows)}

## Nonlinear extension: latent stochastic volatility

The latent log variance follows an AR(1), while observations satisfy
`y_t = exp(h_t/2) ε_t`. This non-Gaussian filtering problem has no ordinary exact
Kalman solution. We compare a bootstrap PF, a deterministic grid reference, and
a Gaussian approximation to log-squared observations with the correct first two
noise moments. All model parameters are known in this experiment.

{_table(['Method', 'Log-variance MSE', '95% state coverage', 'Mean milliseconds / series'], sv_rows)}

The study has **{config['sv_trials']} independent trials**. The grid has
**{config['grid_size']} points** on [-10, 8]; edge mass is retained as a truncation
diagnostic. A two-resolution check on the preselected first series gives a maximum
filtered-mean difference of **{grid['maximum_absolute_mean_difference']:.3g}** and
log-likelihood difference **{grid['loglikelihood_difference']:.3g}**. Refinement is
checked on one series, not claimed to certify all possible datasets. The finite
grid remains a numerical reference, not an analytic ground truth.

![Stochastic volatility](figures/07_stochastic_volatility.png)

## Parameter uncertainty is a separate estimation problem

A preselected independent series has **{config['bootstrap_replications']} parametric
bootstrap refits, of which **{bootstrap['successful_refits']}** report optimizer
success. Its fitted parameters, transformed Wald intervals, bootstrap percentile
intervals, all refits, and boundary indicators are saved in `bootstrap_summary.json`
and `bootstrap_refits.csv`. This illustrates finite-sample parameter uncertainty;
it does not validate bootstrap coverage, and it is not a posterior over parameters.

The project does **not** label plug-in state intervals as accounting for parameter
uncertainty. A full Bayesian joint parameter/state analysis, or a separate validated
bootstrap state-interval calibration study, would be needed for that extension.

## Reproduction and scope

Experiment wall time: **{manifest['elapsed_seconds']:.1f} seconds**. Python
**{manifest['python']}**, platform `{manifest['platform']}`. The manifest records
package versions, each study's runtime, the full configuration, source SHA-256
hashes, and raw-result hashes. Seeds make numerical streams reproducible; wall times
and floating-point optimizer stopping may vary across platforms and library versions.

```bash
python -m latent_state_lab.cli run --profile {config['profile']} --seed {config['seed']} --output results/reproduced
python -m latent_state_lab.cli render results/reproduced
pytest -q
```

This repository implements established methods from statistical literature.
It uses synthetic datasets only and does not claim a new filtering algorithm
or evidence of live trading performance.
See [derivations](../../docs/METHODS.md), [references and provenance](../../docs/REFERENCES.md),
and [limitations](../../docs/LIMITATIONS.md).
"""
    (output / "REPORT.md").write_text(text)
    return output / "REPORT.md"
