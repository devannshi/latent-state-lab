# Discussing this project accurately

## A one-minute explanation

“I built an independently implemented state-space inference lab. Before comparing
algorithms, I checked whether the model was identifiable: a free observation loading
and a free latent innovation scale generate an observational-equivalence ridge.
I normalized the loading, derived an exact likelihood and score, and tested filtering
and smoothing against dense Gaussian conditioning. I then ran a thousand independent
trials, recorded every optimizer outcome, and separated parameter uncertainty from
particle approximation error. Particles converge to the exact filter in the Gaussian
case; their modeling value appears when the observation model becomes non-Gaussian.”

Adapt that explanation to what you can derive and defend yourself. Do not describe
this as personal historical coursework, cite an institution as your affiliation,
claim collaborators who did not contribute, or present generated code as proof that
you independently know every implementation detail.

## Questions to be prepared for

**Why is estimating alpha and q simultaneously a problem?**
Write the observation covariance and show that it contains alpha²q. Rescale x by c
and compensate alpha and q. Explain that a normalization defines state units and that
near-zero persistence still creates a second weak-identification problem.

**Why can't a particle filter outperform a correctly specified Kalman filter?**
Derive the conditional-mean squared-error decomposition. State the assumptions:
same observations, known true parameters, quadratic loss, and correct Gaussian model.
A finite simulation can have accidental reversals; different fitted parameters or
information sets invalidate the simple comparison.

**What is actually being estimated by MLE?**
The structural parameters phi, q, and r from observations, using a likelihood obtained
by integrating out latent states. A filter then computes states conditional on those
parameters. “MLE versus KF” conflates parameter estimation with state inference.

**How did you validate your score and likelihood?**
The innovation likelihood matches a dense multivariate Gaussian calculation, including
missing observations. Score sensitivities match finite differences of that independent
likelihood calculation. State moments match direct Gaussian conditioning.

**What makes an optimal particle proposal optimal?**
It conditions each transition proposal on the current observation, reducing incremental
weight variance conditional on the ancestor. It does not remove ancestor-weight
degeneracy and is not automatically a fully adapted auxiliary filter.

**Is the particle log likelihood unbiased?**
No. Under standard conditions the likelihood estimate is unbiased; taking logs creates
downward Jensen bias. The code tests the likelihood ratio across independent runs.

**What does the ESS threshold measure?**
Weight concentration at the current time. It is a heuristic diagnostic and is not the
number of independent posterior samples or a full measure of genealogical diversity.

**Why is a forward filter with full-sample MLE not a valid live strategy?**
Its parameters already used future observations. Show the separate training/holdout
study and explain the prediction/update ordering of held-out likelihood scores.

**What uncertainty do the shaded bands represent?**
Conditional pointwise state uncertainty given specified parameters. Plug-in fitted
parameters omit their estimation uncertainty. Parameter confidence intervals and
Monte Carlo intervals are separate objects with different repeated-sampling meanings.

**How do you interpret poor parameter-interval coverage?**
As a limitation of finite-sample asymptotic intervals, weak variance-component
identification, and boundary behavior. Show the available-interval denominator and
all-trial coverage; do not quietly remove failed fits or claim the model is solved.

**Why did the robust mixture PF outperform a Gaussian filter under outliers?**
It was given the correct non-Gaussian noise model. That is an advantage of model
specification; a numerical method is necessary to approximate the resulting posterior.
This does not establish superiority over a correctly specified exact Gaussian filter.

**What would you do next?**
Choose one substantive extension: profile-likelihood interval calibration; unknown
contamination-parameter learning; multivariate factor states; sequential parameter
adaptation under breaks; or a real-data application with chronological prediction
and a clear economic question. Do not add methods without a falsifiable comparison.

## Evidence to bring to an interview

- `results/full/REPORT.md`: measured outcomes and honest failure counts.
- `docs/METHODS.md`: equations you should be able to reproduce on a whiteboard.
- `tests/test_exact_inference.py`: independent analytic validation.
- `results/full/paired_risk_differences.csv`: paired inference rather than a ranked
  screenshot of one path.
- `results/full/manifest.json`: configuration, source hashes, and seed namespaces.
- `notebooks/tutorial.ipynb`: an end-to-end explanation you can execute live.
