# Mathematical specification and implementation decisions

All implementations in this repository are original. Equations below are derived
for the project's scalar model; the general filtering literature is listed in
[REFERENCES.md](REFERENCES.md). The package uses **variances**, not standard
deviations, for `q` and `r`.

## 1. An identifiable generative model

For time indices $t=0,\ldots,T-1$,

$$
x_0\sim N\!\left(0,\frac{q}{1-\phi^2}\right),\qquad
x_t=\phi x_{t-1}+\eta_t,\quad \eta_t\sim N(0,q),
$$
$$
y_t=\alpha x_t+\epsilon_t,\qquad \epsilon_t\sim N(0,r).
$$

All innovations and the initial state are independent. We require $|\phi|<1$,
$q,r>0$, and a **known, nonzero** $\alpha$. The main experiment fixes
$(\phi,q,r,\alpha)=(0.85,0.16,0.49,1)$. Stationarity is part of the data-generating
model, not an arbitrary filter initialization. There is no intercept or drift.

The observation covariance is

$$
\operatorname{Cov}(y_t,y_s)
=\frac{\alpha^2q}{1-\phi^2}\phi^{|t-s|}+r\mathbf 1\{t=s\}.
$$

If $\alpha$ and $q$ are both unknown, every transformation

$$
x'_t=cx_t,\qquad \alpha'=\alpha/c,\qquad q'=c^2q,\quad c\ne0
$$

has the same observation distribution. A likelihood optimizer cannot learn a unique
latent scale from observations alone. Fixing $\alpha=1$ chooses units rather than
adding information. An external measurement or an alternative normalization could
also identify scale. If the true loading were 1.5 and we chose loading 1, the
identified state would be $1.5x_t$, not the original $x_t$.

Even with known loading, $\phi=0$ makes $\alpha^2q+r$ the only observed variance;
the two variance components cannot then be separated. Near-zero persistence and
short/noisy series produce weak identification. Therefore normalizing loading does
not imply that every finite dataset will have a well-conditioned likelihood.

## 2. Exact filtering and likelihood

Let $m_t,P_t$ denote filtered moments. At the initial time,
$a_0=0$, $R_0=q/(1-\phi^2)$. Subsequently,

$$
a_t=\phi m_{t-1},\qquad R_t=\phi^2P_{t-1}+q.
$$

For an observed value define

$$
v_t=y_t-\alpha a_t,\quad S_t=\alpha^2R_t+r,\quad K_t=R_t\alpha/S_t,
$$
$$
m_t=a_t+K_tv_t,\qquad P_t=R_tr/S_t.
$$

The variance expression avoids subtracting nearly equal values. The likelihood is

$$
\ell(\theta;y)=-\frac12\sum_{t\in\mathcal O}
\left[\log(2\pi)+\log S_t+v_t^2/S_t\right],
$$

where $\mathcal O$ is the set of observed indices. Missing data produce no
observation update and no likelihood contribution: $m_t=a_t,P_t=R_t$. Prediction
still runs through the missing period. Tests compare this recursion with a dense
multivariate Gaussian likelihood, including removal of missing rows/columns.

## 3. Marginal parameter estimation

The observation-only estimator maximizes the exact likelihood over

$$
z=(\operatorname{atanh}\phi,\log q,\log r).
$$

The implementation supplies an analytic score to L-BFGS-B. For any parameter
coordinate, differentiating the prediction gives

$$
da_t=m_{t-1}\,d\phi+\phi\,dm_{t-1},\qquad
dR_t=2\phi P_{t-1}\,d\phi+\phi^2\,dP_{t-1}+dq,
$$

and $dv_t=-\alpha da_t$, $dS_t=\alpha^2dR_t+dr$. A negative-likelihood term has
derivative

$$
dL_t=\tfrac12(S_t^{-1}-v_t^2S_t^{-2})dS_t+(v_t/S_t)dv_t.
$$

The update sensitivities follow from $dm_t=da_t+v_tdK_t+K_tdv_t$ and
$P_t=R_tr/S_t$. Crucially, the initial stationary variance is differentiated too.
The final chain rule accounts for $d\phi/dz_1=1-\phi^2$, $dq/dz_2=q$, and
$dr/dz_3=r$. Tests compare the analytic score with independent central differences
of the exact Kalman likelihood for negative/positive persistence, missing data, and
non-unit loadings.

Three deterministic starts use the adjacent-observation regression coefficient
(clipped to ±0.9), plus ±0.85 and ±0.15 with the same sign. The best objective is
retained even if its optimizer status is unsuccessful, and that status is reported.
Each start has at most 300 iterations. Persistence is bounded by ±0.995; variance
bounds run from $10^{-6}s^2$ to $20s^2$, where
$s^2=\max(\operatorname{mean}_{t\in\mathcal O}y_t^2,10^{-4})$; state-variance
bounds are divided by $\alpha^2$. These are computational domain constraints,
not a prior. Boundary fits are disclosed.

The observed information is calculated by central differences of the analytic
score at the fitted parameter, with coordinate steps $10^{-4}\max(1,|z_j|)$.
It is symmetrized before eigenvalue checks. An interval set is considered available
only if the selected optimization succeeds, it is not at a numerical bound, the
smallest information eigenvalue exceeds $10^{-7}$, and the condition number is
below $10^9$. There is no eigenvalue flooring to manufacture an interval.

For available information $H$, transformed-coordinate Wald limits are
$\hat z_j\pm1.96\sqrt{(H^{-1})_{jj}}$, mapped back with `tanh` or `exp`.
These are asymptotic marginal parameter intervals, not a joint 95% region, and may
have poor finite-sample coverage. The simulation measures that limitation.

## 4. Complete-data oracle: a separate information set

If the states were directly observed, the conditional likelihood given $x_0$
has the estimates

$$
\hat\phi=\frac{\sum_{t=1}^{T-1}x_{t-1}x_t}{\sum_{t=1}^{T-1}x_{t-1}^2},\quad
\hat q=\frac1{T-1}\sum_{t=1}^{T-1}(x_t-\hat\phi x_{t-1})^2,\quad
\hat r=\frac1{|\mathcal O|}\sum_{t\in\mathcal O}(y_t-\alpha x_t)^2.
$$

Persistence is clipped to the stated stationary fitting domain. This is a
**conditional** complete-data MLE; it omits the stationary density of $x_0$.
The implementation reports this name explicitly. Approximate standard errors use
$\operatorname{se}(\hat\phi)=\sqrt{\hat q/\sum x_{t-1}^2}$,
$\operatorname{se}(\log\hat q)=\sqrt{2/(T-1)}$, and
$\operatorname{se}(\log\hat r)=\sqrt{2/|\mathcal O|}$, with a delta-method
transform for `atanh(phi)`.

The oracle's parameters are subsequently supplied to a filter that only sees y.
This diagnoses the value of parameter information. It does not compare an estimator
that directly returns the true states against filters and does not establish a
universal oracle-risk lower bound for every realized sample.

## 5. Smoothing is not filtering

For $t=T-2,\ldots,0$, the Rauch–Tung–Striebel backward recursion is

$$
J_t=P_t\phi/R_{t+1},\qquad
m^s_t=m_t+J_t(m^s_{t+1}-a_{t+1}),
$$
$$
P^s_t=P_t+J_t^2(P^s_{t+1}-R_{t+1}).
$$

Terminal moments equal the filtered terminal moments. The smoother conditions on
the complete series. Tests verify all its moments against Gaussian conditioning
on the full observation vector. It has no claim to real-time implementability.

## 6. Sequential Monte Carlo

Particles and normalized weights approximate the filtered distribution. With the
bootstrap proposal, sample the transition and multiply the previous weight by the
observation density. For Gaussian observations, the conditional optimal proposal is

$$
x_t^{(i)}\mid x_{t-1}^{(i)},y_t\sim N(\mu_t^{(i)},V),\quad
V=\frac{qr}{\alpha^2q+r},
$$
$$
\mu_t^{(i)}=\phi x_{t-1}^{(i)}+
\frac{q\alpha}{\alpha^2q+r}(y_t-\alpha\phi x_{t-1}^{(i)}).
$$

The incremental weight is the predictive density
$N(y_t;\alpha\phi x_{t-1}^{(i)},\alpha^2q+r)$. At time zero, use the stationary
prior variance instead of q and prior mean zero. The first optimal proposal then
draws directly from the exact one-observation posterior, with equal weights and
an exact first likelihood increment. This is an optimal proposal **conditional on
an ancestor**, not a fully adapted auxiliary filter with ancestor reweighting.

For old normalized log weights $l_i$ and log increments $g_i$, compute

$$
\log c_t=\operatorname{logsumexp}_i(l_i+g_i),\quad
l'_i=l_i+g_i-\log c_t,\quad \log\hat p(y)=\sum_t\log c_t.
$$

No-observation steps sample the transition, preserve weights, and contribute zero
to log likelihood. Reported posterior moments/quantiles and ESS are computed before
resampling. If $1/\sum_iw_i^2<N/2$, systematic resampling selects ancestors with
a single uniform offset on a regular probability grid, then resets weights to 1/N.
Resampling at the final time is omitted because it cannot improve reported results.

For a valid particle likelihood estimator, unbiasedness concerns $\hat p(y)$,
not $\log\hat p(y)$. Jensen's inequality implies downward log bias. The test
suite checks the mean likelihood ratio against 1 using independent particle runs.

The core study uses exact-likelihood MLE for parameter inference and fixed true
parameters for the KF/PF approximation comparison. It deliberately avoids optimizing
a noisy PF likelihood in a model with a cheap exact likelihood. Parameter inference
and filter approximation error are distinct questions.

### Why a particle filter cannot beat an exact Kalman mean in population MSE

Let $m(Y)=E[X\mid Y]$. For any estimator $a(Y,U)$, including independent
algorithmic randomness U,

$$
E[(X-a)^2]=E[\operatorname{Var}(X\mid Y)]+E[(m(Y)-a(Y,U))^2].
$$

The cross term is zero after conditioning on Y and U. The exact Gaussian filter
has the first term only. A particle approximation adds a nonnegative term.
This argument assumes identical information, the correct model, and known true
parameters. It does not compare smoothing to filtering, different fitted parameter
values, misspecified models, or one favorable realized path.

## 7. Stress tests

Random missingness is independent of x and y; the block-missing experiment removes
the middle third of observations. Under either, exact Gaussian filtering remains
available and should be the benchmark.

The contamination study replaces the observation density by

$$
(1-\pi)N(y_t;\alpha x_t,r)+\pi N(y_t;\alpha x_t,\kappa^2r),
\quad \pi=0.05,\ \kappa=8.
$$

A bootstrap PF evaluates this density using `logaddexp`. It is supplied the true
mixture parameters, whereas the Gaussian baselines retain r. This is a deliberate
comparison of model specification, not a same-model contest between algorithms.

For a nonlinear extension,

$$
h_t=\mu+\phi(h_{t-1}-\mu)+\eta_t,\quad
y_t=e^{h_t/2}\epsilon_t,
$$

with $(\mu,\phi,q)=(-1,0.95,0.08)$. A bootstrap PF targets h directly. A grid
filter uses a discrete quadrature of the Gaussian transition on [-10,8], saving
boundary mass and comparing two resolutions on the preselected first series.
The reference is numerical, with disclosed truncation and discretization error.

For a cheap approximate baseline, transform $z_t=\log y_t^2$. Then
$z_t=h_t+u_t$, where $u_t=\log\epsilon_t^2$ has mean
$-\gamma-\log2$ and variance $\pi^2/2$. A Kalman filter matching these moments
is an approximation because u is not Gaussian. Observation magnitudes are bounded
below by machine tiny before taking logs. No claim is made that this baseline is
the best available stochastic-volatility method.

## 8. Monte Carlo estimands and uncertainty

Each trial's state MSE is $T^{-1}\sum_t(\hat x_t-x_t)^2$. Its square root is the
trial RMSE. Both are averaged across trials and labeled separately. For a trial
outcome $Z_i$, MCSE is $s_Z/\sqrt{R}$; Student-t intervals use R−1 degrees of
freedom. Paired comparisons compute method differences **within each trial** first.
Time points in a series are dependent and are never counted as independent trials.

Parameter bias is the mean estimation error, and parameter RMSE is the square root
of mean squared estimation error. Two-sided t-tests evaluate zero average error;
Holm adjustment covers all six reported estimator/parameter tests. The magnitude
of bias and its Monte Carlo uncertainty accompany p-values.

Conditional state intervals at known parameters are posterior credible intervals
under the specified generative model. Full-sample plug-in intervals condition on
estimated parameters and omit their uncertainty. Parameter Wald intervals, parameter
bootstrap intervals, state credible intervals, and Monte Carlo intervals answer
four different questions and are never given interchangeable labels.

The separate parametric bootstrap generates series from one fitted model and refits
parameters in every replicate. Its percentile intervals illustrate refit dispersion.
They are not asserted to be Bayesian parameter posteriors, calibrated state bands,
or validated 95% finite-sample confidence intervals.

## 9. Chronological evaluation and reproducibility

A separate holdout study fits parameters on the first 240 observations and freezes
them for the next 160. Test observations update states sequentially. Each test
predictive likelihood term is scored before the corresponding update. A full-series
parameter fit is shown only as an explicitly labeled look-ahead comparison.

The complete experiment grid is encoded in `StudyConfig` before simulation. Random
streams use `SeedSequence([master, study, trial, component])`, converted to a stored
64-bit seed. Core trials, convergence, stresses, holdout, bootstrap, and illustrative
paths have separate namespaces. Examples always use a fixed designated stream and
are not selected for attractive outcomes. The report records every core fit and
does not exclude failures or inconvenient trials.
