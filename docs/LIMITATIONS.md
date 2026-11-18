# Scope and limitations

1. **Synthetic inference, not a trading strategy.** This project has no market
   return claim, execution model, investment recommendation, or real-data alpha
   validation. It demonstrates statistical research and numerical implementation.

2. **Known loading fixes units.** Alpha is supplied, not estimated. At zero
   persistence the state and observation variances remain unidentifiable even after
   this normalization; short series can also be weakly identified.

3. **Stationarity and scalar state.** The core model is stationary with independent
   Gaussian innovations. It omits drift, structural breaks, time-varying parameters,
   multivariate states, and endogenous observation errors.

4. **Conditional intervals do not integrate parameter uncertainty.** Fitted-parameter
   state bands are plug-in intervals. The observed-information and oracle parameter
   intervals are asymptotic and can under-cover. Invalid intervals and all boundary
   fits are retained in the raw output and counted explicitly. The illustrative
   bootstrap's coverage is not independently validated.

5. **Information sets differ.** The smoother sees future observations. The oracle
   estimates parameters from true states. Full-sample MLE sees the future when
   learning parameters even if state updates then run forward. Only the separate
   train-only holdout has the stated chronological fitting protocol.

6. **Particle approximation is not model improvement.** Under correct linear
   Gaussian assumptions, exact Kalman filtering is the target. A better noise model
   can outperform a misspecified Gaussian filter, but this does not demonstrate
   that particles beat exact inference under identical assumptions.

7. **Known stress-model parameters.** Contamination probability/scale and nonlinear
   stochastic-volatility parameters are provided to the appropriate filters. This
   isolates state inference and does not solve their parameter-learning problem.

8. **Numerical reference limits.** The SV grid is truncated and discretized. Boundary
   mass is checked on every trial, but the two-resolution audit covers one preselected
   series. A higher-dimensional problem would make this grid approach impractical.

9. **Computational bounds and optimization.** Three initializations reduce local-fit
   risk without proving global optimality. The fitting domain excludes persistence
   magnitude above 0.995. Standard errors can be unreliable near bounds even if
   the optimizer reports convergence. A future extension could use profile likelihood
   intervals and more extensive finite-sample coverage experiments.

10. **Monte Carlo precision and multiple comparisons.** A thousand independent trials
    quantify simulation variability for the selected parameter setting; they do not
    establish uniform performance over every model. The smaller secondary studies
    are identified by their own trial counts. Holm adjustment covers the six core
    bias tests, not every visual comparison in the repository.

11. **Reproducibility is numerical, not timing equality.** Seeds and source/result
    hashes support auditing. Library updates, floating-point hardware, process
    scheduling, and BLAS configuration can change optimizer stopping or runtime.
    Timings are local measurements and are not production latency guarantees.

12. **No claim of novel filtering algorithms.** The contribution is an original,
    reproducible implementation and an identification-aware experimental design,
    including tests that make common methodological errors observable. The classical
    methods and the public project that motivated the scope are credited.
