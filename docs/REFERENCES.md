# References and implementation provenance

The package, simulations, tests, plots, and explanations in this repository are
original implementations. Reference source code was not copied or translated.
No real customer, financial account, or market dataset is redistributed.

## Primary methodological sources

- Simo Särkkä and Lennart Svensson (2023), *Bayesian Filtering and Smoothing*, second
  edition, Cambridge University Press.
  [Publisher record and DOI](https://doi.org/10.1017/9781108917407).
  General reference for Gaussian filtering, smoothing, nonlinear filtering, and
  parameter estimation. The source's prose and implementation code are not reused.

- Arnaud Doucet and Adam M. Johansen (2011), “A Tutorial on Particle Filtering and
  Smoothing: Fifteen Years Later,” in *The Oxford Handbook of Nonlinear Filtering*,
  pp. 656–705. [Author-hosted manuscript](https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF.pdf),
  [university bibliographic record](https://wrap.warwick.ac.uk/id/eprint/37961/).
  Reference for sequential importance sampling, proposal choice, resampling, and
  particle likelihood estimates. Here the scalar proposal formulas are independently
  obtained by completing the square in two Gaussian densities.

- Nikolas Kantas, Arnaud Doucet, Sumeetpal S. Singh, Jan Maciejowski, and Nicolas
  Chopin (2015), “On Particle Methods for Parameter Estimation in State-Space
  Models,” *Statistical Science*, 30(3), 328–351.
  [Author manuscript](https://arxiv.org/abs/1412.8695).
  Context for the distinction between filtering and static parameter inference.
  This repository does not implement the paper's broader parameter-learning methods.

The algebra of the scale equivalence and the risk decomposition is provided in
[METHODS.md](METHODS.md), so those central claims can be checked directly without
depending on a secondary summary or a numerical optimizer.

## Third-party software

NumPy provides array operations and seeded random-number generation; SciPy supplies
optimization, stable log-sum-exp, and statistical distributions; pandas handles
tabular output; Matplotlib creates figures; pytest executes the tests. Their
licenses remain with their respective projects. This repository's MIT license
covers its original code and documentation only.
