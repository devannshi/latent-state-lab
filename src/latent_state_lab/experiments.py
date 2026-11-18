"""Predeclared simulation studies, independent trial streams, and raw audit outputs."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import norm

from .estimation import fit_mle, fit_oracle, transform
from .kalman import kalman_filter, rts_smoother
from .metrics import mean_interval, state_metrics, summarize_parameters, summarize_states
from .model import Parameters, rescale_latent, simulate
from .particle import particle_filter
from .stochastic_volatility import SVParameters, grid_filter, simulate_sv, sv_particle_filter


@dataclass(frozen=True)
class StudyConfig:
    profile: str = "full"
    seed: int = 20261117
    trials: int = 1000
    length: int = 160
    particles: int = 256
    convergence_trials: int = 60
    particle_counts: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048)
    stress_trials: int = 100
    holdout_trials: int = 100
    train_length: int = 240
    test_length: int = 160
    sv_trials: int = 60
    sv_particles: int = 2048
    grid_size: int = 401
    bootstrap_replications: int = 200
    workers: int = 1

    @classmethod
    def for_profile(cls, profile: str, *, seed: int = 20261117, workers: int = 1):
        if profile == "full":
            return cls(seed=seed, workers=workers)
        if profile == "quick":
            return cls(profile=profile, seed=seed, trials=24, length=100, particles=128,
                       convergence_trials=8, particle_counts=(32, 128, 512), stress_trials=12,
                       holdout_trials=8, train_length=120, test_length=80, sv_trials=8,
                       sv_particles=1024, grid_size=301, bootstrap_replications=24, workers=workers)
        if profile == "smoke":
            return cls(profile=profile, seed=seed, trials=4, length=40, particles=64,
                       convergence_trials=2, particle_counts=(32, 128), stress_trials=2,
                       holdout_trials=2, train_length=60, test_length=40, sv_trials=2,
                       sv_particles=256, grid_size=201, bootstrap_replications=4, workers=workers)
        raise ValueError("Profile must be full, quick, or smoke.")


def seed_for(master: int, study: int, trial: int, component: int = 0) -> int:
    return int(np.random.SeedSequence([master, study, trial, component]).generate_state(1, dtype=np.uint64)[0])


def _rng(config, study, trial, component=0):
    return np.random.default_rng(seed_for(config.seed, study, trial, component))


def _timed(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - start


def _oracle_intervals(parameters, x, observed_count):
    """Asymptotic conditional-regression intervals, separately labeled in report."""
    se = np.array([np.sqrt(parameters.q / np.dot(x[:-1], x[:-1])) / (1 - parameters.phi**2),
                   np.sqrt(2 / (len(x) - 1)), np.sqrt(2 / observed_count)])
    z = transform(parameters)
    lo, hi = z - norm.ppf(0.975) * se, z + norm.ppf(0.975) * se
    return (np.array([np.tanh(lo[0]), np.exp(lo[1]), np.exp(lo[2])]),
            np.array([np.tanh(hi[0]), np.exp(hi[1]), np.exp(hi[2])]))


def _core_trial(arguments):
    trial, config = arguments
    truth = Parameters()
    data = simulate(config.length, truth, _rng(config, 1, trial))
    fit, fit_seconds = _timed(fit_mle, data.y)
    oracle, oracle_seconds = _timed(fit_oracle, data.x, data.y)
    known, known_seconds = _timed(kalman_filter, data.y, truth)
    fitted, fitted_seconds = _timed(kalman_filter, data.y, fit.parameters)
    oracle_filter, oracle_filter_seconds = _timed(kalman_filter, data.y, oracle)
    known_smooth, smooth_seconds = _timed(rts_smoother, known, truth)
    fitted_smooth, fitted_smooth_seconds = _timed(rts_smoother, fitted, fit.parameters)
    results = [
        ("KF_known", known, known_seconds, "filter", "known"),
        ("KF_full_sample_MLE", fitted, fit_seconds + fitted_seconds, "filter", "full_sample_MLE"),
        ("KF_oracle_parameters", oracle_filter, oracle_seconds + oracle_filter_seconds, "filter", "oracle"),
        ("RTS_known", known_smooth, known_seconds + smooth_seconds, "smoother", "known"),
        ("RTS_full_sample_MLE", fitted_smooth, fit_seconds + fitted_seconds + fitted_smooth_seconds,
         "smoother", "full_sample_MLE"),
    ]
    state_rows = []
    for method, result, seconds, information_set, parameter_source in results:
        state_rows.append({"trial": trial, "data_seed": seed_for(config.seed, 1, trial),
                           "method": method, "information_set": information_set,
                           "parameter_source": parameter_source, "seconds": seconds,
                           **state_metrics(data.x, result)})
    particle_rows = []
    for component, proposal in enumerate(("bootstrap", "optimal"), start=1):
        pf, seconds = _timed(particle_filter, data.y, truth, config.particles,
                             _rng(config, 1, trial, component), proposal=proposal)
        method = f"PF_{proposal}_known"
        state_rows.append({"trial": trial, "data_seed": seed_for(config.seed, 1, trial),
                           "method": method, "information_set": "filter", "parameter_source": "known",
                           "seconds": seconds, **state_metrics(data.x, pf)})
        particle_rows.append({"trial": trial, "proposal": proposal, "particles": config.particles,
                              "particle_seed": seed_for(config.seed, 1, trial, component),
                              "approximation_mse": float(np.mean((pf.mean - known.mean)**2)),
                              "loglikelihood_error": pf.loglikelihood - known.loglikelihood,
                              "mean_ess_fraction": float(np.mean(pf.ess) / config.particles),
                              "resampling_count": int(pf.resampled.sum()), "seconds": seconds})
    parameter_rows = []
    oracle_lower, oracle_upper = _oracle_intervals(oracle, data.x, data.observed.sum())
    for method, estimate, lower, upper, valid in [
        ("marginal_MLE", fit.parameters, fit.lower, fit.upper, fit.ci_valid),
        ("oracle_conditional_MLE", oracle, oracle_lower, oracle_upper, abs(oracle.phi) < 0.995),
    ]:
        for j, name in enumerate(("phi", "q", "r")):
            parameter_rows.append({"trial": trial, "method": method, "parameter": name,
                                   "estimate": getattr(estimate, name), "lower": lower[j],
                                   "upper": upper[j], "ci_valid": valid})
    diagnostics = {"trial": trial, "data_seed": seed_for(config.seed, 1, trial),
                   "success": fit.success, "message": fit.message, "nll": fit.nll,
                   "iterations": fit.iterations, "evaluations_all_starts": fit.evaluations,
                   "gradient_norm": fit.gradient_norm, "at_boundary": fit.at_boundary,
                   "ci_valid": fit.ci_valid, "hessian_min_eigenvalue": fit.hessian_min_eigenvalue,
                   "hessian_condition": fit.hessian_condition, "fit_seconds": fit_seconds,
                   "starts": fit.starts}
    return state_rows, parameter_rows, particle_rows, diagnostics


def core_study(config: StudyConfig, output: Path):
    states, parameters, particles, diagnostics = [], [], [], []
    tasks = ((i, config) for i in range(config.trials))
    if config.workers > 1:
        executor = ProcessPoolExecutor(max_workers=config.workers)
        iterator = executor.map(_core_trial, tasks, chunksize=8)
    else:
        executor = None
        iterator = map(_core_trial, tasks)
    try:
        for i, (state, parameter, particle, diagnostic) in enumerate(iterator, start=1):
            states.extend(state)
            parameters.extend(parameter)
            particles.extend(particle)
            diagnostics.append(diagnostic)
            if i % max(1, config.trials // 10) == 0:
                print(f"Core Monte Carlo: {i}/{config.trials}", flush=True)
    finally:
        if executor:
            executor.shutdown()
    states, parameters, particles = pd.DataFrame(states), pd.DataFrame(parameters), pd.DataFrame(particles)
    states.to_csv(output / "state_trials.csv", index=False)
    parameters.to_csv(output / "parameter_trials.csv", index=False)
    particles.to_csv(output / "particle_diagnostics.csv", index=False)
    summarize_states(states, ("method", "information_set", "parameter_source")).to_csv(
        output / "state_summary.csv", index=False)
    summarize_parameters(parameters, Parameters().as_dict()).to_csv(output / "parameter_summary.csv", index=False)
    (output / "optimizer_diagnostics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in diagnostics) + "\n")
    pivot = states.pivot(index="trial", columns="method", values="mse")
    paired = [{"comparison": f"{method} minus KF_known", **mean_interval(pivot[method] - pivot.KF_known)}
              for method in pivot.columns if method not in {"KF_known", "RTS_known", "RTS_full_sample_MLE"}]
    paired.append({"comparison": "RTS_known minus KF_known (different information sets)",
                   **mean_interval(pivot.RTS_known - pivot.KF_known)})
    pd.DataFrame(paired).to_csv(output / "paired_risk_differences.csv", index=False)


def convergence_study(config, output):
    rows = []
    for trial in range(config.convergence_trials):
        data = simulate(config.length, Parameters(), _rng(config, 2, trial))
        exact, exact_seconds = _timed(kalman_filter, data.y, Parameters())
        for count in config.particle_counts:
            for component, proposal in enumerate(("bootstrap", "optimal"), start=1):
                seed = seed_for(config.seed, 2, trial, count * 10 + component)
                result, seconds = _timed(particle_filter, data.y, Parameters(), count,
                                         np.random.default_rng(seed), proposal=proposal)
                rows.append({"trial": trial, "proposal": proposal, "particles": count,
                             "data_seed": seed_for(config.seed, 2, trial), "particle_seed": seed,
                             "approximation_mse": float(np.mean((result.mean - exact.mean)**2)),
                             "loglikelihood_error": result.loglikelihood - exact.loglikelihood,
                             "loglikelihood_squared_error": (result.loglikelihood - exact.loglikelihood)**2,
                             "mean_ess_fraction": float(np.mean(result.ess) / count),
                             "seconds": seconds, "kf_seconds": exact_seconds})
    pd.DataFrame(rows).to_csv(output / "convergence_trials.csv", index=False)
    print("Particle-count convergence complete", flush=True)


def stress_study(config, output):
    rows = []
    scenarios = ("matched", "missing_30pct", "missing_block", "outliers_5pct")
    for scenario_index, scenario in enumerate(scenarios):
        for trial in range(config.stress_trials):
            data = simulate(config.length, Parameters(), _rng(config, 3 + scenario_index, trial),
                            missing_probability=0.3 if scenario == "missing_30pct" else 0,
                            outlier_probability=0.05 if scenario == "outliers_5pct" else 0,
                            outlier_scale=8)
            if scenario == "missing_block":
                data.y[config.length // 3:config.length * 2 // 3] = np.nan
            candidates = [("KF_Gaussian", kalman_filter, (data.y, Parameters()), {}),
                          ("PF_Gaussian", particle_filter,
                           (data.y, Parameters(), max(512, config.particles),
                            _rng(config, 3 + scenario_index, trial, 1)), {"proposal": "optimal"})]
            if scenario == "outliers_5pct":
                candidates.append(("PF_known_mixture", particle_filter,
                                   (data.y, Parameters(), max(1024, config.particles),
                                    _rng(config, 3 + scenario_index, trial, 2)),
                                   {"outlier_probability": 0.05, "outlier_scale": 8}))
            for method, function, args, kwargs in candidates:
                result, seconds = _timed(function, *args, **kwargs)
                rows.append({"scenario": scenario, "trial": trial, "method": method,
                             "data_seed": seed_for(config.seed, 3 + scenario_index, trial),
                             "observed_fraction": float(np.isfinite(data.y).mean()),
                             "contaminated_fraction": float(data.contaminated.mean()),
                             "seconds": seconds, **state_metrics(data.x, result)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "stress_trials.csv", index=False)
    summarize_states(frame, ("scenario", "method")).to_csv(output / "stress_summary.csv", index=False)
    print("Missing-data and contamination studies complete", flush=True)


def holdout_study(config, output):
    rows = []
    for trial in range(config.holdout_trials):
        data = simulate(config.train_length + config.test_length, Parameters(), _rng(config, 7, trial))
        train, train_seconds = _timed(fit_mle, data.y[:config.train_length])
        full, full_seconds = _timed(fit_mle, data.y)
        for method, parameters, fit_seconds, success in [
            ("known", Parameters(), 0, True),
            ("train_only_MLE", train.parameters, train_seconds, train.success),
            ("full_sample_MLE_lookahead", full.parameters, full_seconds, full.success),
        ]:
            filtered, seconds = _timed(kalman_filter, data.y, parameters)
            lower, upper = filtered.interval()
            test = slice(config.train_length, None)
            rows.append({"trial": trial, "method": method, "fit_success": success,
                         "data_seed": seed_for(config.seed, 7, trial),
                         "mse": float(np.mean((filtered.mean[test] - data.x[test])**2)),
                         "rmse": float(np.sqrt(np.mean((filtered.mean[test] - data.x[test])**2))),
                         "coverage": float(np.mean((lower[test] <= data.x[test]) & (data.x[test] <= upper[test]))),
                         "interval_width": float(np.mean(upper[test] - lower[test])),
                         "mean_negative_log_predictive_density": float(-filtered.loglikelihood_terms[test].mean()),
                         "seconds": seconds + fit_seconds})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "holdout_trials.csv", index=False)
    summarize_states(frame).to_csv(output / "holdout_summary.csv", index=False)
    print("Chronological holdout study complete", flush=True)


def nonlinear_study(config, output):
    rows = []
    p = SVParameters()
    for trial in range(config.sv_trials):
        h, y = simulate_sv(config.length, p, _rng(config, 8, trial))
        grid, grid_seconds = _timed(grid_filter, y, p, grid_size=config.grid_size)
        pf, pf_seconds = _timed(sv_particle_filter, y, p, config.sv_particles, _rng(config, 8, trial, 1))
        # Exact first two moments of log(epsilon^2), epsilon ~ N(0,1).
        centered = np.log(np.maximum(y**2, np.finfo(float).tiny)) + np.euler_gamma + np.log(2) - p.mu
        approximate, approx_seconds = _timed(kalman_filter, centered, Parameters(p.phi, p.q, np.pi**2 / 2))
        approximate = replace(approximate, mean=approximate.mean + p.mu)
        for method, result, seconds in [("grid_reference", grid, grid_seconds),
                                         ("bootstrap_PF", pf, pf_seconds),
                                         ("log_squared_KF_approximation", approximate, approx_seconds)]:
            rows.append({"trial": trial, "method": method, "data_seed": seed_for(config.seed, 8, trial),
                         "seconds": seconds, "mse_to_grid_mean": float(np.mean((result.mean - grid.mean)**2)),
                         "grid_boundary_mass": grid.boundary_mass, **state_metrics(h, result)})
        if trial == 0:
            refined = grid_filter(y, p, grid_size=2 * config.grid_size - 1)
            check = {"coarse_points": config.grid_size, "fine_points": 2 * config.grid_size - 1,
                     "maximum_absolute_mean_difference": float(np.max(np.abs(refined.mean - grid.mean))),
                     "loglikelihood_difference": refined.loglikelihood - grid.loglikelihood,
                     "coarse_boundary_mass": grid.boundary_mass, "fine_boundary_mass": refined.boundary_mass}
            (output / "sv_grid_check.json").write_text(json.dumps(check, indent=2) + "\n")
            pd.DataFrame({"t": np.arange(config.length), "h": h, "y": y,
                          "grid_mean": grid.mean, "grid_lower": grid.lower, "grid_upper": grid.upper,
                          "particle_mean": pf.mean, "log_squared_KF_mean": approximate.mean}).to_csv(
                output / "sv_example.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "sv_trials.csv", index=False)
    summarize_states(frame).to_csv(output / "sv_summary.csv", index=False)
    print("Nonlinear stochastic-volatility study complete", flush=True)


def bootstrap_study(config, output):
    """One dataset's parametric refit bootstrap, not a coverage validation study."""
    data = simulate(config.length, Parameters(), _rng(config, 9, 0))
    fit = fit_mle(data.y)
    rows = []
    for replication in range(config.bootstrap_replications):
        sample = simulate(config.length, fit.parameters, _rng(config, 9, replication, 1))
        refit = fit_mle(sample.y)
        rows.append({"replication": replication, "data_seed": seed_for(config.seed, 9, replication, 1),
                     "success": refit.success, "at_boundary": refit.at_boundary,
                     **{name: getattr(refit.parameters, name) for name in ("phi", "q", "r")}})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "bootstrap_refits.csv", index=False)
    summary = {"base_data_seed": seed_for(config.seed, 9, 0), "base_fit": fit.parameters.as_dict(),
               "wald_lower": fit.lower.tolist(), "wald_upper": fit.upper.tolist(),
               "percentile_lower": frame[["phi", "q", "r"]].quantile(0.025).tolist(),
               "percentile_upper": frame[["phi", "q", "r"]].quantile(0.975).tolist(),
               "successful_refits": int(frame.success.sum()),
               "note": "Parametric bootstrap parameter intervals; not a parameter posterior or state credible interval."}
    (output / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def example_and_ridge(config, output):
    p = Parameters()
    data = simulate(config.length, p, _rng(config, 10, 0))
    kf = kalman_filter(data.y, p)
    smoother = rts_smoother(kf, p)
    pf = particle_filter(data.y, p, config.particles, _rng(config, 10, 0, 1), proposal="optimal")
    low, high = kf.interval()
    sl, su = smoother.interval()
    pd.DataFrame({"t": np.arange(config.length), "x": data.x, "y": data.y,
                  "kf_mean": kf.mean, "kf_lower": low, "kf_upper": high,
                  "rts_mean": smoother.mean, "rts_lower": sl, "rts_upper": su,
                  "pf_mean": pf.mean, "ess_fraction": pf.ess / config.particles,
                  "resampled": pf.resampled}).to_csv(output / "example_path.csv", index=False)
    rows = []
    for scale in np.geomspace(0.15, 7, 101):
        changed = rescale_latent(p, float(scale))
        result = kalman_filter(data.y, changed)
        rows.append({"scale": scale, "alpha": changed.alpha, "q": changed.q,
                     "likelihood_difference": result.loglikelihood - kf.loglikelihood,
                     "latent_rmse_in_wrong_units": float(np.sqrt(np.mean((result.mean - data.x)**2))),
                     "latent_rmse_after_unit_correction": float(np.sqrt(np.mean((result.mean / scale - data.x)**2)))})
    pd.DataFrame(rows).to_csv(output / "identifiability_ridge.csv", index=False)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_studies(config: StudyConfig, output: Path):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new path.")
    output.mkdir(parents=True, exist_ok=True)
    if config.workers < 1:
        raise ValueError("workers must be positive")
    started = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    phases = {}
    for name, function in [("core", core_study), ("convergence", convergence_study),
                           ("stress", stress_study), ("holdout", holdout_study),
                           ("nonlinear", nonlinear_study), ("bootstrap", bootstrap_study),
                           ("example", example_and_ridge)]:
        phase = perf_counter()
        function(config, output)
        phases[name] = perf_counter() - phase
    source_root = Path(__file__).parent
    repository = source_root.parent.parent
    commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip() or None
    manifest = {"started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": perf_counter() - start, "phase_seconds": phases,
                "config": asdict(config), "truth": Parameters().as_dict(),
                "stochastic_volatility_truth": asdict(SVParameters()),
                "seed_scheme": "SeedSequence([master_seed, study_id, trial_id, component_id]).generate_state(1, uint64)",
                "study_ids": {"core": 1, "convergence": 2, "matched_stress": 3,
                              "missing_random": 4, "missing_block": 5, "contamination": 6,
                              "holdout": 7, "stochastic_volatility": 8, "bootstrap": 9, "example": 10},
                "python": platform.python_version(), "platform": platform.platform(),
                "dependencies": {name: importlib.metadata.version(name)
                                 for name in ("numpy", "scipy", "pandas", "matplotlib")},
                "git_commit_at_execution": commit,
                "source_sha256": {str(path.relative_to(repository)): file_hash(path)
                                  for path in sorted(source_root.glob("*.py"))},
                "result_sha256": {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.is_file()},
                "timing_note": "Per-method wall-clock timings are machine-specific; estimation time is included for fitted methods."
                }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
