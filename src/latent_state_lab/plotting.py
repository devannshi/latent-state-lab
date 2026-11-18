"""Publication-oriented plots generated only from saved experiment outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, PercentFormatter

BLUE, TEAL, ORANGE, GREY = "#173B5F", "#008A8A", "#CB604B", "#7D8792"
COLORS = {"bootstrap": BLUE, "optimal": TEAL}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
                         "legend.frameon": False, "figure.dpi": 120,
                         "savefig.dpi": 180, "pdf.fonttype": 42})


def _save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(results: Path) -> list[Path]:
    results = Path(results)
    destination = results / "figures"
    destination.mkdir(exist_ok=True)
    _style()
    example = pd.read_csv(results / "example_path.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, layout="constrained")
    for ax, prefix, label in [(axes[0], "kf", "Filtering: observations through time t"),
                              (axes[1], "rts", "Smoothing: the complete observation series")]:
        ax.fill_between(example.t, example[f"{prefix}_lower"], example[f"{prefix}_upper"],
                        color=TEAL, alpha=0.17, label="95% conditional state interval")
        ax.plot(example.t, example.x, color=BLUE, linewidth=1.6, label="Latent truth")
        ax.plot(example.t, example[f"{prefix}_mean"], color=TEAL, linewidth=1.3, label="Exact conditional mean")
        ax.scatter(example.t, example.y, s=7, color=GREY, alpha=0.4, label="Noisy observations", zorder=1)
        ax.set(ylabel="State / observation", title=label)
    axes[0].plot(example.t, example.pf_mean, color=ORANGE, linewidth=0.8,
                 linestyle="--", label="Optimal-proposal particle mean")
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    axes[1].set_xlabel("Time index")
    fig.suptitle("Latent-state recovery with known parameters", fontsize=15, fontweight="bold")
    _save(fig, destination, "01_state_recovery")

    summary = pd.read_csv(results / "state_summary.csv")
    labels = {"KF_known": "Exact KF · known θ", "PF_bootstrap_known": "Bootstrap PF · known θ",
              "PF_optimal_known": "Optimal PF · known θ", "KF_full_sample_MLE": "KF · full-sample MLE",
              "KF_oracle_parameters": "KF · oracle parameters", "RTS_known": "RTS · known θ",
              "RTS_full_sample_MLE": "RTS · full-sample MLE"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), layout="constrained",
                              gridspec_kw={"width_ratios": [1.4, 1]})
    for ax, methods, title in [
        (axes[0], list(labels)[:5], "Forward state updates"),
        (axes[1], list(labels)[5:], "Full-series smoothing"),
    ]:
        subset = summary.query("metric == 'mse'").set_index("method").loc[methods]
        positions = np.arange(len(methods))
        ax.errorbar(subset["mean"], positions,
                    xerr=[subset["mean"] - subset.lower, subset.upper - subset["mean"]],
                    fmt="o", color=TEAL, capsize=4, markersize=6)
        ax.set_yticks(positions, [labels[method] for method in methods], fontsize=9)
        ax.invert_yaxis()
        ax.set(xlabel="Mean squared state error", title=title)
        ax.margins(y=0.25)
    fig.suptitle("State risk across independent trials", fontsize=15, fontweight="bold")
    fig.supxlabel("Bars: 95% Monte Carlo intervals across trials. Full-sample MLE is retrospective.", fontsize=9)
    _save(fig, destination, "02_state_risk")

    parameters = pd.read_csv(results / "parameter_trials.csv")
    parameter_summary = pd.read_csv(results / "parameter_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7), layout="constrained")
    for j, (name, truth) in enumerate((("phi", 0.85), ("q", 0.16), ("r", 0.49))):
        for method, color, label in [("marginal_MLE", TEAL, "Observation-only MLE"),
                                      ("oracle_conditional_MLE", BLUE, "Complete-data oracle")]:
            values = parameters.query("parameter == @name and method == @method").estimate
            axes[0, j].hist(values, bins=35, density=True, histtype="step", color=color, linewidth=1.6, label=label)
        axes[0, j].axvline(truth, color=ORANGE, linestyle="--", label="True value")
        axes[0, j].set(title={"phi": "Persistence φ", "q": "State variance q", "r": "Observation variance r"}[name],
                        xlabel="Parameter estimate", ylabel="Density" if j == 0 else "")
        subset = parameter_summary.query("parameter == @name").set_index("method")
        for k, method in enumerate(("marginal_MLE", "oracle_conditional_MLE")):
            row = subset.loc[method]
            axes[1, j].bar(k - 0.15, row.coverage_given_valid_ci, width=0.28, color=TEAL,
                           label="Among available intervals" if k == 0 else None)
            axes[1, j].bar(k + 0.15, row.coverage_all_trials_invalid_as_miss, width=0.28, color=BLUE,
                           label="All trials, invalid = miss" if k == 0 else None)
            axes[1, j].annotate(f"available: {row.ci_available_fraction:.1%}", (k, 0.03),
                                ha="center", fontsize=8, color="white")
        axes[1, j].axhline(0.95, color=ORANGE, linestyle="--", linewidth=1)
        axes[1, j].set(ylim=(0, 1.02), ylabel="95% parameter-interval coverage" if j == 0 else "")
        axes[1, j].set_xticks([0, 1], ["Marginal MLE", "Oracle"], fontsize=9)
        axes[1, j].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    fig.suptitle("Parameter recovery and finite-sample interval calibration", fontsize=15, fontweight="bold")
    _save(fig, destination, "03_parameter_recovery")

    convergence = pd.read_csv(results / "convergence_trials.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout="constrained")
    for proposal, group in convergence.groupby("proposal"):
        aggregate = group.groupby("particles").agg(mse=("approximation_mse", "mean"),
                                                    mse_sem=("approximation_mse", "sem"),
                                                    seconds=("seconds", "median"),
                                                    ess=("mean_ess_fraction", "mean"))
        axes[0].loglog(aggregate.index, np.sqrt(aggregate.mse), "o-", color=COLORS[proposal], label=proposal)
        axes[0].fill_between(aggregate.index,
                             np.sqrt(np.maximum(aggregate.mse - 1.96 * aggregate.mse_sem, 1e-15)),
                             np.sqrt(aggregate.mse + 1.96 * aggregate.mse_sem),
                             color=COLORS[proposal], alpha=0.12)
        axes[1].loglog(aggregate.seconds * 1000, np.sqrt(aggregate.mse), "o-", color=COLORS[proposal], label=proposal)
        axes[2].semilogx(aggregate.index, aggregate.ess, "o-", color=COLORS[proposal], label=proposal)
    counts = np.array(sorted(convergence.particles.unique()))
    first = np.sqrt(convergence.query("particles == @counts[0]").approximation_mse.mean())
    axes[0].loglog(counts, first * np.sqrt(counts[0] / counts), "--", color=GREY, label="N⁻¹ᐟ² reference")
    axes[0].set(xlabel="Number of particles", ylabel="RMSE to exact KF mean", title="Monte Carlo approximation error")
    axes[1].set(xlabel="Median wall time per series (ms)", ylabel="RMSE to exact KF mean", title="Compute / accuracy tradeoff")
    axes[2].set(xlabel="Number of particles", ylabel="Mean ESS / N, before resampling", title="Weight concentration", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.suptitle("Particle filters converge toward the exact benchmark", fontsize=15, fontweight="bold")
    fig.supxlabel("Shading: approximate 95% Monte Carlo intervals across independent series.", fontsize=9)
    _save(fig, destination, "04_particle_convergence")

    ridge = pd.read_csv(results / "identifiability_ridge.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    axes[0].plot(ridge.alpha, ridge.q, color=TEAL)
    axes[0].scatter([1], [0.16], color=ORANGE, zorder=5)
    axes[0].set(xlabel="Observation loading α", ylabel="State innovation variance q", title="Equivalent models: α²q = constant")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].semilogx(ridge.scale, ridge.likelihood_difference, color=BLUE)
    axes[1].set(xlabel="Latent-state scale c", ylabel="Log likelihood difference", title="Observation likelihood is unchanged")
    numerical_scale = max(float(ridge.likelihood_difference.abs().max()), 1e-12)
    axes[1].set_ylim(-2 * numerical_scale, 2 * numerical_scale)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[2].loglog(ridge.scale, ridge.latent_rmse_in_wrong_units, color=ORANGE, label="Compare incompatible state units")
    axes[2].loglog(ridge.scale, ridge.latent_rmse_after_unit_correction, color=TEAL, label="Correct state units")
    axes[2].set(xlabel="Latent-state scale c", ylabel="Latent-state RMSE", title="State RMSE requires a normalization")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("The scale-identification problem", fontsize=15, fontweight="bold")
    _save(fig, destination, "05_identifiability")

    stress = pd.read_csv(results / "stress_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    scenarios = ["matched", "missing_30pct", "missing_block", "outliers_5pct"]
    method_colors = {"KF_Gaussian": BLUE, "PF_Gaussian": TEAL, "PF_known_mixture": ORANGE}
    for ax, metric, label in [(axes[0], "mse", "Mean squared state error"),
                               (axes[1], "coverage", "95% state-interval coverage")]:
        for offset, (method, color) in enumerate(method_colors.items()):
            group = stress.query("method == @method and metric == @metric").set_index("scenario")
            positions = np.array([scenarios.index(scenario) for scenario in group.index]) + (offset - 1) * 0.17
            ax.errorbar(positions, group["mean"], yerr=[group["mean"] - group.lower, group.upper - group["mean"]],
                        fmt="o", capsize=3, color=color, label=method.replace("_", " "))
        ax.set_xticks(range(4), ["Matched", "30% missing", "Missing block", "5% outliers"], rotation=15)
        ax.set_ylabel(label)
    axes[1].axhline(0.95, color=GREY, linestyle="--")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    fig.suptitle("Missing information and model misspecification have different effects", fontsize=14, fontweight="bold")
    fig.supxlabel("The mixture PF is supplied the true contamination model; this changes the observation assumptions.", fontsize=9)
    _save(fig, destination, "06_stress_tests")

    sv = pd.read_csv(results / "sv_example.csv")
    sv_summary = pd.read_csv(results / "sv_summary.csv").query("metric == 'mse'").set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained", gridspec_kw={"width_ratios": [1.7, 1]})
    axes[0].fill_between(sv.t, sv.grid_lower, sv.grid_upper, color=TEAL, alpha=0.17)
    axes[0].plot(sv.t, sv.h, color=BLUE, label="True log variance", linewidth=1.4)
    axes[0].plot(sv.t, sv.grid_mean, color=TEAL, label="Numerical grid reference")
    axes[0].plot(sv.t, sv.particle_mean, color=ORANGE, linestyle="--", label="Bootstrap particle filter", linewidth=0.9)
    axes[0].set(xlabel="Time index", ylabel="Latent log variance h", title="A nonlinear observation model")
    axes[0].legend(fontsize=8)
    methods = ["grid_reference", "bootstrap_PF", "log_squared_KF_approximation"]
    group = sv_summary.loc[methods]
    axes[1].errorbar(group["mean"], np.arange(3),
                     xerr=[group["mean"] - group.lower, group.upper - group["mean"]],
                     fmt="o", color=TEAL, capsize=4)
    axes[1].set_yticks(range(3), ["Grid reference", "Particle filter", "Log-squared KF"], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].margins(y=0.3)
    axes[1].set(xlabel="Mean squared error in h", title="Risk across independent trials")
    fig.suptitle("Stochastic volatility: where particles add modeling flexibility", fontsize=14, fontweight="bold")
    _save(fig, destination, "07_stochastic_volatility")
    return sorted(destination.glob("*.png"))
