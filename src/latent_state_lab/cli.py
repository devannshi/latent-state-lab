"""Reproduce simulations or regenerate figures from existing raw results."""

import argparse
from pathlib import Path

from .experiments import StudyConfig, run_studies
from .plotting import create_figures
from .reporting import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a predeclared experiment suite")
    run.add_argument("--profile", choices=["smoke", "quick", "full"], default="quick")
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--seed", type=int, default=20261117)
    run.add_argument("--workers", type=int, default=1)
    render = commands.add_parser("render", help="Regenerate plots and report from saved raw outputs")
    render.add_argument("results", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        config = StudyConfig.for_profile(arguments.profile, seed=arguments.seed, workers=arguments.workers)
        output = arguments.output or Path("results") / arguments.profile
        run_studies(config, output)
    else:
        output = arguments.results
    create_figures(output)
    write_report(output)
    print(f"Report: {(output / 'REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
