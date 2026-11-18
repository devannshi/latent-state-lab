"""Check saved study dimensions, statistical accounting, and recorded hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results/full"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.results.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["config"]
    for relative, expected in manifest["source_sha256"].items():
        assert digest(root / relative) == expected, f"Source hash mismatch: {relative}"
    for relative, expected in manifest["result_sha256"].items():
        assert digest(output / relative) == expected, f"Result hash mismatch: {relative}"
    state = pd.read_csv(output / "state_trials.csv")
    parameter = pd.read_csv(output / "parameter_trials.csv")
    convergence = pd.read_csv(output / "convergence_trials.csv")
    assert state.trial.nunique() == config["trials"]
    assert len(state) == 7 * config["trials"]
    assert len(parameter) == 6 * config["trials"]
    assert len(convergence) == 2 * config["convergence_trials"] * len(config["particle_counts"])
    assert state.groupby("trial").method.nunique().eq(7).all()
    assert state.groupby("trial").data_seed.nunique().eq(1).all()
    assert state[["mse", "rmse", "coverage", "interval_width"]].notna().all().all()
    assert state.coverage.between(0, 1).all()
    np.testing.assert_allclose(state.rmse**2, state.mse, atol=1e-12)
    summaries = pd.read_csv(output / "parameter_summary.csv")
    for row in summaries.itertuples():
        group = parameter.query("method == @row.method and parameter == @row.parameter")
        valid = group.ci_valid
        covered = (group.lower <= row.truth) & (row.truth <= group.upper) & valid
        np.testing.assert_allclose(row.coverage_all_trials_invalid_as_miss, covered.mean())
        assert row.ci_valid_count == int(valid.sum())
    diagnostics = (output / "optimizer_diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics) == config["trials"]
    assert len(list((output / "figures").glob("*.png"))) == 7
    assert (output / "REPORT.md").is_file()
    print(f"Verified {config['trials']} core trials, all recorded source/result hashes, "
          "interval accounting, and seven figures.")


if __name__ == "__main__":
    main()
