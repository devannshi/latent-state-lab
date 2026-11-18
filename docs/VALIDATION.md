# Validation record

The committed full run executed the current source with 1,000 independent core
trials and every configured secondary study. Its measured experiment runtime was
81.78 seconds. See `results/full/manifest.json` for timestamps,
package versions, seeds, source hashes, result hashes, and phase timings.

Completed checks:

- 56 numerical/statistical tests passed (`pytest -q`).
- `ruff check src tests scripts` passed.
- A full final-source reproduction matched every original parameter estimate,
  parameter summary, paired risk difference, and core state metric exactly;
  measured runtime columns were excluded from equality checks.
- All recorded source/result hashes, row counts, interval accounting, and seven
  figures passed `scripts/verify_results.py`.
- Editable package construction and installation succeeded with declared build
  tooling; the installed `latent-state-lab --help` command ran successfully.
- All nine tutorial code cells executed; notebook schema validation passed.
- All seven generated figures were visually inspected. A crowded logarithmic
  axis was corrected before the final run. Both PNG and vector PDF are retained.

The initial smoke run and initial full-run artifacts were development checks;
only the final full-profile results are intended for version control. The repeated
full run used the same predeclared seeds and data-generating parameters, not a
selected replacement seed. No failed or unfavorable trial was removed.
