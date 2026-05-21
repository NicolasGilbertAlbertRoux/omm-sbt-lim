# Reproducibility plan

## Objective

Provide a clean, public, reproducible benchmark package for the LIM identity-reconstruction experiments within OMM--SBT.

## Formal experiment names

- Preliminary memory-necessity anti-cheat experiment.
- Observable-alias memory-necessity experiment.
- Strict double-blind memory-necessity benchmark.
- Latent-memory calibration scan.

## Publication-grade requirements

1. Keep public experiment names descriptive rather than tied to exploratory numbering.
2. Preserve deterministic seeds and parameter grids inside each script.
3. Keep compact summaries under `results/summaries/`.
4. Keep raw outputs only when computationally feasible.
5. Use `scripts/inspect_available_results.py` to list available result files and their columns.
6. Use `pytest` for basic metric and package smoke tests.
7. Archive exact code, summary CSVs, and environment files with any manuscript release.

## Falsification orientation

The benchmark is designed to fail cleanly if observable-only reconstruction, false latent controls, or shuffled-label controls perform as well as coherent latent memory under strict aliasing.
