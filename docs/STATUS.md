# Status (15 October 2026)

## Done
- Reproduction of MUBen with DNN, ChemBERTa, GIN (scaffold and random splits; eight UQ methods) and Uni-Mol (scaffold; eight methods), against the published tables, with bridge checks against the authors' own trainers (Report 1, H1 and H2).
- Last-layer Laplace on all four backbones (three repeats, then 10 to 12 networks per dataset), the schedule test, the recalibration baseline, wide prior grid, evidence-based prior and Laplace on every ensemble member (Report 2, H1 to H5).
- Split conformal prediction on all four backbones, shift-aware checks, few-label and similarity-weighted calibration, conformal under the linear schedule (Report 2, H6 and H7).

## Open (not run)
- Fresh-data confirmation: new scaffold splits with claims fixed in advance (would remove the "exploratory" label).
- The schedule test for the DNN, GIN regression and Uni-Mol (only ChemBERTa, and the GIN classification bridge, were tested).
- Uni-Mol in the few-label and similarity-weighted conformal analyses; heteroscedastic Laplace.
- The GIN SGLD divergence on regression (seed 100); the cause of the published DNN QM7 and ChemBERTa ESOL numbers (the authors release only aggregated tables).
- GROVER (checkpoint link unreachable) and TorchMD-NET were not run; evidential deep learning was excluded by decision.

## Known limits
- The test sets were used for the baselines before the extensions, so every extension result is exploratory.
- Three repeats for the reproduction and most conformal analyses; scaffold splits are the authors' fixed files, so split variance is not measured.
- The conformal calibration set (the validation split) also served early stopping and Laplace tuning, so the guarantee is approximate.
- The pooled ensemble variance in the published design is an average of member variances (no between-member spread); ensemble comparisons are conditional on that aggregation.
- `wqhat` (weighted quantile, `stages/expanded_v16_round1/cp_common.py`) has a floating-point edge at integer boundaries of `(n + 1)(1 - a)`: it can return infinity (the conservative full set) where the unweighted rank would return the largest score. The reported round-1 results were produced with this code; the effect is conservative and rare (see `tests/test_core.py`).
