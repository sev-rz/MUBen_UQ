# Conformal-prediction extension protocol (frozen 2026-10-08, before any Tier-B result and before any conformal computation)

## Start condition
Run only after Tier B is complete (expanded_v4/results/TIERB_SUMMARY.md exists). Uses saved predictions only; no retraining.

## Questions
Q1. At equal guaranteed coverage, which uncertainty method gives the most informative (narrowest) conformal intervals / smallest sets?
Q2. Does marginal coverage hold on scaffold splits, where calibration and test chemistry differ, and on the novel subgroup?
Q3. Does novelty-conditional (Mondrian) conformal restore subgroup coverage, and at what width cost?

## Inputs
- Tier B predictions: DNN-rdkit and ChemBERTa, 8 datasets, scaffold and random splits, all methods.
- Phase 2 predictions (expanded_v2): DNN Laplace and Novelty Laplace on bace, bbbp, esol (scaffold). Laplace comparisons are limited to these.

## Calibration
- Calibration set = the full validation split of each run. It was already used for early stopping and tuning, so the guarantee is
  approximate; this is stated, not hidden. Empirical coverage is always measured on the untouched test split.
- Levels: alpha = 0.1 (primary) and 0.2 (secondary).

## Regression (esol, freesolv, lipo, qm7)
- Normalised split conformal: score = |y - mu(x)| / sigma_m(x), sigma_m from each method's predictive SD (Deterministic: variance head).
- Also an unnormalised baseline (sigma = 1) so the value of each uncertainty estimate is measured against "no uncertainty".
- Report: empirical coverage, mean and median interval width in original units, per seed/split; mean ± SD over repeats.

## Classification (bace, bbbp, clintox, tox21; per task, missing labels skipped, then averaged over tasks)
- LAC score 1 - p(true class). Primary: marginal; secondary: class-conditional (Mondrian by label) for imbalanced tasks.
- Report: coverage, mean set size, fraction of singleton sets, accuracy among singletons, fraction of empty sets.

## Novelty (Q2, Q3)
- d(x) = max Tanimoto similarity to the training set (Morgan radius 2, 2048 bits, chirality off; same as expanded_v2).
- Novel: d < 0.4; familiar: d >= 0.4 (threshold from expanded_v2, not tuned).
- Report test coverage separately per subgroup under marginal calibration (Q2); then Mondrian calibration per subgroup (Q3).
  A subgroup with fewer than 20 calibration molecules is reported as not estimable, not pooled.

## Decision rules (fixed now)
- Coverage is called "held" when the 95% binomial interval of test coverage contains 1 - alpha; "under-covered" when it lies below.
- Method comparisons on width/set size are paired by seed (same molecules); "consistent" only if all repeats agree in sign.
- No method, threshold, score or alpha is added or changed after seeing results.

## Outputs
results/conformal_regression.csv, results/conformal_classification.csv, results/conformal_novelty.csv,
figures (width by method at alpha=0.1; coverage novel vs familiar), CONFORMAL_SUMMARY.md, then Report 2 section "E5 Conformal guarantees".
