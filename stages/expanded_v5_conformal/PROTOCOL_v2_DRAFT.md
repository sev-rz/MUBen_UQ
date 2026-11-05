# Conformal-prediction extension, amendment v2 (DRAFT; to be frozen before any conformal computation on the test split)
Amends expanded_v5_conformal/PROTOCOL.md (frozen 2026-10-08, sha256 in PROTOCOL.sha256), which stays valid for everything it covers. Student's instruction (2026-10-10): after the GIN results and Laplace on GIN, add conformal prediction to everything done so far, as the strong extension point of the project.

## 1. What is added to the scope
- Backbones: DNN-rdkit and ChemBERTa (Tier B, all methods, as in v5), plus GIN (reproduction, expanded_v8_gin, all methods).
- Laplace family for each of the three backbones, all eight datasets, scaffold and random-0..2: Laplace, Laplace constant, Novelty only, Novelty Laplace (stages 1-3: expanded_v6_laplace, expanded_v6_laplace/chemberta, expanded_v8_gin/laplace).
- This replaces the v5 sentence "Laplace comparisons are limited to bace/bbbp/esol (expanded_v2)". The expanded_v2 Laplace predictions (preds/DNN-v2) are kept only as a consistency cross-check against stage 1 on those three datasets, not in the primary tables.
- Unchanged from v5: calibration set = the full validation split of each run (guarantee approximate, stated), alpha 0.1 primary and 0.2 secondary, regression normalised and unnormalised scores, classification LAC (marginal primary, class-conditional secondary), novelty subgroup d < 0.4 similarity with Mondrian calibration, minimum 20 calibration molecules per subgroup, coverage verdicts by the 95% interval of test coverage (held / under-covered / over-covered), no change of method, score, threshold or alpha after seeing results. Test predictions are the ones already produced by each stage; nothing is retrained.

## 2. Questions
Q1, Q2, Q3 as in v5 (narrowest intervals or smallest sets at equal coverage; does coverage hold on scaffold splits and on the novel subgroup; does Mondrian conditioning restore subgroup coverage and at what width cost), now over three backbones.
Q4 (new, the extension point): does conformal prediction on top of a Laplace predictive distribution give narrower intervals (regression) or smaller sets (classification) than conformal prediction on the Deterministic network and on the Ensembles, at the same coverage?
Q5 (new, descriptive): how do coverage and width of the Deterministic network under scaffold shift compare across DNN, ChemBERTa and GIN?

## 3. Primary comparisons (multiplicity limited on purpose; everything else is descriptive)
Per backbone, at alpha 0.1, paired by seed (scaffold) or split (random): (P1) normalised-score mean width (regression) or mean set size (classification), Laplace minus Deterministic; (P2) Laplace minus Ensembles; (P3) on scaffold only, Novelty Laplace minus Laplace constant (the control). A sign is "consistent" only if all three repeats agree; an overall statement needs the same consistent sign on at least 6 of 8 datasets for that split type and backbone. Coverage is reported next to every width: a width comparison is interpreted only where both methods hold coverage.
Q2 and Q3 summaries: share of (dataset, repeat) cases with verdict held, by backbone, method and subgroup. No formal tests; counts and signs as in v5, v6.

## 4. Inputs to prepare (validation data only; no test label used)
Validation predictions of GIN (valid_preds_gin.py) and of the Laplace family per backbone (valid_preds_laplace.py), completion of the Tier-B ChemBERTa validation predictions (job 223577 stopped after the DNN step), staged test copies for the Laplace family in preds_test/<BACKBONE>-Laplace/. The Laplace family is stored as separate backbone folders (DNN-Laplace, ChemBERTa-Laplace, GIN-Laplace) so that incomplete Tier-B folders are not mixed with it.

## 5. Checks before the full run
conformal.py self-test (validation halves) on every new group, crepes agreement re-run on a sample (as in v5: smoothing off), coverage of the self-test within the expected range. The full run starts only when all prepared groups pass.

## 6. Outputs
results/conformal_regression.csv, results/conformal_classification.csv, results/conformal_novelty.csv, figures (width by method at alpha 0.1 per backbone; coverage novel vs familiar), CONFORMAL_SUMMARY.md, then the extension section of Report 2.
