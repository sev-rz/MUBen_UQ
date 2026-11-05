# Conformal-prediction extension, amendment v2 (FROZEN 2026-10-10 15:35Z, before any conformal computation on the test split)
Amends expanded_v5_conformal/PROTOCOL.md (frozen 2026-10-08, sha256 in PROTOCOL.sha256), which stays valid for everything it covers. Student's instruction (2026-10-10): after the GIN results and Laplace on GIN, add conformal prediction to everything done so far, as the strong extension point of the project. This file replaces PROTOCOL_v2_DRAFT.md (kept for the record); the differences are listed in section 7.

## 1. What is added to the scope
- Backbones: DNN-rdkit and ChemBERTa (Tier B, all methods, as in v5), plus GIN (reproduction, expanded_v8_gin, all methods).
- Laplace family for each of the three backbones, all eight datasets, scaffold and random-0..2: Laplace, Laplace constant, Novelty only, Novelty Laplace (stages 1-3: expanded_v6_laplace, expanded_v6_laplace/chemberta, expanded_v8_gin/laplace).
- This replaces the v5 sentence "Laplace comparisons are limited to bace/bbbp/esol (expanded_v2)". The expanded_v2 Laplace predictions (preds/DNN-v2) are kept only as a consistency cross-check against stage 1 on those three datasets, not in the primary tables.
- Unchanged from v5: calibration set = the full validation split of each run (guarantee approximate, stated), alpha 0.1 primary and 0.2 secondary, regression normalised and unnormalised scores, classification LAC (marginal primary, class-conditional secondary), novelty subgroup d < 0.4 similarity with Mondrian calibration, minimum 20 calibration molecules per subgroup, coverage verdicts by the 95% interval of test coverage (held / under-covered / over-covered), no change of method, score, threshold or alpha after seeing results. Test predictions are the ones already produced by each stage; nothing is retrained.

## 2. Questions
Q1, Q2, Q3 as in v5 (narrowest intervals or smallest sets at equal coverage; does coverage hold on scaffold splits and on the novel subgroup; does Mondrian conditioning restore subgroup coverage and at what width cost), now over three backbones.
Q4 (the extension point): does conformal prediction on top of a Laplace predictive distribution give narrower intervals (regression) or smaller sets (classification) than conformal prediction on the Deterministic network and on the Ensembles, at the same coverage?
Q5 (descriptive): how do coverage and width of the Deterministic network under scaffold shift compare across DNN, ChemBERTa and GIN?

## 3. Primary comparisons and decision rules (multiplicity limited on purpose; everything else is descriptive)
Quantity at alpha 0.1: regression = mean interval width of the normalised score; classification = mean LAC set size; tasks averaged as in conformal_v2.py. Pairing: same backbone, split, dataset and seed (scaffold seeds 0/100/200; random: one seed per split); scaffold Ensembles are one pooled value per dataset.
Contrasts (negative = narrower / smaller): P1 Laplace minus Deterministic; P2 Laplace minus Ensembles; P3 (scaffold only) Novelty Laplace minus Laplace constant, the control.
- A dataset has a consistent sign only if all repeats agree.
- A case "holds coverage" if more than half of its tasks have verdict "held" (verdict_held_tasks * 2 > tasks).
- A dataset is interpretable for a contrast only if both methods hold coverage in every repeat; widths are interpreted only on interpretable datasets.
- An overall statement for a backbone and split type needs the same consistent sign on at least 6 of 8 interpretable datasets; otherwise "no overall statement". Counts over all datasets (not only interpretable) are reported next to it, labelled descriptive.
- Q2 and Q3: share of cases with verdict held (overall, and by novelty subgroup with at least 20 test molecules, Wilson interval around the subgroup coverage) by backbone, method and split type; Mondrian against marginal calibration on the novel subgroup (coverage gap to 1 - alpha; width cost = Mondrian novel-subgroup width relative to the overall marginal width, set size difference for classification; not like-for-like, read as an indication). No formal tests; counts and signs as in v5, v6.

## 4. Inputs (validation data only; no test label used)
Six prediction families, 32 groups each (8 datasets x scaffold, random-0, random-1, random-2), under preds/ with DONE markers: DNN, ChemBERTa, GIN (Tier-B and reproduction methods) and DNN-Laplace, ChemBERTa-Laplace, GIN-Laplace (Laplace family, with staged test copies in preds_test/). preds/DNN-v2 is a cross-check only. Test predictions are the saved ones of each stage (expanded_v4/results, expanded_v8_gin/results, preds_test/).
Computed on the same device as the test predictions: ChemBERTa validation predictions on the GPU (valid_preds_gpu.py; journal n0081), GIN on the GPU, DNN on CPU as in Tier B.

## 5. Checks before the full run (all passed before this file was written)
- Input coverage (check_inputs_v2.py, journal cell c0257): six families (DNN, ChemBERTa, GIN, DNN-Laplace, ChemBERTa-Laplace, GIN-Laplace) with 32 of 32 groups each; 320 validation files for each of DNN, ChemBERTa and GIN and 192 for each Laplace family; 0 files without a test counterpart. preds/DNN-v2 (3 groups) is a cross-check only.
- Self-test of conformal_v2.py on validation halves, test split untouched (c0257, c0258): 2,844 regression and 3,480 classification result rows; mean coverage at alpha 0.1 between 0.911 and 0.921 (regression) and between 0.900 and 0.909 (classification) for the six families, with 5% to 95% range of single cases 0.84 to 1.00 on halves of about 60 to 400 molecules; 67 of 1,422 regression cases and 213 of 1,740 classification cases have an under-covered task, consistent with that scatter.
- Agreement with the crepes library, smoothing off, validation halves (verify_crepes_v2.py, c0259): 3,162 comparisons. Regression normalised intervals identical (largest difference 1.1e-13, 1,422 comparisons). Classification LAC sets identical in all 1,494 comparisons where (n_cal + 1)(1 - alpha) is not an integer; in the 246 comparisons where it is an integer 145 differ, always with crepes' set equal or larger (up to 0.135 in mean set size on halves of 74 molecules). Cause: at an integer boundary crepes keeps labels with p >= alpha, one rank more than the textbook rank ceil((n + 1)(1 - alpha)) used here; ours keeps the exact 1 - alpha guarantee. Not changed.
- Pre-freeze fixes recorded in the journal: GIN test-file lookup (n0079) and ChemBERTa validation predictions on the GPU (n0081).

## 6. Outputs
results_v2/conformal_regression.csv, results_v2/conformal_classification.csv (one row per backbone, split, dataset, method, seed, alpha, score), summaries from conformal_summary.py (contrasts, tallies, Q2 coverage shares, Q3 Mondrian tables), figures (width by method at alpha 0.1 per backbone; coverage novel vs familiar), CONFORMAL_SUMMARY.md, then the extension section of the next numbered report.

## 7. Differences from PROTOCOL_v2_DRAFT.md
- conformal_v2.py: one added line, a third lookup folder for test files (expanded_v8_gin/results/<backbone>/), because GIN groups would have been skipped silently (journal n0079). check_inputs_v2.py fails loudly on any missing test counterpart.
- Operational definitions of "holds coverage", "interpretable dataset" and the overall-statement rule (section 3), implemented in conformal_summary.py and fixed here before the full run.
- ChemBERTa validation predictions recomputed on the GPU (n0081).

## 8. Code frozen with this protocol (sha256)
- conformal_v2.py: 41f61010043fe3aaa19e0b0103db1f00cff3a4c5cf5aece87b2569e966bb75aa
- check_inputs_v2.py: 93333655016591a7269a106a4aa7363fc6485da9f2be5c330d2090b13500fac0
- conformal_summary.py: afe595009006ecfe8beaac3260d5c37fa4f102e0585cc9e76994f24f7583b56c
- valid_preds.py: f097a13a016373ba1398da6697f4788289b9edd3ba9b4f367437197eb2142e46
- valid_preds_gpu.py: c18450dedee7a832dbb071c3306fbb67bd1d42402b501aab04c229946c47ec65
- valid_preds_gin.py: 0ef34e776695411110bbfea9174eb75fde3f47c88faf88ecd4557c8b0148cc92
- valid_preds_laplace.py: 124b133ff4c81820469766c17286b81878789cea672bd6ee1294bd3efc07ccf1
- PROTOCOL.md: 040ea3cd732814170f67f154088458eae24a610f3d15bfa5f03d88770d28cf9c
