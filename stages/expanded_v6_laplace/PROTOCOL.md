# Laplace on the Tier-B setting: protocol (FROZEN 2026-10-10, before any posterior was fitted or any test label opened)
Frozen 2026-10-10 after the student confirmed section 9 (a) novelty arm secondary, (b) ChemBERTa gated stage 2, (c) existing Tier-B nets. sha256 in PROTOCOL.sha256;
any change after this point is recorded in DECISIONS.md, never edited here. This is an EXTENSION. Tier-B reproduction results (expanded_v4) are not modified.

## 1. Question
Does last-layer Laplace (as in expanded_v2) change NLL and calibration relative to the Deterministic network, at lower cost than the 10/3-member ensembles, on the
eight Tier-B datasets, and does the effect hold beyond bace/bbbp/esol (where it was first studied)? Secondary: does a novelty-corrected variance help on scaffold splits?

## 2. Scope (student's decision: all 8 datasets)
- Backbone: DNN-rdkit. Datasets: bace, bbbp, clintox (2 tasks), tox21 (12 tasks, missing labels), esol, freesolv, lipo, qm7.
- Splits and seeds as Tier B: scaffold (seeds 0, 100, 200) and random-0..2 (seed 100k). 8 x 4 x 3 = 96 posteriors.
- Networks: the SAVED Tier-B Deterministic nets (models/DNN/<split>/<dataset>/det/seed-*/best.pt, reloaded with tierb_dnn.fit). No retraining. Centre = early-stopped estimate (as v2; stated, not hidden).
- ChemBERTa: NOT in this protocol (stage 2, gated; section 8).

## 3. Method
- Classification: affine last layer, Bernoulli likelihood, GLM probit predictive, laplace-torch 0.2.3 as in v2; multi-task datasets get one independent posterior per task output
  (129 parameters each); labels with mask 0 are excluded from the Hessian and from the alpha selection. alpha grid 1e-4..1e4 (9 values), chosen per task by validation NLL.
- Regression: conditional Laplace over the mean head (129 parameters), frozen representation and frozen predicted noise: H = Phi^T diag(1/noise) Phi, cov = (H + alpha I)^-1,
  predictive variance = noise + phi^T cov phi (original scale via the training-label scaler as Tier B). alpha grid as above, validation NLL. The analytic Hessian is checked
  against autograd on one dataset before the full run. This is not a joint mean-and-variance posterior.
- Comparators, same molecules and seeds, taken from expanded_v4/results: Deterministic, Temperature (classification), MC dropout, SWAG, Ensembles. BBP, SGLD and Focal Loss are reported for context only.

## 4. Secondary arm: novelty-corrected Laplace (scaffold splits only for conclusions)
- As v2: d(x) = 1 - max Tanimoto to the training set (Morgan radius 2, 2048 bits, no chirality); extra variance lambda * d^2, lambda grid [0, .01, .1, 1, 10, 100] chosen by validation NLL AFTER alpha;
  controls: novelty-only (no Laplace variance) and constant inflation on top of Laplace. Novel subgroup: nearest-training similarity < 0.4 (not tuned); subgroup with < 20 molecules is "not estimable".
- Reported on random splits for completeness, but no claim about novelty is drawn from them (few novel molecules). Called a project hypothesis, not a verified method.

## 5. Evaluation (tierb_eval metric code, unchanged)
Classification: roc-auc, prc-auc, ece, mce, nll, brier. Regression: rmse, mae, nll, ce (source definition) and paired PIT calibration, 90% coverage, mean width.
Macro over tasks (single-class test tasks skipped, as Tier B); mean and SD over seeds/splits.
Primary endpoint: paired NLL difference Laplace minus Deterministic per dataset/split. Secondary: against MC dropout and Ensembles; cost (fit seconds, predictions per molecule).
Decision rules (fixed now): "improves/worsens" only if all repeats of that dataset/split agree in sign; an overall statement needs agreement in at least 6 of 8 datasets on that split type.
Eight datasets x two split types are reported together; no dataset is dropped for an unfavourable result. Negative and null results are published.

## 6. Test isolation
alpha and lambda are chosen on validation data only; all grids and choices are saved and a FROZEN marker is written per backbone/split/dataset BEFORE test predictions are produced.
The Tier-B test sets were already opened for the baselines, so Laplace results on them are exploratory in the sense of v2/v5 (no tuning on test); this is stated in the report.

## 7. Fidelity and reproduction context (discussion to be carried into the report)
Where our DNN differs from the authors':
| Item | Authors | Ours (Tier B) | Effect found |
|---|---|---|---|
| Depth | 8 hidden 128-wide layers plus the input layer: 9 linear layers, 158,082 parameters (paper Table 5; authors' code) | 8 linear layers in total, 141,570 parameters (we read "8 hidden" as 8 total) | none measurable on ESOL and QM7: the authors' 9-layer network matches ours within seed noise (n0057) |
| Activation | paper: GELU; code default: ReLU | GELU | example-config run (ReLU) moved QM7 RMSE toward published (137.6) |
| Training | paper: batch 256, lr 2e-4 constant, 400 epochs, tolerance 50; code example config: lr 1e-4, batch 128, 200 epochs | paper values | training settings change the learned variance (ESOL NLL 0.41 vs 0.76) |
| Seeds | 0, 1, 2 | 0, 100, 200 (so earlier fits can be reused) | arbitrary labels |
| Code | their trainer, features and metrics | own PyTorch code; their SWAG, BBP and SGLD modules; their metric definitions copied | authors' code reproduces our numbers on QM7 and ESOL (n0057) |
| Other | ensembles 10 (scaffold) / 3 (random) as in paper; 8 Tox21 SMILES unparsable | same; zero descriptors for those 8 rows | -- |
What we found when we ran the authors' own code (n0054, n0057): on QM7 it reproduces our Tier-B numbers (NLL 5.47 vs 5.48, RMSE 126.9 vs 127.3) and not the published ones (NLL 10.30, RMSE 142.9); on ESOL both agree with the published row.
How to read this: our reimplementation is a faithful stand-in for the authors' DNN pipeline on the cases tested, so the Tier-B baselines are a sound base for Laplace.
The remaining distance between published and reproduced numbers (regression NLL/CE; ChemBERTa scaffold) is unexplained. Possible explanations, NONE tested: published tables were produced with settings
other than those the paper or the released config state (the repo contains older scripts in legacy/ and only an example config); a different variance-head training or aggregation in the original runs;
reporting differences. We cannot decide between these without the authors' predictions or settings (the repo ships only aggregated report CSVs). Consequence for this protocol:
Laplace is compared mainly with OUR reproduced baselines (same seeds, molecules, code); comparisons with published numbers are secondary and carry this caveat.
Not covered by the authors'-code check: other UQ methods, FreeSolv, Lipo, random splits, ChemBERTa, classification.

## 8. Stages and gate
Stage 1 (this protocol): DNN, 8 datasets, all splits. Order: smoke test (bace and esol, scaffold, seed 0, validation only); Hessian check; alpha/lambda selection for all groups and FROZEN markers;
test predictions and metrics; add Laplace predictions to the conformal inputs (v5 reads preds_test/); report section.
Stage 2 (ChemBERTa Laplace, own protocol later): run only if stage 1 shows a consistent NLL effect (rule in section 5); ChemBERTa scaffold is the least-reproduced part of Tier B, so an extension there is
interpreted with that caveat. Compute is small (features and a 129-parameter Hessian per task).
Compute: CPU on ws-ia, one process per dataset/split, resumable, completion markers; nothing on the login node.

## 9. Student to confirm before freezing
(a) novelty-corrected arm included as secondary (recommended: yes); (b) ChemBERTa as gated stage 2 (recommended: yes); (c) Laplace on the existing Tier-B nets (8 linear layers) rather than a rebuilt 9-layer DNN (recommended: yes, noted above).
