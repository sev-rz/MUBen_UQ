# Laplace on the Tier-B ChemBERTa backbone: stage-2 protocol (FROZEN 2026-10-10, before any ChemBERTa posterior was fitted or any test label used for it)
Stage 2 of expanded_v6_laplace/PROTOCOL.md (section 8). Student's decision (2026-10-10): proceed with ChemBERTa Laplace; the alpha grid is NOT widened (frozen grid is primary, as in stage 1).
This is an EXTENSION; Tier-B reproduction results (expanded_v4) are not modified. Everything not stated here follows stage 1 (PROTOCOL.md, DECISIONS.md D1-D6).

## 1. Question
Does last-layer Laplace on ChemBERTa-77M-MLM change NLL and calibration relative to the Deterministic network, at lower cost than the ensembles, on the eight Tier-B datasets? Is the DNN stage-1 pattern
(scaffold: consistent NLL gain; random: not consistent) repeated on a pretrained backbone? Secondary: novelty-corrected variants as in stage 1.

## 2. Scope
- Backbone ChemBERTa-77M-MLM (pinned checkpoint of Tier B); datasets bace, bbbp, clintox, tox21, esol, freesolv, lipo, qm7; splits scaffold (seeds 0, 100, 200) and random-0..2 (seed 100k). 96 posteriors over the head.
- Networks: the SAVED Tier-B ChemBERTa Deterministic nets (expanded_v4/models/ChemBERTa/<split>/<dataset>/det/seed-*/best.pt), reloaded with tierb_chemberta.fit. No retraining, no fine-tuning of the backbone.
- Last layer = the output Linear head on the pooler output (hidden size H read from the checkpoint config; H + 1 parameters per task and output). The backbone is frozen as a feature extractor.

## 3. Method (as stage 1)
- Classification: per-task independent affine Bernoulli posterior over the head weights of that task, probit predictive; regression: conditional Laplace over the mean head, frozen features and frozen predicted noise,
  H = Phi^T diag(m/noise) Phi, predictive variance noise + phi^T cov phi on the standardised scale, times scale^2. Analytic Hessians (checked against autograd at every start). alpha grid 1e-4..1e4 (nine values), per task, by validation NLL.
  Edge hits are REPORTED, not extended (stage-1 D4; the student asked not to widen the grid).
- Features are computed on the GPU, the device on which these nets were trained and evaluated in Tier B (the Tier-B cross-device check found a small CPU/GPU shift), in FP32 with TF32 off.
- Comparators, same molecules and seeds, from expanded_v4/results/ChemBERTa: Deterministic, Temperature (classification), MC dropout, SWAG, Ensembles; BBP, SGLD, Focal Loss for context.

## 4. Secondary arm: novelty-corrected variants (as stage 1, section 4)
Laplace constant, Novelty only, Novelty Laplace; d(x) = 1 - max Tanimoto to the split's training molecules (Morgan r2, 2048 bits, no chirality; unparsable query SMILES get d = 1); lambda grid [0, .01, .1, 1, 10, 100]
per task by validation NLL after alpha; novel subgroup d > 0.6 with at least 20 molecules; conclusions on novelty only from scaffold splits; the constant-inflation control is part of every comparison.

## 5. Evaluation and decision rules (as stage 1, section 5; fixed now)
Metric code unchanged (tierb_eval.macro); macro over tasks; primary endpoint paired NLL Laplace minus Deterministic per dataset and split type; a direction claim per dataset/split type needs all three repeats to agree in sign; an overall claim needs
at least 6 of 8 datasets with the same sign on that split type; no dataset dropped. Secondary: against MC dropout and Ensembles, other metrics, cost. Descriptive comparison with the DNN stage-1 table (same rule, not a test).

## 6. Test isolation
alpha and lambda from validation data only; a FROZEN marker per split/dataset (primary) and per split/dataset (novelty) is written, carrying this protocol's sha256, BEFORE any test feature is extracted for Laplace. Test sets were already opened for the Tier-B baselines: results are exploratory in that sense.

## 7. Known context to carry into the report
ChemBERTa scaffold is the least-reproduced part of Tier B (ROC-AUC median z about -2, RMSE/MAE about +1.7 published SD, cause not investigated, n0048); the authors'-code check (n0054, n0057) covered DNN only, not ChemBERTa. The Laplace comparison is therefore
against our own reproduced ChemBERTa baselines (same seeds, molecules and code), not against published numbers. The BACE GPU/CPU shift is about a third of a seed SD (n0037).

## 8. Compute
One GPU Slurm job (partition gpu, 1 GPU), resumable with per-group completion markers; stages fit, novelty, evaluate in order; nothing on the login node. Outputs under expanded_v6_laplace/chemberta/.
