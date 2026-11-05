# Protocol: Uni-Mol (3D) on the eight core datasets, option B -- FROZEN before any Uni-Mol training result was read
Written 2026-10-11 (student's approval: "proceed with B"). Plan: work/expanded_v7_backbones/PLAN.md (sections 7, 10). Authors' paper: arXiv 2306.10060v4 (Appendix B, Table 5); code commit f0ec556.

## Scope
Backbone Uni-Mol-base (47.59 M parameters, checkpoint mol_pre_no_h_220816.pt sha256 da27196a...4585e9, n0106), scaffold split only (the authors' files; the Uni-Mol LMDB row order is identical to them, check D n0104). Datasets: bace, bbbp, clintox, tox21, esol, freesolv, lipo, qm7.
Out of scope for this protocol: random splits, SWAG, BBP, SGLD, focal loss (published targets exist for them; not run). GROVER is not part of this protocol.

## Model and training (all fits identical in settings)
Authors' muben.model.UniMol encoder and pooler with the pre-trained checkpoint, the authors' DatasetUniMol / CollatorUniMol / ConfigUniMol imported unchanged (11 conformers per molecule: one sampled per training step; at inference all 11 and the transformed outputs averaged, as muben.train.trainer_unimol). Our pipeline: linear output head (T x L outputs, xavier init), masked loss (tierb_dnn.masked_loss), our loop and scoring (tierb_dnn.valid_score; model selection on the validation score, tolerance 40), batch 128, AdamW (lr 5e-5, betas 0.9/0.99, eps 1e-6, weight decay 0.01), 100 epochs maximum, linear schedule with 10 % warm-up, gradient-norm clipping 1 (authors' default), dropouts 0.1 (authors' fixed values), FP32, deterministic algorithms where available.
Regression targets standardised on the training labels, as in the other Tier-B backbones.

## Units and seeds
Per dataset: deterministic fits for seeds 0-9 (10 fits). Seeds 0, 1, 2 are the three Deterministic runs (authors' seeds); seeds 0-9 together are the pooled deep ensemble (as published for scaffold: M = 10, no SD). Post-hoc on seeds 0-2: per-task temperature on validation (classification) and MC dropout (30 draws at test time). Tier-B DNN, ChemBERTa and GIN used Deterministic seeds 0/100/200; this changes the seed set for this backbone only and is stated in the report.

## Test access
Test predictions are computed only by the evaluate stage after the FROZEN marker of a dataset (all 13 units done), with the code and protocol hashes recorded in the marker.

## Agreement with the published tables
Same rule as the other backbones: a row agrees at k if |reproduced mean - published mean| <= k x published SD, counted only over rows with published SD > 0 (scaffold ensemble rows have no SD); methods available here: Deterministic, MC dropout, Temperature (classification), Ensembles. Published rows: reports/primary/Uni-Mol-<dataset>.csv. Not comparable with the DNN/ChemBERTa totals (fewer methods, one split type): reported separately.

## Bridge check against the authors' own trainer (registered now)
Authors' unmodified run/unimol.py with the paper settings (lr 5e-5, batch 128, 100 epochs, tolerance 40, linear scheduler, warm-up 0.1, uncertainty method none, regression with variance), seeds 0-9 on ESOL and BACE, seeds 0-4 on QM7, on the scaffold files, compared with our Deterministic fits of the same seeds: difference of means in units of the pooled seed SD with a Welch 95 % interval; consistent if |d| < 1; different if |d| >= 1 and the interval excludes 0; metrics RMSE, MAE, NLL, CE (regression) and ROC-AUC, PRC-AUC, ECE, NLL, Brier (classification).
Gate: if our Deterministic differs from the authors' trainer on RMSE or NLL (regression) or ROC-AUC or NLL (classification) with |d| >= 2 on more than one of the three datasets, the Uni-Mol results are reported as a pipeline-limited reproduction and Laplace and conformal on this backbone are labelled accordingly; nothing is dropped silently.

## Pre-registered extension questions (as for the other backbones)
Laplace (last layer, analytic Hessian, alpha grid 1e-4..1e4 on validation NLL, as the other backbones; no wider grid): primary endpoint the paired NLL difference (Laplace minus Deterministic) per dataset on scaffold, a direction needs all three seeds to agree; an overall claim needs the same sign on at least 6 of 8 datasets. Conformal (protocol_v2: split conformal alpha 0.1, LAC / normalised score, calibration on validation, Mondrian by novelty): same contrasts and rules as the other backbones. Statistics (check E) and shift-aware calibration (check F) applied as for the other backbones.
Cross-backbone claims (a) and (b) of n0070 are re-evaluated with Uni-Mol added as a fourth backbone, restricted to the methods available here (the published gap for claim (b) recomputed on the same restricted method set).

## Cost gates
Measured step time replaces the paper's. Pilot (smoke, 2 epochs ESOL): about 0.16 s per training step, 1.6 s per epoch including validation (RTX 5000 Ada). If the full B grid is projected to exceed 24 GPU-hours the largest datasets (tox21, qm7) are dropped first and this is reported.

## Not done / limits (stated in advance)
Head initialisation and loss path are ours, not the authors' output layer; one linear head. Scaffold only, three Deterministic seeds, no random-split or SWAG/BBP/SGLD/focal rows; all results exploratory (as for the other extensions: test sets of other backbones were opened earlier, but this backbone's test set is unopened until the freeze).
