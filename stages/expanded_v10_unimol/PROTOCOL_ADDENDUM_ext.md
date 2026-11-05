# Addendum to the Uni-Mol protocol: scaffold-only SWAG, BBP, SGLD and focal loss -- FROZEN 2026-10-12 before any of these units was trained
Student's instruction (2026-10-12): after the current jobs, run the scaffold-only methods for Uni-Mol. Base protocol: PROTOCOL.md (sha256 8ab19980bd7d211addecd004a13981ed8285ac692d9f46d703e309fc9adbb3e1), unchanged. Random splits stay out of scope.

## Added units (per dataset, seeds 0, 1, 2 = the three Deterministic seeds)
- SWAG: from the Deterministic net of the same seed, 20 extra epochs at lr/2 (peak lr 2.5e-5, constant), 20 snapshots, rank-20 SWAG (the other backbones' recipe, tierb_dnn.posthoc); 30 posterior draws at test time.
- BBP: Bayes-by-backprop output head (tierb BBPOutputLayer, prior sigma 0.1) on the pre-trained encoder, KL term weighted 1/(steps x batch) as in the other backbones; 30 draws at test time. Trained like a Deterministic fit (same schedule, selection, tolerance 40, 100 epochs maximum).
- SGLD: encoder trained with AdamW, output head sampled with SGLD (norm_sigma 0.1), constant learning rate 5e-5 (the authors' code forces a constant rate for SGLD); training with validation selection as a Deterministic fit, then 60 further epochs from the best weights keeping every second epoch (30 snapshots, as the other backbones); predictions pool the 30 snapshots.
- Focal loss (classification datasets only): focal loss (alpha 0.25, gamma 2) instead of cross-entropy, otherwise a Deterministic fit.
Training details shared with the base protocol: batch 128 realised by gradient accumulation over micro-batches of 32 (n0113), clipping at 1, FP32. In BBP each micro-batch draws its own weight sample (the KL term is the same for the whole batch); this is a stated deviation from one draw per batch of 128.

## Evaluation and rules
Test predictions of these methods are produced only by the extended evaluate stage after a per-dataset FROZEN marker (all extension units done) written by the extension module with this addendum's hash. The extended evaluate recomputes the base methods with the unchanged base code (a copy of the earlier results is kept to confirm identical numbers) and adds SWAG, BBP, SGLD and Focal Loss rows. Agreement with the published rows uses the same rule as the base protocol, now over all published methods except those that are random-split only; claims C3 (BBP and SGLD ROC-AUC below Deterministic) become testable. Laplace and conformal results already produced are unchanged; conformal is rerun on the full method set afterwards.

## Cost gate
Projected 12 to 14 GPU-hours (units are 3 seeds x 4 or 3 methods, about 11 fit-equivalents per classification and 8 per regression dataset against 10 Deterministic fits measured at 14.4 GPU-h in total). Wall-clock rule as in n0109: if all units are not finished 30 h after the start, Tox21 and QM7 SGLD and SWAG units are dropped first and this is reported.
