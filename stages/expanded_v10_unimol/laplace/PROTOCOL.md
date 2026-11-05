# Laplace on the Tier-B Uni-Mol backbone (stage 4): protocol (FROZEN 2026-10-11, before any Uni-Mol posterior was fitted)
Same design as stage 1 (DNN), stage 2 (ChemBERTa) and stage 3 (GIN) so the four backbones are comparable; the frozen alpha grid is not widened. Applies once the Uni-Mol reproduction (expanded_v10_unimol/PROTOCOL.md, sha256 8ab19980...) has trained and frozen its nets for a dataset.
EXTENSION: no published reference exists. Uni-Mol reproduction results are not modified.

## 1. Question
Does last-layer Laplace on Uni-Mol change NLL and calibration relative to the Deterministic Uni-Mol, at lower cost than the ensembles, on the eight datasets (scaffold only)? Does the stage 1-3 pattern (consistent gain on scaffold) hold for a 3D pre-trained model?

## 2. Scope and method (as stage 3, section 2)
- Networks: the saved Uni-Mol Deterministic nets (expanded_v10_unimol/models/Uni-Mol/scaffold/<dataset>/det/seed-{0,1,2}/best.pt); no retraining. Eight datasets, scaffold only: 24 posteriors.
- Last layer = our output Linear head on the pooled Uni-Mol representation (512 inputs; 513 parameters per task and output). Backbone frozen as a feature extractor; features on the GPU, FP32.
- DEVIATION, stated in advance: the Deterministic prediction averages the transformed outputs (sigmoid, mean, softplus variance) over the 11 conformers of a molecule (the authors' rule). The Laplace stage needs one feature vector per molecule and therefore uses the conformer-AVERAGED pooled representation; the head is linear, so the mean of the raw outputs is exactly reproduced, but the averaging of probabilities (classification) and of softplus variances (regression) is replaced by the output of the averaged features. The Deterministic comparator is the authors' aggregation. The feature check (maximum difference to the saved net's conformer-mean raw outputs) must be below 1e-2.
- Classification: independent affine Bernoulli posterior per task, probit predictive. Regression: conditional Laplace over the mean head with frozen features and frozen predicted noise. Analytic Hessians checked against autograd at every start. alpha grid 1e-4..1e4 per task by validation NLL; edge hits reported, not extended.
- Secondary arm, as before: Laplace constant, Novelty only, Novelty Laplace; lambda grid [0, .01, .1, 1, 10, 100] per task by validation NLL after alpha; novelty distance to the training molecules (Morgan r2, 2048 bits), novel = similarity < 0.4, subgroups with at least 20 molecules; the constant-inflation control is part of every comparison.
- Comparators from the Uni-Mol results (same seeds and molecules): Deterministic, Temperature, MC dropout, Ensembles.

## 3. Evaluation and decision rules (as stages 1-3, fixed now)
tierb_eval metrics; macro over tasks; primary endpoint paired NLL Laplace minus Deterministic per dataset on scaffold; a direction claim needs all three seeds to agree in sign; an overall claim needs at least 6 of 8 datasets with the same sign; no dataset dropped; negative results reported. Statistics as check E (bootstrap interval per dataset, exact sign-flip test, Holm over the family of all backbone/split tests).

## 4. Test isolation
alpha and lambda from validation data only; FROZEN markers (primary and novelty) carrying this protocol's sha256 are written per dataset before any test feature is extracted for Laplace. Uni-Mol test sets are opened by the Uni-Mol reproduction's evaluation stage only after its own freeze.
