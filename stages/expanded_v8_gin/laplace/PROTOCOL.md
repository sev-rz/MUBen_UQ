# Laplace on the Tier-B GIN backbone (stage 3): protocol (FROZEN 2026-10-10, before any GIN posterior was fitted)
Student's instruction (2026-10-10): try Laplace on the new backbone. Same design as stage 1 (DNN) and stage 2 (ChemBERTa) so the three backbones are comparable; the frozen alpha grid is not widened. Applies once the GIN reproduction (expanded_v8_gin/PROTOCOL.md) has trained and frozen its nets for a split/dataset.
EXTENSION: no published reference exists. Tier-B and GIN reproduction results are not modified.

## 1. Question
Does last-layer Laplace on GIN change NLL and calibration relative to the Deterministic GIN, at lower cost than the ensembles, on the eight datasets? Does the stage-1/2 pattern (consistent gain on scaffold, not on random) hold for a non-pretrained graph model?

## 2. Scope and method (as stage 2, section 2-4)
- Networks: the saved GIN Deterministic nets (expanded_v8_gin/models/GIN/<split>/<dataset>/det/seed-*/best.pt), single-model seeds only (scaffold 0, 100, 200; random-k seed 100k); no retraining. Eight datasets, scaffold and random-0..2: 96 posteriors.
- Last layer = the output Linear head on the sum-pooled GIN representation (128 inputs; 129 parameters per task and output). Backbone frozen as a feature extractor; features computed on the GPU, FP32.
- Classification: independent affine Bernoulli posterior per task, probit predictive. Regression: conditional Laplace over the mean head with frozen features and frozen predicted noise. Analytic Hessians checked against autograd at every start. alpha grid 1e-4..1e4 per task by validation NLL; edge hits reported, not extended.
- Secondary arm, as before: Laplace constant, Novelty only, Novelty Laplace; lambda grid [0, .01, .1, 1, 10, 100] per task by validation NLL after alpha; novelty distance to the split's training molecules (Morgan r2, 2048 bits, no chirality), novel = similarity < 0.4, subgroups with at least 20 molecules; conclusions on novelty from scaffold only; the constant-inflation control is part of every comparison.
- Comparators from the GIN results (same seeds and molecules): Deterministic, Temperature, MC dropout, SWAG, Ensembles.

## 3. Evaluation and decision rules (as stages 1-2, fixed now)
tierb_eval metrics; macro over tasks; primary endpoint paired NLL Laplace minus Deterministic per dataset and split type; a direction claim needs all three repeats to agree in sign; an overall claim needs at least 6 of 8 datasets with the same sign on that split type; no dataset dropped; negative results reported. Descriptive comparison with the DNN and ChemBERTa tables (same rule, not a test).

## 4. Test isolation
alpha and lambda from validation data only; FROZEN markers (primary and novelty) carrying this protocol's sha256 are written per split/dataset before any test feature is extracted for Laplace. The GIN test sets are opened by the GIN reproduction's own evaluation, so results are exploratory in that sense; nothing is chosen on test.

## 5. Context to carry into the report
GIN is randomly initialised (no pre-training), so its uncertainty behaviour differs from ChemBERTa's; the GIN encoder and featuriser are the authors' code inside our pipeline (bridge check registered in the GIN protocol, section 5). Compute: one GPU Slurm job, resumable; outputs under expanded_v8_gin/laplace/.
