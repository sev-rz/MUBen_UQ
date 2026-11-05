# Does the Laplace gain depend on the learning-rate schedule? ChemBERTa, linear schedule: protocol (FROZEN 2026-10-12, before any linear-schedule ChemBERTa fit of this study)
Motivation: n0116 showed that a linear schedule with 10 % warm-up lowers ChemBERTa Deterministic NLL on ESOL (1.033 -> 0.758) and QM7 (5.307 -> 5.101) relative to the constant schedule of Tier B. Question: does last-layer Laplace (stage 2) still improve NLL when the Deterministic network is trained with the linear schedule, and by how much?

## Scope
ChemBERTa-77M-MLM (pinned checkpoint), scaffold split, eight datasets, seeds 0, 100, 200 (the Tier-B single-model seeds), Deterministic networks only. Everything is as Tier B (tierb_chemberta.py: lr 1e-5, batch 128, up to 200 epochs, tolerance 40, selection on the validation score, Xavier head, FP32) except one change: a linear learning-rate schedule with warm-up of ceil(0.1 x total steps), total = epochs x steps per epoch, decaying to 0 (HuggingFace linear), stepped after every optimiser step. Code: tierb_chemberta_linear.py (the Tier-B file with only the scheduler lines added).
Laplace: the unchanged stage-2 code and protocol (expanded_v6_laplace/chemberta/PROTOCOL.md, sha256 f133fd02...), including the frozen alpha grid 1e-4..1e4 (not widened), primary arm and novelty arm, FROZEN markers per dataset before any test feature is extracted. Test sets are first opened by the Deterministic test metrics and the Laplace evaluate stage after the markers.
Not done: random splits, other backbones, conformal prediction (possible follow-up).

## Questions, endpoints and rules (fixed now)
Q1 (descriptive): Deterministic test NLL, ECE or CE under the linear schedule versus the Tier-B constant schedule, per dataset (same seeds), no test.
Q2 (primary): paired NLL difference Laplace minus Deterministic per dataset under the linear schedule; a dataset 'improves' if all three seeds agree in sign (negative); the gain is called robust to the schedule if at least 6 of 8 datasets improve (the rule of stages 1 to 3).
Q3 (size): ratio of the mean paired gain under the linear schedule to that under the constant schedule (stage 2, same seeds), per dataset and as a ratio of the means over the eight datasets. 'Shrinks' if the ratio of means is below 0.5; 'similar' if between 0.5 and 1.5; 'grows' above 1.5. Descriptive; also report whether Laplace on the constant schedule reaches the Deterministic NLL of the linear schedule (does Laplace close the schedule gap).
Hypothesis written in advance: the gain is smaller under the linear schedule on the regression datasets (the linear-schedule networks are already better calibrated) but still improves on at least 6 of 8 datasets.
Statistics: bootstrap intervals and exact sign-flip test as check E (stats_E.py design), reported as exploratory.

## Cost
Measured Tier-B fit times (RTX 5000 Ada): ESOL 58 s, FreeSolv 22 s, Lipo 279 s, QM7 142 s, BACE 64 s, BBBP 110 s per fit; 24 fits on two GPUs, expected under one hour in total, plus about 30 minutes for the Laplace stage.
