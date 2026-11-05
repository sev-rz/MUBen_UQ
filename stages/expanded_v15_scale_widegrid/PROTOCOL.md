# Item B2/B1: one-parameter recalibration baseline and wide prior grid for the last-layer Laplace (protocol FROZEN before any of these computations)
Motivation. (B1) A reviewer's first question: does last-layer Laplace beat the simplest recalibration of the same network, a single parameter per task fitted on validation data? (B2) The prior-strength alpha sits at a grid edge in many fits (DNN 124/240 task fits, ChemBERTa 100, GIN 154, Uni-Mol 52/200); a wide-grid sensitivity was run only for the DNN (journal n0061).

## Scope
Same networks as item B (expanded_v14_laplace10): scaffold split, eight datasets, DNN/ChemBERTa/GIN seeds 0-9,100,200 (12 networks), Uni-Mol seeds 0-9 (10 networks); no retraining; Deterministic comparator = the network's own prediction (Uni-Mol: authors' averaging of 11 conformers); alpha and all scaling parameters chosen on VALIDATION data only; test evaluated per network after its own fit; nothing chosen on test.
Arms per network: (1) Deterministic; (2) Laplace primary (grid 1e-4..1e4, unchanged stage code); (3) Laplace wide (grid 1e-8..1e4, 13 values, Cholesky with eigen-decomposition fallback as in n0061); (4) Scaling baseline: regression: one variance multiplier s per task, s = mean over observed validation labels of (y-mu)^2 / sigma^2 (closed-form maximum-likelihood value, noise head and mean unchanged); classification: one temperature T per task from the grid logspace(0.05, 20, 81) minimising validation NLL (p = sigmoid(z/T)). The primary Laplace arm must reproduce the item-B per-network Laplace NLL (tolerance 1e-6) or the run is flagged.

## Endpoints and rules (fixed now)
d_LS = NLL(Laplace primary) - NLL(Scaling) per network (negative: Laplace better). d_WP = NLL(Laplace wide) - NLL(Laplace primary). d_SD = NLL(Scaling) - NLL(Deterministic).
Rule R10 as in item B: a dataset is 'Laplace better' if at least 90 % of the networks have d_LS < 0 and the 95 % t-interval of the mean lies below zero; 'scaling better' symmetrically; otherwise 'unclear'. A backbone 'Laplace beats scaling' needs 'Laplace better' on at least 6 of 8 datasets.
For B2: the item-B verdict (R10 of NLL(Laplace) - NLL(Deterministic)) is recomputed with the wide-grid arm; a backbone is 'robust to the grid' if the verdict is unchanged on at least 7 of 8 datasets. Also reported: edge-hit counts under the wide grid, and mean d_WP.
Exploratory (test sets were opened for baselines); all statistics descriptive: per-dataset mean, interval, share of networks; family sign-flip test over the eight dataset t-statistics per backbone.

## Hypotheses written in advance
H1: scaling recovers a large part of the Laplace gain on regression datasets; Laplace beats scaling on at least 6 of 8 datasets for at most two of the four backbones (so the cheap posterior is not clearly better than one fitted scalar). H2: wide-grid verdicts equal the primary verdicts on at least 7 of 8 datasets for every backbone (as found for the DNN). A rejected hypothesis will be reported as rejected.

## Cost
Two Slurm jobs (Uni-Mol + GIN; ChemBERTa + DNN) as in item B; expected 1.5 to 2.5 hours wall. Outputs results/<backbone>/<dataset>.json, resumable per dataset.
