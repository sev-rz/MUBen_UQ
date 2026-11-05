# Item B: last-layer Laplace on ten or more networks per dataset: protocol (FROZEN 2026-10-12, before any of these computations)
Motivation: the Laplace results of stages 1 to 4 rest on three repeats per dataset (the registered rule: all three agree in sign). Every backbone has more trained Deterministic networks on the scaffold split: the ten ensemble members (seeds 0 to 9) and, for DNN, ChemBERTa and GIN, the single-model seeds 100 and 200.

## Scope
Backbones DNN, ChemBERTa, GIN (repeats: seeds 0 to 9, 100, 200 = 12 networks per dataset) and Uni-Mol (seeds 0 to 9 = 10 networks); scaffold split; eight datasets; primary Laplace arm only (no novelty arms); the unchanged stage code of each backbone (fit_group = analytic last-layer Laplace, alpha per task by validation NLL on the frozen grid 1e-4..1e4, not widened; part_outputs; compose with no variant), run per network; no retraining. The Deterministic comparator is the same network's own prediction (Uni-Mol: authors' averaging of the 11 conformers, as in stage 4). Alpha is chosen on validation data only; the test split is evaluated per network after its own fit, and nothing is chosen on test. Seeds already used in the earlier stages are recomputed with the same code; they must reproduce the earlier per-seed Laplace NLL (tolerance 1e-6) or the run is flagged.

## Endpoint and rules (fixed now)
Primary endpoint per backbone and dataset: paired test NLL difference d = NLL(Laplace) - NLL(Deterministic) per network.
Rule R10 (new, because 'all repeats agree' is too strict with 10 or 12 repeats): a dataset 'improves' if at least 90 % of the repeats have d < 0 and the 95 % t-interval of the mean of d lies below zero; 'worsens' symmetrically; otherwise 'unclear'. An overall statement for a backbone needs 'improves' on at least 6 of 8 datasets (the rule of stages 1 to 4). The original three-repeat rule (all of the original repeats negative: seeds 0/100/200, Uni-Mol 0/1/2) is reported next to it, and the two verdicts are compared per dataset.
Family test (exploratory): per backbone an exact one-sided sign-flip test over the eight dataset t-statistics of the mean difference; Holm correction over the four backbones.
Secondary (descriptive): mean gain per backbone with the spread over repeats, the fraction of repeats improving per dataset, and the same endpoint on ECE (classification) or CE (regression).

## Hypotheses written in advance
H1: R10 gives at least 6 of 8 'improves' for each of the four backbones. H2: the three-repeat verdict agrees with the R10 verdict on at least 7 of 8 datasets per backbone (the three-repeat results were not an artefact of too few repeats). Datasets whose gain is near zero (ClinTox, Tox21, QM7 for some backbones) may change verdict; that would be reported as such.

## Cost and efficiency
Features are cheap (Uni-Mol fit_s 5 to 11 s per network in stage 4). Two GPU jobs (Uni-Mol with GIN; ChemBERTa with DNN), expected under two hours in total. Outputs: results/<backbone>/<dataset>.json, resumable per dataset.
