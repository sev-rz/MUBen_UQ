# Tier-B reproduction protocol (frozen 2026-10-08, before any Tier-B result)

## Question
How closely do MUBen's published results reproduce for its 8 core datasets with the DNN-rdkit and ChemBERTa-77M backbones,
on scaffold (authors' splits) and random splits, across all of the paper's uncertainty methods?

## Scope (the paper's own 8-dataset core: the datasets used in its random-split and frozen experiments)
- Datasets: bace, bbbp, clintox (2 tasks), tox21 (12 tasks, missing labels) — classification; esol, freesolv, lipo, qm7 — regression.
  Inputs: data/<d>/{train,valid,test}.csv + meta.json from commit f0ec556 (checksums in INPUT_SHA256.txt).
- Backbones: DNN-rdkit (as in expanded_v2: 200 RDKit descriptors, 8x128 GELU, dropout 0.1) and ChemBERTa-77M-MLM (as in expanded_v3, pinned revision).
- Methods: classification — Deterministic, Temperature, Focal Loss, MC dropout, SWAG, BBP, SGLD, Ensembles;
  regression — Deterministic, MC dropout, SWAG, BBP, SGLD, Ensembles. Exactly the rows of references/primary/*.csv.
- Splits: scaffold = authors' files; random = regenerated with the source algorithm, random_state 0/1/2, original sizes (as expanded_v3).
- Repeats: 3 seeds (scaffold) / 3 splits (random). Ensembles: number of members as the pinned source/paper specifies for each split
  (read from source/paper first; record in DECISIONS.md). If GPU budget forces fewer ChemBERTa members, decide from the timing pilot,
  before any test evaluation, and log it as a deviation.

## Fidelity rules
1. Before coding multi-task and new regression handling, read the pinned source (muben/) for: masked multi-task loss, how per-task
   metrics are averaged, regression target normalisation, and any dataset-specific options. Implement those; log every ambiguity in DECISIONS.md.
2. Reuse verified code: expanded_v2/run_expansion.py (DNN, all methods except BBP/SGLD classification), expanded_v3 (BBP/SGLD regression,
   ChemBERTa, random splits). Extend, do not rewrite. New BBP/SGLD classification follows the source.
3. Earlier results are reused only where the configuration is identical (DNN scaffold/random on bace/bbbp/esol, BBP/SGLD esol).
   Ensembles with a different member count are topped up with new members, not retrained.
4. ChemBERTa runs on the GPU (FP32, deterministic algorithms where available). This deviates from the CPU single-thread protocol of
   expanded_v3; BACE ChemBERTa Deterministic is rerun on GPU and compared with the CPU result as a cross-device check.
5. Selection uses validation data only. A dataset's test set is evaluated only after all its fits for that backbone/split are frozen
   (FROZEN marker per backbone/split/dataset). No hyperparameter is changed after seeing test results.

## Compute plan
- DNN: CPU jobs on ws-ia (<=24 CPUs, 2 running jobs; one thread per process).
- ChemBERTa: GPU partition (1 GPU, 16 CPUs, 90 GB, no time limit).
- Timing pilot first (one ChemBERTa seed on tox21 and lipo; DNN on tox21); write PILOT_TIMING.json with a projected total before launching.
- Long work runs as resumable Slurm jobs with per-run completion markers; never rerun a completed run.

## Outputs
- results/comparison_scaffold.csv and results/comparison_random.csv (same format as work/reproduction_comparison.csv:
  backbone, dataset, method, metric, upstream mean/std, reproduced mean/std, delta), per-seed metrics, predictions.
- results/TIERB_SUMMARY.md: within-1-SD / within-2-SD counts per backbone x split x dataset, claims check, deviations.
- OPERATIONS_LOG.md for every failure, fix and rerun.
