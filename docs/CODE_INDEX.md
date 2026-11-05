# Code index (stage folder, script, first line of its docstring)

## expanded_v2
- `analyze_expansion.py`: Recompute every result from saved predictions, compare references and plot.
- `build_report.py`: Build a concise six-page report from saved measurements, never invented scores.
- `evaluate_saved.py`: Evaluate frozen fits with a shape-aware loader for the upstream SWAG state.
- `run_expansion.py`: CPU MUBen expansion. Run --stage fit, then --stage evaluate.
- `verify_archive.py`: Verify the portable package's versioned files from any working directory.

## expanded_v3
- `active_learning.py`: BACE discovery under a fixed labeling budget (PROTOCOL.md, 'Main extension').
- `al_audit.py`: Audit of the saved active-learning runs (no training; nothing is rewritten).
- `bayesian_regression.py`: Pinned MUBen BBP and SGLD heads on ESOL; paper/source sampling ablation.
- `build_final_report.py`: Build the final technical report: overview + phase-2 report (unchanged) + phase-3 chapter.
- `build_two_reports.prev.py`: Build two reports from saved results (no retyped numbers). Run from work/:
- `build_two_reports.py`: Build two reports from saved results (no retyped numbers). Run from work/:
- `chemberta.py`: Public pinned ChemBERTa checkpoint, CPU pilot and resumable fine-tuning.
- `phase3_analysis.py`: Phase-3 comparison tables from saved predictions (no training, no test-driven choices).
- `phase3_digest.py`: Descriptive digest behind PHASE3_SUMMARY.md, computed only from the result CSVs/JSONs.
- `plot_discovery.py`: Redraw results/active_learning/discovery_curves.png from curves_per_seed.csv.
- `random_splits.py`: Source-algorithm random splits; reuse frozen v2 implementation unchanged.
- `run_chemberta_seed.py`: Run one ChemBERTa seed of the unchanged train_chemberta.fit (parallel per-seed jobs).
- `run_chemberta_seed_repair.py`: Re-run one crashed ChemBERTa seed with the unchanged train_chemberta.fit (see OPERATIONS_LOG.md).
- `train_chemberta.py`: Full, resumable CPU fine-tuning selected solely by the training-only pilot.

## expanded_v4
- `cb_robust_worker.py`: Operational wrapper around tierb_chemberta.worker (scientific code untouched, tierb_chemberta.py digest unchanged).
- `crossdevice_bace.py`: Protocol rule 4 cross-device check: BACE scaffold ChemBERTa, v4 GPU (batch 128) vs expanded_v3 CPU (micro-batches of 8), test side.
- `diagnostics_nll/diag_nll.py`: Descriptive NLL/CE diagnostic on SAVED Tier-B test predictions. No training, no tuning, frozen files untouched.
- `pilot_projection.py`: ChemBERTa GPU projection from CHEMBERTA_PILOT_UNITS.json; adds it to PILOT_TIMING.json (DNN part already there).
- `pilot_projection_gpu.py`: ChemBERTa GPU projection from the production units finished so far (the CPU-run pilot 222938 was invalid, journal n0029);
- `tierb_chemberta.py`: Tier-B ChemBERTa-77M-MLM on one GPU (FP32, deterministic algorithms): same units, seeds, selection,
- `tierb_compare.py`: Comparison tables against the authors' CSVs (references/, commit f0ec556) and agreement counts.
- `tierb_dnn.py`: Tier-B DNN-rdkit: masked multi-task training and all MUBen UQ methods on scaffold and random splits.
- `tierb_eval.py`: Test evaluation for Tier-B, only behind a FROZEN marker. Per-task metrics with v2's verified

## expanded_v5_conformal
- `check_inputs_v2.py`: Input coverage check for conformal_v2.py (run before the full run; reads file names only, no prediction values).
- `conformal.py`: Conformal analysis exactly as frozen in PROTOCOL.md (expanded_v5_conformal). Saved predictions only.
- `conformal_figures.py`: Presentation tables, two figures and CONFORMAL_SUMMARY.md from the frozen run (PROTOCOL_v2.md section 6). Reads results_v2/ only; nothing is recomputed from predictions.
- `conformal_overall.py`: Overall statement of PROTOCOL_v2.md section 3 across the 8 datasets (4 regression + 4 classification), per backbone, split type and contrast.
- `conformal_summary.py`: Summaries of the conformal results under PROTOCOL_v2.md section 3 (rules fixed before the full run; reads the CSVs written by conformal_v2.py only).
- `conformal_unimol.py`: Uni-Mol copy of conformal_v2.py (same analysis; families Uni-Mol and Uni-Mol-Laplace only; Uni-Mol test paths added; outputs results_unimol / selftest_unimol).
- `conformal_v2.py`: Conformal analysis (v2 of conformal.py: identical analysis as frozen in PROTOCOL.md; only change = unparsable-SMILES guard in novelty()). Saved predictions only.
- `freeze_v2.py`: Freeze PROTOCOL_v2.md from PROTOCOL_v2_TEMPLATE.md. Refuses unless: no full-run output exists yet (results_v2/), check_inputs_v2.py exits 0,
- `v2_laplace_preds.py`: Validation (and test, for format) predictions of the phase-2 Laplace family with the unchanged phase-2 code and saved
- `valid_preds.py`: Validation-split predictions for conformal calibration (no retraining, test split never touched).
- `valid_preds_gin.py`: Validation-split predictions of the GIN reproduction for conformal calibration (no retraining; test split never touched). Mirrors valid_preds.py.
- `valid_preds_gpu.py`: GPU-correct launcher for valid_preds.py ChemBERTa (valid_preds.py itself is unchanged).
- `valid_preds_laplace.py`: Validation predictions (from the saved posteriors, test never touched) and staged TEST copies for the Laplace family of one backbone, for conformal calibration.
- `valid_preds_laplace_unimol.py`: Validation predictions (from the saved posteriors, test never touched) and staged TEST copies for the Uni-Mol Laplace family, for conformal calibration. Mirrors valid_pre
- `valid_preds_unimol.py`: Validation-split predictions of the Uni-Mol reproduction for conformal calibration (no retraining; test split never touched). Mirrors valid_preds_gin.py.
- `verify_crepes.py`: Cross-check conformal.py against the crepes library on the self-test split (validation halves; test split untouched).
- `verify_crepes_v2.py`: v2: cross-check conformal_v2.py (DONE folders only) against the crepes library on the self-test split (validation halves; test split untouched).

## expanded_v6_laplace
- `analyze.py`: Registered analysis of Laplace on Tier B (PROTOCOL.md section 5). Reads per_seed_metrics.json of expanded_v6_laplace and expanded_v4 (DNN). No tuning.
- `chemberta/analyze_chemberta.py`: Registered analysis of Laplace on Tier B (PROTOCOL.md section 5). Reads per_seed_metrics.json of expanded_v6_laplace and expanded_v4 (DNN). No tuning.
- `chemberta/chemberta_laplace.py`: Laplace on the Tier-B ChemBERTa nets (protocol: chemberta/PROTOCOL.md, sha256 in PROTOCOL.sha256). Stage-1 maths (tierb_laplace.py) reused unchanged;
- `sens_wide.py`: SENSITIVITY: Laplace on Tier B with a wider alpha grid 1e-8..1e4 (registered in journal n0061). Same code path as tierb_laplace.py with ALPHAS replaced;
- `tierb_laplace.py`: Laplace on the Tier-B DNN (protocol: PROTOCOL.md, sha256 in PROTOCOL.sha256). Reuses the SAVED Tier-B Deterministic nets (tierb_dnn.fit reloads them);

## expanded_v7_backbones

## expanded_v8_gin
- `bridge_metrics.py`: Metrics of the authors' GIN runs (their own functions) for the registered bridge check; reads only the authors' prediction files.
- `claims_crossbackbone.py`: Cross-backbone paper claims, operational definitions registered in journal n0070 (before any GIN test result was looked at).
- `claims_crossbackbone4.py`: Cross-backbone claims with Uni-Mol as a fourth backbone (definitions registered in the journal note 'Cross-backbone claims with Uni-Mol as a fourth backbone', before this
- `gin_compare.py`: GIN reproduction vs the authors' published GIN rows (reports/primary/GIN-*.csv, commit f0ec556): agreement counts, claims check, bridge check. Reuses expanded_v4/tierb_co
- `laplace/analyze_gin.py`: Registered analysis of Laplace on the GIN backbone (expanded_v8_gin/laplace/PROTOCOL.md section 3), adapted unchanged from expanded_v6_laplace/chemberta/analyze_chemberta
- `laplace/gin_laplace.py`: Laplace on the Tier-B GIN nets (protocol: laplace/PROTOCOL.md, sha256 in PROTOCOL.sha256). Stage-1 maths (tierb_laplace.py) reused unchanged;
- `tierb_gin.py`: Tier-B GIN (2D graph): our Tier-B pipeline (structure of tierb_chemberta.py) with the authors' GIN encoder (muben.model.gin: atom embedding + torch_geometric GIN, jk=cat)

## expanded_v9_checks
- `bridge_analysis.py`: Checks A and B of the registered fix plan (n0095). A: authors' GIN trainer (seeds 0-9; tox21 0-4) vs OUR saved GIN Deterministic nets (scaffold ensemble members, seeds 0-
- `check_D.py`: Check D (n0095): is our random split the authors' random split? Authors' routine (assist/dataset_build_from_unimol.py): LMDB rows concatenated train,valid,test (Uni-Mol p
- `conformal_F.py`: Check F (n0096): shift-aware (test-internal) conformal calibration. Test molecules are grouped by Bemis-Murcko scaffold; scaffold groups are assigned at random 50/50 to a
- `conformal_F_unimol.py`: Check F (n0096): shift-aware (test-internal) conformal calibration. Test molecules are grouped by Bemis-Murcko scaffold; scaffold groups are assigned at random 50/50 to a
- `descriptor_drift.py`: Check C (n0095): recompute the 200 normalised RDKit descriptors in the authors' environment (RDKit 2023.09.5, descriptastorus 2.6.1)
- `stats_E.py`: Check E (n0096): bootstrap intervals and a family-level sign-flip test for the paired NLL difference Laplace minus Deterministic, on the SAVED test predictions.
- `stats_E_debug.py`: 
- `stats_E_unimol.py`: Check E (n0096): bootstrap intervals and a family-level sign-flip test for the paired NLL difference Laplace minus Deterministic, on the SAVED test predictions.

## expanded_v10_unimol
- `bridge_unimol.py`: Bridge check (PROTOCOL.md): authors' unmodified Uni-Mol trainer (authors_out) vs our Deterministic Uni-Mol on the SAME seeds, test metrics. ESOL and BACE seeds 0-9, QM7 s
- `laplace/control_no_posterior.py`: Control for the conformer-averaging deviation (laplace/PROTOCOL.md section 2): posterior-free prediction from the conformer-averaged features (Laplace with zero covarianc
- `laplace/unimol_laplace.py`: Laplace on the Tier-B Uni-Mol nets (protocol: laplace/PROTOCOL.md, sha256 in PROTOCOL.sha256). Stage-1 maths (tierb_laplace.py) reused unchanged;
- `smoke_cpu.py`: 
- `tierb_unimol.py`: Tier-B Uni-Mol (3D): our Tier-B pipeline with the authors' Uni-Mol model (muben.model.UniMol, pre-trained checkpoint), dataset and collator imported (11 conformers per mo
- `tierb_unimol_ext.py`: Uni-Mol extension units (PROTOCOL_ADDENDUM_ext.md): SWAG, BBP, SGLD, focal loss on the scaffold split, seeds 0-2, in our Tier-B pipeline around the authors' Uni-Mol encod
- `unimol_compare.py`: Uni-Mol reproduction vs the authors' published Uni-Mol rows (reports/primary/Uni-Mol-*.csv, commit f0ec556): agreement counts and claims check, restricted to the methods 

## expanded_v11_chemberta_sched

## expanded_v12_headinit
- `analyze_arms.py`: Compare ChemBERTa arms (ESOL, QM7; Deterministic, scaffold) and the GIN head-init arm. Authors' metrics from their own functions (load_results/regression_metrics).
- `chemberta_arm.py`: ChemBERTa test arms (n0111 follow-up). arm = linear (our pipeline + linear schedule, 10 % warm-up) | default (our pipeline + PyTorch-default head init). Deterministic, sc
- `gin_headinit.py`: Head-initialisation test (n0111 follow-up): our GIN pipeline with the output head at PyTorch default init (the authors' OutputLayer) instead of Xavier + bias 0.01. Determ
- `tierb_chemberta_default.py`: Tier-B ChemBERTa-77M-MLM on one GPU (FP32, deterministic algorithms): same units, seeds, selection,
- `tierb_chemberta_linear.py`: Tier-B ChemBERTa-77M-MLM on one GPU (FP32, deterministic algorithms): same units, seeds, selection,
- `tierb_gin_headinit.py`: Tier-B GIN (2D graph): our Tier-B pipeline (structure of tierb_chemberta.py) with the authors' GIN encoder (muben.model.gin: atom embedding + torch_geometric GIN, jk=cat)

## expanded_v13_schedule
- `analyze_sched.py`: Item 9 analysis (rules in PROTOCOL.md): Q1 Deterministic constant vs linear; Q2 paired Laplace gain under the linear schedule; Q3 gain ratio and gap closing. Exploratory 
- `det_preds_lin.py`: Save per-molecule test predictions of the linear-schedule Deterministic nets (for the check-E statistics): laplace/results/scaffold/<d>/Deterministic-<seed>.npz (mean, va
- `laplace/chemberta_laplace_lin.py`: Laplace on the Tier-B ChemBERTa nets (protocol: chemberta/PROTOCOL.md, sha256 in PROTOCOL.sha256). Stage-1 maths (tierb_laplace.py) reused unchanged;
- `merge_lin.py`: 
- `run_lin.py`: Deterministic linear-schedule ChemBERTa fits (seeds 0/100/200) and their test metrics; usage: run_lin.py DATASET...  (processes may run in parallel on different datasets)
- `run_lin_seed.py`: One seed of one dataset (linear-schedule Deterministic ChemBERTa) plus its test metrics; usage: run_lin_seed.py DATASET SEED. The three seed files are merged into metrics
- `stats_sched.py`: Item 9 statistics (check-E design): per-molecule paired NLL Laplace minus Deterministic under the linear schedule, bootstrap over test molecules (2000), exact sign-flip o
- `tierb_chemberta_linear.py`: Tier-B ChemBERTa-77M-MLM on one GPU (FP32, deterministic algorithms): same units, seeds, selection,

## expanded_v14_laplace10
- `analyze_laplace10.py`: Item B analysis (rules in PROTOCOL.md): R10 verdicts per backbone and dataset, comparison with the original three-repeat verdicts, family sign-flip test with Holm over th
- `laplace10.py`: Item B runner (PROTOCOL.md): per network last-layer Laplace (unchanged stage code) and Deterministic test metrics, scaffold split.

## expanded_v15_scale_widegrid
- `analyze_wide_scale.py`: Analysis of items B1/B2 (PROTOCOL.md): R10 verdicts for Laplace vs Deterministic (primary and wide grid) and Laplace vs one-parameter recalibration; sign-flip family test
- `wide_scale.py`: Item B1/B2 runner (PROTOCOL.md): per network Laplace primary grid, Laplace wide grid, one-parameter recalibration baseline, Deterministic. Scaffold split.

## expanded_v16_round1
- `c1_fewlabel.py`: C1: few labelled target molecules (PROTOCOL.md). Saved predictions only.  python c1_fewlabel.py [--bb DNN ChemBERTa GIN] [--datasets d ...] [--R 50]
- `c2_weighted.py`: C2: similarity-weighted conformal prediction under covariate shift (PROTOCOL.md + ADDENDUM). Saved predictions only.  python c2_weighted.py [--bb ...] [--datasets ...]
- `c6_linear_schedule.py`: C6: conformal under the linear learning-rate schedule (ChemBERTa, scaffold; PROTOCOL.md). Validation predictions come from the saved linear-schedule nets and posteriors (
- `cp_common.py`: 

## expanded_v17_gin_linear
- `gin_linear_bridge.py`: B3 runner (PROTOCOL.md): our GIN Deterministic with the linear schedule vs the authors' ten-seed bridge runs (classification datasets), same comparison rule as gin_bridge
- `tierb_gin_linear.py`: Tier-B GIN (2D graph): our Tier-B pipeline (structure of tierb_chemberta.py) with the authors' GIN encoder (muben.model.gin: atom embedding + torch_geometric GIN, jk=cat)
