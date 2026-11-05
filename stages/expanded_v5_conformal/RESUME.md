# Conformal extension — resume notes
- Code: valid_preds.py (validation predictions from saved models, test untouched), conformal.py (frozen analysis; 'selftest' uses validation halves only).
- Self-test passed on DNN scaffold bace/esol (selftest/), coverage ~0.88-0.96 at alpha 0.1 on ~60-molecule halves.
- Full run: job 223577 (slurm/conformal_all.sh), depends afterok on Tier-B finish job 223323. If 223323 fails or is replaced, resubmit:
  sbatch -p gpu --gres=gpu:1 -c 6 --mem=40G -J conformal -o slurm/conformal_%j.out slurm/conformal_all.sh
  (it skips finished validation-prediction folders via DONE markers).
- Not yet covered: Laplace / Novelty Laplace (expanded_v2) need validation predictions from v2 code — next session.
- 2026-10-08T21:52:30Z: conformal.py verified against crepes 0.8.0 (verify_crepes.py, smoothing=False): 76/76 self-test comparisons identical (coverage, widths, set sizes). With crepes' default smoothed p-values, differences are randomized tie-breaking only.
- 2026-10-08T21:54:00Z: Laplace family added (v2_laplace_preds.py, phase-2 code + saved posteriors, no refit): preds/DNN-v2 (validation) and
  preds_test/DNN-v2 (test; asserted identical to expanded_v2/results to 1e-9). conformal.py reads preds_test/ first. Included in job 223577 automatically.

## 2026-10-10 ~15:25Z (lab agent turn, Slurm 224483): state and plan for v2
- GIN reproduction finished (expanded_v8_gin: 336 fits, 32 groups frozen + evaluated). GIN Laplace done (job 224891 step 1). Reports 05 (reproduction, 3 backbones) and 06 (extensions) published.
- Inputs still being generated (validation predictions, no test labels): job 224891 (ChemBERTa scaffold -> random-0..2, then GIN, GIN-Laplace) and helper job 224907 (GIN, GIN-Laplace, ChemBERTa random-2 -> random-1 -> random-0). Folders carry DONE markers and are skipped when finished; expected ~2 h total (ChemBERTa groups take 5-20 min each).
- Bug fixed before the freeze (journal n0079): conformal_v2.py now also looks for test files in expanded_v8_gin/results/<backbone>/ (GIN would have been skipped silently).
- New files, not yet frozen: check_inputs_v2.py (fails loudly unless 6 backbones x 32 groups are present and every validation file has a test counterpart), conformal_summary.py (P1-P3 contrasts, interpretable-dataset rule, Q2/Q3 tables; code-tested on selftest_v2 only).
NEXT, in this order (all before any conformal computation on test):
 1. python check_inputs_v2.py  (exit 0 required; scope = DNN, ChemBERTa, GIN and the three -Laplace folders; DNN-v2 is a cross-check only)
 2. python conformal_v2.py selftest  (validation halves only; coverage ~0.90 at alpha 0.1 expected), then verify_crepes.py on a sample
 3. Write PROTOCOL_v2.md from PROTOCOL_v2_DRAFT.md + the amendments above (path fix, input check, summary rules, 'holds coverage' = more than half of the tasks 'held', interpretable dataset = both methods hold in every repeat), record the sha256 of conformal_v2.py, check_inputs_v2.py and conformal_summary.py inside it, write PROTOCOL_v2.sha256.
 4. python conformal_v2.py run  (results_v2/), then python conformal_summary.py results_v2, figures, CONFORMAL_SUMMARY.md, report.

## 2026-10-10 ~15:40Z: v2 frozen and run (lab agent turn, Slurm 224483)
- PROTOCOL_v2.md frozen (sha256 5cbbf963...), journal n0082. All 6 families x 32 groups prepared (ChemBERTa validation predictions recomputed on the GPU, job 224909; the CPU-computed scaffold/bace folder is in archive_cpu_valid_preds/).
- Full run done: results_v2/conformal_regression.csv (2,844 rows), conformal_classification.csv (3,480 rows); summaries: summary_contrasts_*.csv, summary_tally_*.csv, summary_q2_coverage_*.csv, summary_q3_mondrian_*.csv (conformal_summary.py, frozen) and summary_overall_8datasets.csv (conformal_overall.py, add-on: the frozen script applies the 6-of-8 rule per kind, see journal n0083).
- First look: journal n0084 (no consistent gain of conformal-on-Laplace over Deterministic or Ensembles; Mondrian improves but does not close the novel-subgroup gap on scaffold; GIN classification coverage on scaffold ~0.75).
- TO DO: read the Q2 and regression Q3 tables in full, Q5 (Deterministic coverage/width across backbones on scaffold), figures (width by method at alpha 0.1 per backbone; coverage novel vs familiar), CONFORMAL_SUMMARY.md, then a numbered report (extensions: conformal). Do not edit frozen files (digests in PROTOCOL_v2.md section 8).

## 2026-10-10 (lab agent turn, Slurm 224584): conformal read-out done, report 07 published
- Q2, Q3, Q5 read from results_v2/summary_*.csv; presentation script conformal_figures.py (not frozen) wrote CONFORMAL_SUMMARY.md, results_v2/figures/*.png (ratio_reg, ratio_cls, coverage_novel_vs_familiar), results_v2/report_*.csv. Journal: n0087-n0092; report reports/07-conformal-prediction-on-the-reproduced-m.
- Open, needs the student (nothing running): (1) GIN bridge check, ESOL RMSE 2.07 seed SD (n0075/n0067); (2) GIN SGLD instability, seed 100 (n0087/n0088) - diagnose or leave; (3) shift-aware calibration for classification under scaffold shift (new registered step); (4) wider alpha grid for GIN/ChemBERTa Laplace (n0078).
