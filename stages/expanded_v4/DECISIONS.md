# Tier-B decisions (PROTOCOL.md is frozen; choices and ambiguities are recorded here)

Source: MUBen at f0ec55667c1c4cc7b63369018fd011a96f31fdc4, tarball in upstream_src/ (sha256 in TARBALL.sha256),
read before coding (journal c0062). Paper: arXiv 2306.10060v4 HTML (upstream_src/paper_2306.10060v4.html, c0063).

## D1. Read from the pinned source (fidelity rule 1)
| Item | Source / paper | Tier-B implementation |
|---|---|---|
| Multi-task loss | `Trainer.get_loss`: element loss with reduction none on observed labels, `sum / masks.sum()` | `masked_loss` = same; identical to v2's mean loss when every label is observed |
| Missing labels | stored as -1 with mask 0 (tox21 csv) | label set to 0 and excluded by mask |
| Test metrics | `utils/metrics.py`: per task on observed rows, skip tasks with one class (classification), then `np.mean` (macro) | v2's verified per-task functions, then macro mean; single-class test tasks skipped |
| Selection metric | `EVAL_METRICS`: ROC-AUC (bace, bbbp, clintox, tox21), RMSE (esol, freesolv, lipo), **MAE (qm7)**; macro over tasks; ties update the checkpoint | same; regression scored on the original scale (v2/v3 used standardised RMSE: same ordering for one task) |
| Regression normalisation | `StandardScaler`: nanmean / nanstd (population) of observed training labels; SD 0 -> 1; predictions inverted (var x SD^2) | same |
| Temperature scaling | `TSModel`: one temperature **per task**; trained on validation set, lr 0.01, 20 epochs, mini-batches, BCE | per-task T, AdamW lr 0.01 wd 0.01 (v2), batch = backbone batch, 20 epochs, masked BCE |
| BBP | last layer only; loss + KL / n_steps_per_epoch / batch_len; prior sigma 0.1; 30 test samples | v3's BBP head and KL term, unchanged, now with T x L outputs and masked BCE for classification |
| SGLD | last layer only, SGLD on head + AdamW on trunk from the start; after early stopping continue from best weights; `n_langevin_samples=30`, `sgld_sampling_interval=2` -> 60 epochs, every 2nd kept | source schedule (interval 2). Paper text says 30 consecutive epochs; v3 showed both; Tier-B uses the source, which produced the reference CSVs |
| SWAG | 20 SWA epochs, k=20 checkpoints, lr x 0.5, 30 samples | v2 implementation unchanged (masked loss) |
| MC dropout | 30 passes | same |
| Ensembles | paper App. C: **M=10 for scaffold** datasets (except QM8/QM9/MUV); **M=3 per random split**. `results_get_metrics.py` pools all member predictions into one ensemble for scaffold (published SD blank); random: one ensemble per data split, SD over 3 splits | scaffold: one 10-member ensemble, seeds 0..9 (source `seed + idx`); random split k: seeds 100k..100k+2 |
| Regression ensemble/MC/SWAG/BBP/SGLD aggregation | mean of means, mean of variances (v2 FIDELITY) | same |

## D2. Seeds and reuse (fidelity rule 3)
- Single-model methods use seeds 0, 100, 200 on the scaffold split (as v2/v3; paper uses 0, 1, 2 — seed values are arbitrary
  labels, kept so that identical earlier fits can be reused) and seed 100k on random split k (as v3).
- Reused without retraining (identical configuration, verified on v4 inputs by recomputing the validation score):
  v2 scaffold Deterministic/Focal seeds and ensemble members 0..4 on bace/bbbp/esol; v3 random-split Deterministic/Focal and
  members 100k..100k+2 on bace/bbbp/esol; v3 scaffold BBP/SGLD on esol (SGLD: the even-epoch snapshots = source interval 2).
- Scaffold ensembles are topped up with new members 5..9 for bace/bbbp/esol; random ensembles use the first 3 of v3's 5 members.
- Temperature and SWAG are refit in v4 (cheap, deterministic; Temperature is now per task, identical to v2 for one task).

## D4. Descriptor failures (DNN only)
- Source keeps descriptastorus output regardless of its success flag and maps NaN to 0. Where RDKit 2024.09.6 cannot parse a SMILES at
  all (hypervalent Al in Tox21), the row is kept with all-zero descriptors, i.e. the same NaN -> 0 rule. Affected rows are listed in
  `features/<dataset>.json`: Tox21 train/61, 78, 585, 4325, 4502, 4546, 6071 and test/426 (8 of 7831); no other dataset.
  Splits and labels are unchanged. v4 descriptors for bace/bbbp/esol are bit-identical to v2's cache (c0066).

## D5. Compute
- ChemBERTa uses the GPU (protocol rule 4), FP32, TF32 off, `use_deterministic_algorithms(warn_only=True)` ("where available").
  Batch 128 is processed whole (v3 used micro-batches of 8 on CPU; same objective up to summation order).
- Several ChemBERTa worker processes may share the one GPU; units are claimed with exclusive files (`claims/`).

## D3. Architecture
- DNN: v2 trunk (8 x 128 GELU, dropout 0.1, xavier init, bias 0.01). Output head T x 2 (mean, softplus variance) for
  regression, T logits for classification. For T = 1 the head is the v2 head (same parameter shapes and RNG order).

## D6. ChemBERTa compute plan and operational wrapper (2026-10-08, journal n0030, n0032, n0033)
- The first ChemBERTa pilot (job 222938) ran on CPU because work/.venv has torch+cpu; it was discarded (archive_cpu_pilot/). All ChemBERTa units
  use work/.venv-gpu (torch 2.6.0+cu124, other packages identical, requirements-gpu.lock.txt). Timing: Tox21 2.25 s/epoch on one RTX 5000 Ada.
- Two worker processes per GPU (3-6 processes add only ~10-30 % throughput). GPU projection in PILOT_TIMING.json['ChemBERTa']: typical 9.4 GPU-h
  (0.39 GPU-day), worst 16.6 GPU-h; the registered gate (4 GPU-days) is passed, so the full ChemBERTa grid (scaffold then random) is run.
- Operational wrapper `cb_robust_worker.py` (retry a unit after CUDA OOM; 20 x 90 s; unit restarts from its seed) and `slurm/cb_worker2.sh`
  (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, all four splits). `tierb_chemberta.py` is NOT edited, so its code digest, recorded in every
  summary.json and FROZEN marker and checked by `tierb_eval`, is the same for every unit. OOM retries change no numbers: the seed is reset at unit start.
- The 23 units that failed with OOM at ~17:30 UTC (journal n0033) were re-queued (`requeue_oom.sh`): failure records archived to
  `failures_archive/oom_20261008T1750Z/`, claims released. Non-OOM failures would be kept and block freezing (cb_finish*.sh).

## D7. Final reporting rules (2026-10-09, journal n0045, n0046, n0048)
- Agreement = |reproduced mean - published mean| <= k x published SD over rows with published SD > 0 (as expanded_v3); scaffold Ensembles rows have no published SD and are not counted.
- TIERB_SUMMARY.md is generated by `tierb_compare.py` from results/*/*/*/per_seed_metrics.json and the CROSSDEVICE_BACE.json; it adds an agreement-by-metric table (median z) and the deviations/limits list.
- Reproduction gaps that were not explained (regression NLL/CE, ChemBERTa scaffold ROC-AUC/RMSE, DNN QM7 scaffold) are reported as limits; no tuning or rerun was done to close them (protocol: frozen settings).
