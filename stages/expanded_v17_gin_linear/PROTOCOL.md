# B3: is the GIN classification gap a learning-rate-schedule difference? protocol FROZEN before any of these fits
Observation (n0137, read-only code inspection). The authors' GIN argument class (muben/args/args_2d.py) sets lr_scheduler_type default 'linear' (warm-up ratio 0.1 from the base class); our Tier-B GIN loop uses a constant learning rate (as the paper's Appendix B states for GIN: constant lr 1e-4). The ten-seed bridge (gin_bridge10.csv, n0095) found classification metrics consistent on 14 of 20 comparisons (5 different, 1 unclear; ours better calibrated: BACE ECE 0.17 vs authors 0.30; ClinTox ROC-AUC 0.69 vs 0.65; Tox21 ROC-AUC 0.692 vs 0.679). The launch command of the authors' GIN runs could not be located in the project files, so the schedule they ran with is inferred, not known.

## Test
Retrain OUR GIN Deterministic with the linear schedule with 10 % warm-up (the item-9 mechanism: LambdaLR, warm = ceil(0.1 total), total = epochs x steps per epoch, stepped per optimiser step; nothing else changed: same encoder, head, batch 128, lr 1e-4, 200 epochs, tolerance 50, selection by validation ROC-AUC, same seeds) on the four classification datasets: BACE, BBBP, ClinTox seeds 0-9; Tox21 seeds 0-4 (the same seeds as the authors' bridge runs). Compare with the authors' ten-seed (Tox21 five-seed) metrics with the bridge rule of n0095 (d = difference / pooled SD, Welch 95 % interval; consistent if |d| < 1, different if |d| >= 1 and the interval excludes 0, otherwise unclear) on ROC-AUC, PRC-AUC, ECE, NLL, Brier, as in gin_bridge10.csv. The constant-schedule rows of gin_bridge10.csv are the comparison.
Endpoint: number of consistent comparisons out of 20 (5 metrics x 4 datasets) with the linear schedule versus 14 of 20 with the constant schedule.

## Hypothesis and rule written in advance
H_B3: the linear schedule makes the classification comparison consistent on at least 18 of 20 comparisons (the gap was the schedule). If it stays at 15 or fewer, the schedule is rejected as the main cause of the GIN classification gap. 16 or 17: reported as partial. No other setting is changed in this test; any further hypothesis (selection tie rule, loss path) needs its own registration.

## Cost and efficiency
35 Deterministic fits (3 x 10 + 5), each minutes on one GPU; one Slurm job; resumable per fit. Outputs: expanded_v17_gin_linear/ (models, gin_linear_bridge.csv).
