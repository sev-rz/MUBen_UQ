# MUBen reproduction, phase 3 (expanded_v3): results summary

Protocol: `../PROTOCOL.md` (frozen, sha256 5e6875b7…, unchanged). Implementation and execution
fixes are in `../OPERATIONS_LOG.md`. v1 (`work/`) and v2 (`work/expanded_v2`) were not
modified. All numbers come from journal cells: c0028 (comparisons + active learning),
c0030/c0031 (digest + figure), c0032 (AL audit), c0029 (ChemBERTa repair, freeze and evaluation)
and c0034 (ChemBERTa comparison).

| Task | Deliverable | State |
|---|---|---|
| 1 Random splits vs published random tables | `random_split_comparison.csv` (+ `random_split_summary.csv`, `random_split_per_split_metrics.csv`, `random_split_audit.json`) | done |
| 2 BBP/SGLD on ESOL vs published ESOL rows | `bayesian_comparison.csv` (+ `bayesian_summary.csv`) | done |
| 3 BACE fixed-budget discovery (main extension) | `active_learning/` (summary.csv, paired_comparisons.csv, per_seed_endpoints.csv, curves_per_seed.csv, query_ids.csv, discovery_curves.png, per-seed folders with query files, models, predictions) | done |
| 4 ChemBERTa vs DNN vs authors (BACE) | `chemberta_comparison.csv` (+ `../chemberta_results/`) | done |

Notation: "paper" = the authors' published rows at the pinned source commit f0ec556…
(`references/`). Ours are mean ± SD (ddof=1).

## 1. Random splits (task 1)

Setup: 3 random splits regenerated from the supplied CSVs with the source algorithm
(sklearn `train_test_split`, random_state 0/1/2, no stratification, original
valid/test sizes). The v2 paper DNN (8×128 GELU) with 5 members per split. Each split
has one evaluation per method, so our SD is the SD across splits.

- **Agreement.** 66/88 method×metric comparisons are within 1 published SD and 80/88
  within 2 SD. On v2 scaffold splits the counts were 47/72 and 64/72 for rows with a
  published SD.
- **Discrimination matches.**
  - BACE ROC-AUC: ours 0.880–0.885, paper 0.885–0.891 (differences −0.002 to −0.008).
  - BBBP ROC-AUC: ours 0.919–0.922, paper 0.918–0.923.
- **ESOL.**
  - RMSE: ours 0.592–0.619, paper 0.646–0.660. Lower by 0.034–0.054 (0.55–1.04 published SD).
  - NLL: ours −0.086 to 0.011, paper 0.079–0.121. Lower by 0.10–0.17, about 1.1–1.8 published SD.
  - CE (source formula) is 0.004–0.007 higher.
- **Outside 2 SD (8 rows), all calibration-tail statistics.**
  - MCE: BACE Deterministic/Temperature and BBBP SWAG are 0.17–0.19 lower. MCE is a
    single-bin maximum, so it is unstable.
  - BBBP ECE (Deterministic, MC dropout, SWAG) is 0.002–0.019 lower, against published
    SDs of only 0.001–0.007.
  - BACE Focal-loss Brier: +0.015.
  - ESOL Ensemble CE: +0.007.
- **Split effect is reproduced (same code; unpaired, different test molecules).**
  Deterministic random vs scaffold:
  - BACE AUC 0.883 vs 0.812
  - BBBP AUC 0.921 vs 0.686
  - ESOL RMSE 0.619 vs 0.913
- **Split audit.**
  - BACE: no canonical-SMILES duplicates across train/test.
  - BBBP: 5–9 molecules in both train and test; ESOL: 1–4. These are dataset
    duplicates that random splitting places on both sides.
  - Nonchiral Murcko scaffolds shared by train and test: 58–66 (BACE), 72–84 (BBBP),
    28–29 (ESOL).
- **Extension controls on random splits (paired within split, 3 splits; descriptive).**
  Test NLL differences:
  - Laplace − Deterministic: < 0 in 9/9 dataset×split pairs (−0.0003 to −0.055).
  - Novelty-Laplace − Laplace: ≤ 0 in 9/9 (0.000 to −0.092). The largest gains are on
    ESOL; the two exact ties are splits where validation selected λ=0.

**Supports:** within published variability, the reproduced DNN pipeline matches the
authors' random-split performance and most calibration metrics. The large scaffold-vs-random
gap is a property of the split, not of our implementation.

**Does not support:** claims about exact row-level split identity. LMDB ordering cannot be
recovered from the CSVs, so these are regenerated splits, not the authors' splits.

## 2. BBP and SGLD on ESOL (task 2)

Setup: official scaffold split, seeds 0/100/200, Bayesian heads trained from scratch with
the source BBP layer and SGLD optimizer. Source checks pass (`BAYESIAN_CHECKS.json`):
BBP KL matches torch, SGLD drift matches the algebra, SGLD noise variance is 2.017e-4 vs
an expected 2e-4.

| Method | RMSE | MAE | NLL | CE (source formula) | CE (paired PIT) |
|---|---|---|---|---|---|
| paper BBP | 0.956±0.030 | 0.707±0.024 | 0.431±0.032 | 0.032±0.004 | – |
| ours BBP | 0.973±0.027 | 0.732±0.011 | 0.437±0.016 | 0.031±0.003 | 0.001±0.000 |
| paper SGLD | 0.963±0.042 | 0.718±0.030 | 0.510±0.038 | 0.016±0.002 | – |
| ours SGLD, paper timing (30 ep, every epoch) | 0.947±0.011 | 0.712±0.008 | 0.451±0.036 | 0.022±0.006 | 0.005±0.004 |
| ours SGLD, source timing (60 ep, every 2nd) | 0.933±0.012 | 0.700±0.006 | 0.445±0.019 | 0.022±0.003 | 0.004±0.002 |

- **BBP is reproduced within 1 published SD on all four metrics.**
- **SGLD.**
  - RMSE and MAE are reproduced.
  - NLL is lower than published (−0.06, about 1.6–1.7 SD).
  - Source-formula CE is higher (+0.006, about 3 published SDs).
  - The paper-vs-source sampling-interval discrepancy moves SGLD by ≤ 0.014 RMSE and
    ≤ 0.006 NLL, which is within the seed SD. On ESOL it does not matter materially.
- **Definitions.**
  - The published NLL omits 0.5·log 2π. Ours with the constant (`nll_full`) is 1.36–1.37,
    and the comparison uses the matching definition.
  - The published CE is the source's all-pairs broadcast formula. A correctly paired PIT
    calibration error is 5–34× smaller, so the published CE magnitudes are mostly a
    property of the formula.
- 12 definition-matched comparisons: 7 within 1 SD, 10 within 2 SD.

**Supports:** BBP/SGLD on the DNN-rdkit backbone for ESOL are reproducible on CPU.
**Untested:** other regression datasets and backbones.

## 3. Main extension: BACE discovery under a fixed labeling budget (task 3)

**Design** (`../PROTOCOL.md`; pre-registered in journal n0004 before the full run;
details in `active_learning/config.json`):
- Labels: pool = 1210 official training molecules; 151 validation labels are a shared
  initial cost; 100 label-blind random initial labels; 6 rounds × 25 acquisitions. That
  makes 401 labels per run.
- Model: the paper DNN, retrained from scratch each round via the unchanged v2 `train()`.
  The seed schedule is shared across strategies.
- Label access: pool labels are released only by an oracle that reads each saved
  25-ID query file.
- Strategies: random, greedy (predicted activity), entropy, diversity (farthest-point,
  Morgan r2/2048) and balanced (equal ranks of activity, entropy and distance; distance
  rank recomputed as the batch grows).
- Seeds 0–4. Test molecules were read only to score each final model.

**Audit** (c0032, `../al_audit.py`):
- All 150 saved batches recompute exactly from the previous round's saved pool
  predictions and fingerprints, without labels.
- Initial sets and round-0 models are identical across strategies.
- Batches are disjoint and pool-only, with no validation or test IDs.
- Every run uses 401 labels.

| Strategy | Actives found (of 150) | AUDC | Distinct active scaffolds | New active scaffolds | Final test AUC | Final test NLL |
|---|---|---|---|---|---|---|
| random | 55.8 ± 1.3 [54–57] | 32.8 | 35.2 ± 5.1 | 20.8 ± 6.8 | 0.764 ± 0.023 | 1.46 ± 0.39 |
| greedy | **111.8 ± 10.4** [96–123] | **66.8** | 32.4 ± 5.5 | 18.6 ± 3.6 | 0.753 ± 0.030 | **0.67 ± 0.02** |
| entropy | 65.0 ± 6.3 [58–74] | 38.9 | 36.6 ± 4.5 | 24.0 ± 4.4 | 0.780 ± 0.036 | 1.42 ± 0.49 |
| diversity | 36.0 ± 2.9 [32–39] | 21.8 | 29.6 ± 2.6 | 25.2 ± 2.9 | 0.759 ± 0.029 | 1.11 ± 0.22 |
| balanced | 60.4 ± 5.9 [54–68] | 34.5 | **42.2 ± 5.8** | **33.2 ± 3.7** | 0.777 ± 0.039 | 1.22 ± 0.41 |

Random-sampling expectation from pool prevalence: 59.3–60.4 actives. Acyclic actives were
≤ 1 per run and are counted separately (`acyclic_actives`).

**Paired by seed** (5 seeds; exact two-sided sign-flip test, whose minimum is p = 0.0625;
rule pre-registered in n0004):

| Contrast | Actives (per seed) | New active scaffolds (per seed) | Verdict |
|---|---|---|---|
| greedy − random | +56.0 [66 59 61 54 40] | −2.2 [−9 −5 4 5 −6] | more actives 5/5; no scaffold difference |
| entropy − random | +9.2 [17 5 9 4 11] | +3.2 [0 −6 11 2 9] | more actives 5/5; no scaffold difference |
| diversity − random | −19.8 [−18 −21 −19 −22 −19] | +4.4 [1 −1 14 4 4] | **fewer actives 5/5** (negative); scaffold tendency 4/5 |
| balanced − random | +4.6 [11 0 5 0 7] | +12.4 [11 6 22 13 10] | actives: no detectable difference (3 higher, 2 tied); more new scaffolds 5/5 |
| balanced − greedy | −51.4 [−55 −59 −56 −54 −33] | +14.6 [20 11 18 8 16] | fewer actives 5/5; more new scaffolds 5/5 |

Also, balanced − greedy distinct active scaffolds: +9.8, 5/5. Greedy final test AUC is
lower than random in 5/5 seeds, but only by 0.010 on average. The other test-AUC
differences are inconsistent.

**Interpretation, under the pre-stated rule:**
- Greedy exploitation doubles hit yield over random (2.0×) but finds no more new
  active scaffolds than random. Its 112 hits fall in fewer distinct scaffolds than random's
  56 (32.4 vs 35.2).
- Pure diversity is a negative result: it loses about 20 actives relative to random, and
  its new-scaffold gain over random is only a 4/5-seed tendency.
- The simple equal-rank balanced policy shows the expected trade-off. It finds as many
  actives as random (not detectably more) and the highest mean number of new active scaffolds: +12 vs random and +15 vs greedy, higher in every seed. It is not better
  overall: greedy finds about 51 more actives.
- Whether "new scaffolds" or "total actives" matters more is a value judgement outside
  this simulation.
- Final-model held-out AUC hardly depends on the strategy (0.75–0.78). Greedy's much lower
  test NLL is a secondary, descriptive observation; its cause is untested (possibly the
  positive-rich labeled set).

**Limits:**
- Retrospective, on one dataset with one model.
- Labels are BACE assay endpoints, not clinical benefit.
- Only 5 seeds, so no claim at the 0.05 level is possible.
- The pool is the scaffold split's training set, so validation and test scaffolds are
  never acquirable.
- The balanced policy is a simple hybrid. It is not presented as a novel algorithm.

Runtime (measured): 7–13 s wall per seed×strategy run (6–7 model fits), 1 CPU thread.

## 4. ChemBERTa (task 4)

**Setup.**
- Public `DeepChem/ChemBERTa-77M-MLM` at a pinned revision (`CHEMBERTA_SOURCE.json`).
- Full fine-tuning: all 3.43 M parameters trainable, FP32, CPU with one thread per seed.
- Microbatch 8 with sample-weighted accumulation to an effective batch of 128; AdamW lr
  1e-5, wd 0.01; max 200 epochs, stopping when stale > 40; checkpoint selected on
  validation ROC-AUC.
- The representation is the source's `pooler_output`. The pooler is newly initialised
  for this MLM checkpoint, as in the source. Note the paper's wording says CLS.
- Scope (three full fine-tunes) was decided from pilot runtime and memory only, before
  any test result.
- No token truncation: max 129 train tokens and 184 test tokens, against a limit of 512.

**Training** (c0029; seed 200 from job 217603):

| Seed | Epochs | Best epoch | Validation AUC |
|---|---|---|---|
| 0 | 70 | 29 | 0.721 |
| 100 | 90 | 49 | 0.707 |
| 200 | 75 | 34 | 0.730 |

Measured training time: 1200–2350 s per seed on differently loaded nodes. Temperature
was fit on validation (≈1.19).

**Test results** (c0034, `chemberta_comparison.csv`; 3 seeds; BACE scaffold test set,
152 molecules):

| Method | ROC-AUC paper / ours | PRC-AUC paper / ours | NLL paper / ours | Brier paper / ours | ECE paper / ours |
|---|---|---|---|---|---|
| Deterministic | 0.722±0.009 / 0.733±0.040 | 0.703±0.001 / 0.717±0.035 | 0.757±0.020 / 0.759±0.028 | 0.272±0.008 / 0.270±0.013 | 0.241±0.013 / 0.243±0.017 |
| Temperature | 0.722±0.009 / 0.733±0.040 | 0.703±0.001 / 0.717±0.035 | 0.694±0.012 / 0.719±0.023 | 0.250±0.005 / 0.258±0.012 | 0.176±0.006 / 0.207±0.015 |
| MC dropout | 0.761±0.022 / 0.755±0.042 | 0.738±0.026 / 0.732±0.049 | 0.746±0.026 / 0.735±0.029 | 0.267±0.007 / 0.265±0.011 | 0.239±0.026 / 0.236±0.037 |

- **Agreement.** 9/18 ChemBERTa comparisons are within 1 published SD and 13/18 within
  2 SD. The 5 outside 2 SD are:
  - PRC-AUC +0.014 (twice). The published SD of 0.001 is implausibly small for 3
    fine-tunes; ours across seeds is 0.035.
  - Deterministic MCE +0.11.
  - Temperature-scaled ECE +0.031 and NLL +0.025: temperature scaling helps a little
    less for us than in the paper.
  - For the matched v2 DNN rows: 15/18 within 1 SD and 16/18 within 2 SD.
- **Variation across seeds** is larger than published: test AUC 0.745 / 0.765 / 0.688.
  The seed SD is 0.04, so the standard error of a 3-seed mean is about 0.023.
- **Backbone contrast (ChemBERTa − DNN, our v2 DNN, same test molecules, unpaired seeds).**
  - ROC-AUC: paper −0.096, ours −0.079 (Deterministic); paper −0.056, ours −0.055
    (MC dropout).
  - PRC-AUC: paper −0.123, ours −0.102.
  - NLL: ChemBERTa is lower in both (paper −0.138, ours −0.078 for Deterministic).
  - Brier: the sign differs. Paper −0.009, ours +0.016. This is because our DNN's Brier
    (0.254) is lower than the authors' DNN (0.281); the ChemBERTa Brier values agree.

**Supports:**
- The authors' ChemBERTa BACE results reproduce within roughly published variability
  for discrimination, NLL, Brier and ECE.
- The paper's qualitative finding reproduces: on BACE, the pretrained ChemBERTa-77M
  backbone discriminates worse than the RDKit-descriptor DNN (by about 0.06–0.08 AUC),
  while MC dropout narrows the gap.

**Scope:** one pretrained backbone, one dataset, 3 seeds (protocol: partial evidence). BBP,
SGLD, SWAG, ensembles and focal loss were not run for ChemBERTa.

## Deviations from the protocol and operational notes

1. **ChemBERTa per-seed race** (OPERATIONS_LOG; journal n0003). Seeds 0 and 100 crashed at
   start in job 217603. They were rerun with unchanged code in c0029/217630, on a different
   node. `CHEMBERTA_DECISION.json` is malformed (trailing `}` from the race) and lists
   seeds [0]. Its valid prefix shows the pilot-only basis; the scope is three seeds.
2. **Random splits are regenerated**, not the authors' exact splits.
   - Ensembles have 5 members (user-requested). The protocol documents the paper's
     random-split and scaffold-split member counts as three and ten.
   - Our random-split SD is across 3 splits, with one 5-member group per split.
3. **AL details the protocol left open** were fixed before outcomes (OPERATIONS_LOG):
   the diversity reference includes validation molecules, the AUDC definition, and
   rank averaging for ties.
4. **AL figure.** It was redrawn (c0030/c0031) with a validated colour-blind-safe palette and
   direct labels. The data are unchanged.
5. A failed `mktemp` for MPLCONFIGDIR in the c0028 and c0029 jobs only affected the
   matplotlib cache location.
6. **ChemBERTa completion marker.** The objective says to wait for `slurm/chemberta.out` to
   print CHEMBERTA_DONE. Because of the race, 217603 printed "seed 0 missing" instead. The
   unchanged all-seed freeze and evaluate ran in the repair job, whose log
   (`jobs/c0029-*/slurm-217630.log`) prints CHEMBERTA_DONE. CHEMBERTA_FROZEN.json lists
   seeds [0,100,200] with test_unopened = true.
7. **ChemBERTa analysis dtype fix** (c0034). The first comparison attempt in c0029 stopped
   on a 1e-8 recompute mismatch. Cause: metrics were computed in float64 but predictions
   were saved as float32. The analysis now recomputes in float64 and reproduces every
   stored metric exactly. No result changed.

## What each finding supports

- **Supported:** the DNN-rdkit results on random splits (BACE/BBBP/ESOL) are reproduced
  within published variability, as are BBP on ESOL and SGLD accuracy on ESOL.
- **Supported:** the scaffold-vs-random split gap is reproduced qualitatively.
- **Supported:** ChemBERTa-77M on BACE (Deterministic, Temperature and MC dropout)
  reproduces within roughly published variability on most metrics. The paper's
  ChemBERTa < DNN-rdkit discrimination gap on BACE is reproduced: AUC −0.079 vs −0.096
  (Deterministic) and −0.055 vs −0.056 (MC dropout).
- **Supported (new, exploratory):** on BACE under a 401-label budget there is a trade-off.
  Greedy acquisition doubles hits; an equal-rank activity/uncertainty/diversity policy
  finds the most new active scaffolds while matching random hit yield; pure diversity
  costs hits.
- **Partly supported:** SGLD calibration on ESOL. Our NLL is better and our source-formula
  CE is slightly worse than published.
- **Contradicted within scope:** none of the authors' qualitative claims tested here. Some
  calibration-tail metrics differ beyond the published SD, in both directions:
  - DNN random splits: our MCE/ECE are lower.
  - ChemBERTa: our MCE and temperature-scaled ECE/NLL are higher.
  - Published ChemBERTa PRC-AUC SDs (0.001) are smaller than our seed spread (0.035).
  These are high-variance statistics over 3 seeds or splits.
- **Untested:**
  - ChemBERTa with other UQ methods (BBP, SGLD, SWAG, ensembles, focal loss) or on other
    datasets; other backbones and datasets for BBP/SGLD.
  - Prospective active learning; other AL budgets, batch sizes, models and datasets.
