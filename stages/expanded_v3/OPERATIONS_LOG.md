# Phase 3 operational log

Implementation and execution fixes only. PROTOCOL.md is frozen and unchanged; no entry here
changes a scientific setting or was chosen from test results.

## 2026-10-04 ChemBERTa per-seed race (journal n0003)

- Job 217603 (`slurm/chemberta_all.sh`) started three per-seed processes
  (`run_chemberta_seed.py 0|100|200`). All three called `train_chemberta.decision()` at
  once. Each wrote `CHEMBERTA_DECISION.json.tmp`, so two `os.replace` calls failed with
  FileNotFoundError: seeds 0 and 100 crashed before training (`slurm/cb-fit-0.log`,
  `cb-fit-100.log`). Seed 200 trained normally.
- Because each per-seed process overrides `SEEDS`, the manifest written by the winning
  process lists `"seeds": [0]`. Its `scope` field ("three full fine-tunes on BACE") and
  its basis (pilot runtime/memory only, `test_results_available: false`) are correct. The
  manifest is left unedited for provenance; the pilot-derived scope is three seeds
  0/100/200.
- The same race left the manifest file malformed: a complete JSON object followed by a
  stray `\n}\n` from the second writer. Code that does not tolerate trailing data cannot
  parse it. Training and evaluation never read it (`decision()` only checks that it
  exists). The valid prefix parses with `json.JSONDecoder().raw_decode`, and its
  code_sha256 (8cff8c1e…) equals the current `train_chemberta.py`.
- After seed 200 finishes, `chemberta_all.sh` exits with "seed 0 missing" and never
  prints CHEMBERTA_DONE. Job 217603 was not cancelled or modified.
- Fix: `run_chemberta_seed_repair.py` reruns seeds 0 and 100 (one thread each) with the
  unchanged `train_chemberta.py` (same code_sha256). It skips the per-seed
  CHEMBERTA_FROZEN.json write, which had the same race risk. Then the unchanged
  all-seed `--stage fit` freezes, and `--stage evaluate` runs. The repair job's log
  prints CHEMBERTA_DONE.
- Observed: seed 200 early-stopped at epoch 75 (best validation AUC 0.7297). At 16:26 UTC,
  217603 wrote a per-seed CHEMBERTA_FROZEN.json (seeds [200]) and then printed
  "seed 0 missing" to slurm/chemberta.out, as predicted. The repair job (c0029, Slurm
  217630, node ws-l5-031) started at 16:23. For ~3 min it overlapped seed 200, with no
  shared writes. Its all-seed freeze waits for that end line and then overwrites
  CHEMBERTA_FROZEN.json with seeds [0,100,200]. Repair epochs take ~28 s (vs 16 s for
  seed 200 on ws-l5-032): a node-load difference, not a change in settings.
- Effect: seeds 0 and 100 train later than seed 200 on a different allocation. Each seed
  is seeded independently, so the results do not depend on concurrency. Wall-clock
  timings come from different nodes and loads.

## 2026-10-04 ChemBERTa repair outcome and analysis dtype fix

- Repair job c0029/217630:
  - Seeds 0 and 100 early-stopped at epochs 70 and 90 (best validation AUC 0.7212 and
    0.7073, best epochs 29 and 49).
  - The all-seed freeze wrote CHEMBERTA_FROZEN.json with seeds [0,100,200],
    test_unopened true and code 8cff8c1e….
  - Evaluate wrote 9 rows, and CHEMBERTA_DONE was printed.
- The job then failed in my `phase3_analysis.py chemberta` exact-recompute check
  (max abs diff 1.1e-8). The cause is a dtype mismatch: `train_chemberta.evaluate`
  computes metrics on `p.astype(float)` but saves `p` as float32. Recomputing from the
  saved float32 changes torchmetrics ECE/MCE by up to 4e-8, while the float64 cast
  reproduces every stored metric exactly (0.0).
- Fix: the analysis script now casts to float64 before recomputing. No ChemBERTa result
  or setting changed.

## 2026-10-04 Active-learning implementation choices (pre-outcome; journal n0004)

Fixed in `active_learning.py` CONFIG before any full run. The protocol leaves these
details open:
- Training: the unchanged v2 `run_expansion.train`. Only the data source (the labeled
  pool subset plus the official validation set) and the output root are redirected.
  Train seed = 10000 + 100*seed + round, identical across strategies, so the round-0
  model is shared within a seed.
- Diversity reference set = every labeled molecule (labeled pool molecules plus the 151
  validation molecules) plus the growing batch. Distance = 1 - max Tanimoto.
- Balanced policy = sum of average ranks of predicted activity, entropy and distance
  among the remaining candidates. The distance rank is recomputed after every pick.
  No weights.
- AUDC = mean cumulative acquired positives over rounds 1..6, at batch level.
- Test molecules are read only after round 6, to score the final model of each run.
- A smoke test with 1 round and seeds 0-1 was written to a temp folder (not results) to
  check code paths and timing (about 1 s per model fit). Nothing was changed after it.
- Every per-seed process writes its own `seed-S/config.json`. No shared files, which
  avoids the race above.

- 2026-10-04T20:05:59.157415+00:00: CHEMBERTA_DECISION.json repaired (valid prefix kept, seeds set to [0,100,200] matching CHEMBERTA_FROZEN.json; corrupted original preserved as CHEMBERTA_DECISION.corrupted.json). No result affected.
