# Tier-B operations log (failures, fixes, reruns)

- 2026-10-08 c0064 (smoke, temp output, 3 epochs): `prep` failed on Tox21 row `CCOC(=O)/C=C(/C)O[AlH3](OC(C)CC)OC(C)CC`:
  descriptastorus returned a False flag. Source `utils/chem.py` ignores the flag (keeps `process()[1:]`, NaN -> 0); v4 now does the same
  and lists flagged rows in `features/<d>.json`. Smoke results for clintox/esol were finite for every method; ESOL reuse checks passed.
- 2026-10-08 c0065: `process()` returns None for that row: RDKit 2024.09.6 rejects hypervalent Al ("Explicit valence for atom #9 Al, 6")
  that the authors' older RDKit accepted. Fix (DECISIONS D4): keep the row with all-zero descriptors (source NaN -> 0 rule applied to
  every feature); rows listed under `rdkit_unparsed_zero_features`. ChemBERTa (SMILES tokens) is unaffected. `/usr/bin/time` is absent on
  the workbench node; timing taken with `date`.
- Workbench slot: ws-ia allows 2 running jobs and the workbench + magic-shell hold both, so the DNN Slurm job waits for the workbench slot.
- 2026-10-08 c0066: DNN pilot ran real scaffold Tox21 units posthoc-0 and posthoc-100 (both complete: summary.json, swag.pt). The worker's
  progress print after posthoc-100 hit BrokenPipe (my `| head -1`), which the unit guard recorded as `failures/DNN_scaffold_tox21_posthoc_100.txt`;
  the unit itself had completed, so the spurious failure file was removed. Lesson: do not truncate worker stdout with head.
- 2026-10-08 14:20 UTC: stopped the workbench (only waiting on jobs) so DNN job 222944 (c0073, 22 one-thread workers) could take the
  ws-ia slot; ChemBERTa pilot job 222938 (c0067) runs on the gpu partition meanwhile.
- 2026-10-08 c0073 / job 222944: DNN complete in 14 min: 336 unit markers, no failures, 32 FROZEN markers, 32 evaluations
  (results/DNN/<split>/<dataset>/). c0074: comparison CSVs + TIERB_SUMMARY.md (DNN only so far).
- 2026-10-08 c0075 / job 222966: ChemBERTa scaffold job submitted while the pilot (job 222938) was still on its first Tox21 fit
  (>24 min, epochs >= 7 s). Gate registered beforehand (journal n0025) and enforced inside the job: projection -> PILOT_TIMING.json,
  launch only if typical total <= 4 GPU-days; 1-epoch smoke of fit/freeze/evaluate on ClinTox first. Pending on the gpu QoS
  CPU/GPU limit until the pilot ends.
- 2026-10-08T17:08:27Z: cancelled CPU-run ChemBERTa pilot 222938 and gate job 222966 (work/.venv had CPU-only torch, journal n0029); moved the partial unit and ChemBERTa claims to archive_cpu_pilot/. All ChemBERTa units will run with work/.venv-gpu.
- 2026-10-08T17:10:26Z (chat): cancelled 222938 (CPU-run ChemBERTa pilot: .venv had torch+cpu) and 222966 (gate job, would also run on CPU). Archived its outputs/claims to archive_cpu_pilot/. All ChemBERTa now uses work/.venv-gpu (journal n0029).
- 2026-10-08T17:12Z (chat, c0095): production ChemBERTa launched: gpu job 223273 and ws-ia job 223274 (2 workers each, `--splits scaffold random`), finish job 223275.
- 2026-10-08T17:30Z, job 223273: 23 units failed with CUDA OOM within a minute (error names a 23.6 GiB foreign process on the same GPU). Found at 17:45Z (lab agent turn
  223248). Also found: `--splits scaffold random` is invalid for the workers (no split 'random': IndexError in t.units), so they would crash after the scaffold queue
  and never run the random splits. Actions (17:50-17:52Z, c0096): requeue_oom.sh (23 failure records -> failures_archive/oom_20261008T1750Z, claims released);
  queued 223321 cb-mop-gpu (afterany 223273), 223322 cb-mop-wsia (afterany 223274), 223323 cb-finish2 (afterany both) running cb_robust_worker.py on all splits.
  Jobs 223273, 223274, 223275 left running/pending. The agent sandbox cannot reach the Slurm controller (squeue: connect failure) or nvidia-smi.
- 2026-10-08T17:52Z: PILOT_TIMING.json['ChemBERTa'] written by pilot_projection_gpu.py from the 48 finished production fits (n0032).
- 2026-10-08T19:10Z: old workers 223273/223274 ended (scaffold queue done; WORKER_DONE, they did not reach the invalid split 'random'). Chat's finish 223275 froze and evaluated ChemBERTa scaffold bace, bbbp, esol, freesolv, lipo, qm7 (clintox, tox21 had no complete units yet). Mop jobs 223321 (gpu) and 223322 (ws-ia) started ~19:10Z and completed the 23 re-queued units plus random-0 so far; 0 failures, 0 OOM_RETRY lines by 19:47Z.
- 2026-10-08 c0098: BACE scaffold cross-device check on the test side (results/CROSSDEVICE_BACE.json, journal n0037). c0099: interim tierb_compare run (works with partial ChemBERTa).
- slurm/gpumem_<job>.log lists compute processes of the whole node (8 GPUs), not only ours; it cannot identify the owner of the 23.6 GiB seen at the OOM. Not used further.
- 2026-10-09T13:58Z (lab agent turn, Slurm 223584): the sc-hub MCP server failed to connect (CONNECT_TIMEOUT) for the whole turn, so no journal read, no run() cells, no note()/handoff(). Only read-only file checks were made. State found: ChemBERTa 336/336 fit markers, 32/32 FROZEN, 32/32 evaluated (cb-finish2 223323 printed CHEMBERTA_EVAL_DONE), failures/ empty, 0 OOM_RETRY lines in the mop worker logs, PROTOCOL.sha256 OK. Still to do: final tierb_compare.py run via run(), final TIERB_SUMMARY.md, verdict note.
- 2026-10-09 (lab agent turn, Slurm 223656): sc-hub tools back. c0120/c0123 final tierb_compare.py (64 groups), c0121-c0122, c0125-c0126 descriptive breakdowns, c0124 integrity check (64/64 groups evaluated after FROZEN with matching code digest, finite metrics, failures/ empty, PROTOCOL.sha256 OK). Notes n0045 (counts), n0046 (verdict), n0047/n0048 (corrected ranges). PENDING_FOR_NEXT_TURN.md items 1-4 are done except the optional QM7 / NLL investigations.
