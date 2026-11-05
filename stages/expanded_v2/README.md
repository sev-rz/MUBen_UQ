# Expanded MUBen CPU reproduction

Persistent cluster project:
`/l/users/sevinch.zoda/schub/projects/muben-laplace/work/expanded_v2`

The Slurm job supplies computation. It does not own these Lustre files; they remain accessible after the job expires, subject to your institution's storage policy. This project does not rely on compute-node /tmp or an active Python process for its results. A portable archive and SHA256 manifest are delivered alongside the report; keep the local copy as a second copy.

## Contents

- `PROTOCOL.md`: fixed design before new test evaluation.
- `FIDELITY.md`: paper/source differences and reproduced versus extended components.
- `data/`: pinned raw train/validation/test CSVs and dataset metadata.
- `features/`: cached200-dimensional descriptors, molecular IDs and feature audit.
- `models/`: best checkpoints, optimizer/RNG resume states, histories and summaries.
- `fits/`: Laplace posterior matrices, SWAG state and validation-selected parameters.
- `FROZEN.json`: settings saved before held-out evaluation, code/protocol hashes.
- `results/`: individual predictions, all metrics, original comparisons, checks and figures.
- `source/`: pinned upstream modules needed for the new methods and numerical checks.
- `requirements-lock.txt`, `INPUT_SOURCE_MANIFEST.json`, final `MANIFEST.json`: provenance.
- `fit.log`, `evaluation.log`, `analysis.log`: execution evidence, including failed attempts.
- `technical_report.pdf` and its editable builder: concise report with original result comparisons.

## Run on an allocated compute node

Never run model training on the HPC login node. In an interactive CPU allocation:

```bash
cd /l/users/sevinch.zoda/schub/projects/muben-laplace/work
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
.venv/bin/python expanded_v2/run_expansion.py --stage fit
.venv/bin/python expanded_v2/evaluate_saved.py
.venv/bin/python expanded_v2/analyze_expansion.py
```

The fit command deliberately refuses to replace an existing FROZEN.json. For a new independent reproduction, extract the archive into a fresh project and keep the supplied results as reference. Copy the code, protocol, raw data, reference tables and source modules into a fresh `expanded_v2` directory without models/fits/features/results/FROZEN.json; then run the commands. Interrupted fitting resumes from checkpoints when source hashes agree. Re-evaluation uses saved fitted parameters and must never select new settings from test labels.

For this SC Hub project, launch the same commands through a journalled BenchService cell in the current allocation, as used in the preserved parent `bench_existing_job.py`. After job expiry, use a new valid allocation ID; old217442 will no longer be usable. No new job or session token is necessary to read existing files.

## Portable archive layout

Extract into a chosen directory. Parent `train_baseline.py` and `evaluate_uncertainty.py` supply previously verified feature/Hessian helper code; `expanded_v2/run_expansion.py` contains the new experiment. The bundled source license accompanies upstream code. Create a Python3.12 environment on Linux, install the recorded CPU PyTorch build from https://download.pytorch.org/whl/cpu and the dependency lock, then run from an allocated CPU. Exact package wheels may require platform-specific installation steps on macOS/Windows.

Verify every manifest-listed file with `python expanded_v2/verify_archive.py` from the extracted parent directory. The archive itself has a separate SHA256 sidecar. The manifest cannot hash itself recursively and therefore excludes itself.

## Interpretation

Three datasets and a descriptor DNN remain a reduced reproduction. Five-member ensembles deliberately differ from the original ten-member scaffold ensembles. No pretrained-backbone conclusions or wet-lab efficacy claims are supported. Novelty-aware Laplace is a testable project extension, and null/negative results are retained. Read the report and fidelity matrix before comparing scores across variants.

## Operational repair log

The first SWAG import required regex2024.11.6; the lock records the installed dependency. The first evaluation stopped before producing metrics because the original SWAG loader had rank-one covariance buffers while fitted buffers had rank20. `evaluate_saved.py` resizes buffers before strict loading and verifies every loaded tensor exactly. It leaves the frozen experiment source, model parameters and all selected settings unchanged. Use this evaluation entry point.

Source-scaler verification uses explicit float casts because the pinned implementation returns object arrays when its missing-value token is None. The training code uses ordinary numeric standardization and was unaffected. No scientific results were changed by this verification repair.
