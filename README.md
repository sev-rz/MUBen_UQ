# muben-uq

Reproduction of **MUBen** (Li et al., *MUBen: Benchmarking the Uncertainty of Molecular Representation Models*, arXiv:2306.10060v4, code commit `f0ec556`) and two extensions it does not contain, tested on new chemical scaffolds: **last-layer Laplace** and **split conformal prediction**. Course project, CBIO7215, MBZUAI, 2026.

Two technical reports document the work; this repository holds the code behind them.

| | |
|---|---|
| [`reports/Report1_Reproduction.pdf`](reports/Report1_Reproduction.pdf) | Report 1, reproduction of the paper: contributions and limitations of MUBen, three hypotheses (the tables reproduce; our pipeline is interchangeable with the authors' trainers; the paper's claims hold), implementation details |
| [`reports/Report2_Extension.pdf`](reports/Report2_Extension.pdf) | Report 2, extension: seven hypotheses on last-layer Laplace and conformal prediction under scaffold shift, each with a verdict |

HTML versions of both reports are in `reports/` (they need the bundled `reports/katex/` folder for the equations).

## Findings in brief

1. **The published tables reproduce partly.** About 77 % of DNN and ChemBERTa rows lie within two published SDs (a perfect copy would reach roughly 82 to 87 % because the published SDs come from three seeds); regression calibration and ChemBERTa and GIN on scaffold splits agree least.
2. **Our pipeline matches the authors' own trainers wherever the settings are matched** (Uni-Mol 13 of 13 comparisons, GIN regression 16 of 16, GIN classification 20 of 20 once the authors' learning-rate schedule is used, ChemBERTa under either schedule). Two published numbers are not reproduced by the authors' code either: the DNN QM7 NLL (10.30; their code gives 5.47, ours 5.48) and the ChemBERTa ESOL NLL (2.28).
3. **The learning-rate schedule moves calibration metrics as much as the choice of method** (GIN: the paper states a constant rate, the released code defaults to a linear one; ChemBERTa ESOL NLL changes by about 0.27 between the two).
4. **Last-layer Laplace lowers NLL on new scaffolds on average** (negative in 32 of 32 model-dataset cells) but mostly by inflating variances: one fitted scalar per task does about as well, the benefit shrinks by about 60 % under a better schedule, and a validation-free prior gives much less.
5. **Conformal classification loses coverage under scaffold shift** (0.75 to 0.82 against a 0.90 target, for every uncertainty method), regression holds (0.89 to 0.92), and about 20 labelled molecules from the target scaffolds restore coverage.

All extension results are **exploratory**: the test sets had been used for the baselines before the extensions. Numbers and caveats are in the reports; this README does not repeat them.

## Repository layout

```
muben-uq/
├── reports/              the two reports (PDF, HTML) and their figures
├── report_generator/     scripts that build the reports and figures from the result tables
├── stages/               one folder per experiment: scripts, frozen PROTOCOL.md (with sha256), summary tables
├── results_tables/       CSV tables behind the reported numbers
├── docs/                 MATH.md (derivations), PROTOCOL_INDEX.md (every protocol and what it tested), STATUS.md (open items)
├── tests/                pytest tests of the conformal and Laplace core
├── toy/                  self-contained synthetic demo (numpy only, a few seconds)
├── figures/              figures used in the reports and the presentation
└── requirements.txt, LICENSE
```

### Which code produced which part of the reports

| Stage folder | Content | Report sections |
|---|---|---|
| `expanded_v4` | DNN and ChemBERTa pipeline; comparison with the published tables | R1: 3 |
| `expanded_v8_gin`, `expanded_v10_unimol` | GIN and Uni-Mol with the authors' encoders; comparisons; their Laplace stages; Uni-Mol conformer control | R1: 3, 5; R2: 2 |
| `expanded_v9_checks` | bridge analysis against the authors' trainers, descriptor drift, random-split identity, statistics, shift-aware conformal check | R1: 4; R2: 2, 7, 8 |
| `expanded_v11_chemberta_sched`, `expanded_v12_headinit` | ChemBERTa schedule and head-initialisation arms; GIN head initialisation | R1: 4.2, 4.3 |
| `expanded_v17_gin_linear` | GIN with the linear schedule against the authors' ten-seed runs | R1: 4.3 |
| `expanded_v6_laplace`, `expanded_v14_laplace10` | last-layer Laplace on 3, then 10 to 12 networks per dataset | R2: 2 |
| `expanded_v13_schedule` | ChemBERTa linear schedule and its Laplace gain | R2: 3 |
| `expanded_v15_scale_widegrid` | one-parameter recalibration baseline; wide prior grid | R2: 4, 5 |
| `expanded_v18_round2` | evidence-based prior; Laplace on every ensemble member | R2: 5, 6 |
| `expanded_v5_conformal`, `expanded_v16_round1` | split conformal prediction; few-label and similarity-weighted calibration; conformal under the linear schedule | R2: 7, 8 |

Folder names are the order in which the stages were run. The protocols use internal labels (&ldquo;Tier B&rdquo;, &ldquo;item 9&rdquo;, B1&ndash;B3, C1&ndash;C6); `docs/PROTOCOL_INDEX.md` maps each to the report hypothesis it tests.

## Quick start

```bash
pip install -r requirements.txt
python toy/toy_laplace_conformal.py     # synthetic demo: Laplace, recalibration and conformal coverage under shift
pytest tests/                            # unit tests (conformal tests are skipped if RDKit is not installed)
python report_generator/build_final.py   # rebuild both reports (needs Google Chrome for the PDF step)
```

The demo trains a logistic last layer on &ldquo;familiar chemistry&rdquo;, tests it on a shifted &ldquo;new scaffold&rdquo;, and prints Laplace and temperature-scaling NLL, conformal coverage with a validation calibration set (about 0.77 against 0.91 on the same chemistry), and coverage after calibrating on 20 or 50 labelled molecules of the new scaffold (about 0.89 to 0.93).

## Reproducing the full study

The runs used the MBZUAI Slurm cluster (one GPU, FP32, deterministic algorithms). Requirements:

1. **Authors' code**: MUBen at commit `f0ec556`, used unmodified (not redistributed here), with its data files and published tables, and the public Uni-Mol and ChemBERTa-77M-MLM checkpoints. Stage scripts import its `muben` package.
2. **Paths**: scripts read their inputs from one working directory and contain the absolute cluster path they were run with. Run `scripts/set_work_dir.sh /your/path/to/work` once to point them at your own directory (protocols and result files are left unchanged so their hashes stay valid). Trained checkpoints, cached features and per-molecule prediction files are regenerated by the scripts and are not in the repository.
3. **Order** (read each stage's `PROTOCOL.md` first):
   - reproduction, per backbone: `prep` (features), `worker` (resumable training units), `freeze` (a FROZEN marker per split and dataset with the protocol and code digests), `evaluate` (test metrics only behind a FROZEN marker);
   - comparison with the published tables and bridge checks against the authors' trainers;
   - Laplace stages, then conformal stages, then the analysis scripts of each stage.
4. **Environment**: `requirements.txt` for the demo and tests; the exact locked versions used on the cluster are in `requirements/` (CPU base, GPU stack, Laplace stage).
5. **Integrity**: `docs/MANIFEST.csv` lists the sha256 of every file of the code package (paths relative to `stages/`); `docs/CODE_INDEX.md` describes the files.

## Conventions

- **Protocols before runs.** Each stage has a `PROTOCOL.md` with its question, endpoints, decision rules and hypotheses, hashed before any result exists; changes are recorded as addenda, never edited in place.
- **Test isolation.** Choices (including the Laplace prior strength and all recalibration parameters) use training or validation data only; test evaluation is gated by FROZEN markers.
- **Bridge checks.** Every gap with the paper was compared with the authors' own, unmodified trainer.
- **Exploratory labels.** No result is called confirmed.

## Data and credits

- MUBen: Li, Kong, Du, Yu, Zhuang, Mu, Zhang, arXiv:2306.10060; code github.com/Yinghao-Li/MUBen (used unmodified, not redistributed).
- Models: ChemBERTa-77M-MLM and Uni-Mol public checkpoints; GIN encoder and featuriser from the MUBen package.
- Datasets: BACE, BBBP, ClinTox, Tox21, ESOL, FreeSolv, Lipo, QM7, with the scaffold splits of the MUBen data files.

## License

MIT (see `LICENSE`). The authors' MUBen code keeps its own license.
