# Frozen protocols: index

Each protocol is the first file of its stage and was hashed before the stage's results existed; the sha256 is stored in `PROTOCOL.sha256` next to it. Protocols use internal labels from the order of work; this table maps them to the reports.

| Stage folder | Protocol | Internal label | Question | Report |
|---|---|---|---|---|
| `expanded_v4` | `PROTOCOL.md` (2f1fb3ae) | Tier B | reproduction of DNN and ChemBERTa, eight datasets, all UQ methods | R1 H1 |
| `expanded_v5_conformal` | `PROTOCOL.md` (040ea3cd), `PROTOCOL_v2.md` (5cbbf963), `PROTOCOL_unimol.md` (15d857ff) | | does conformal coverage hold under scaffold shift; Laplace base against plain networks and ensembles | R2 H6 |
| `expanded_v6_laplace` | `PROTOCOL.md` (7bde8e21), `chemberta/PROTOCOL.md` (f133fd02) | | last-layer Laplace on the DNN and ChemBERTa; novelty variants | R2 H1, H3 |
| `expanded_v8_gin` | `PROTOCOL.md` (6ae86f97), `laplace/PROTOCOL.md` (89fd53e4) | | GIN reproduction, bridge check and its Laplace stage | R1 H1, H2; R2 H1 |
| `expanded_v10_unimol` | `PROTOCOL.md` (8ab19980), `PROTOCOL_ADDENDUM_ext.md` (fbefa527), `laplace/PROTOCOL.md` (54c8025f) | | Uni-Mol reproduction, extension methods, Laplace and its conformer-averaging control | R1 H1, H2; R2 H1 |
| `expanded_v13_schedule` | `PROTOCOL.md` (0f767b85) | item 9 | does the Laplace gain depend on the learning-rate schedule | R2 H2 |
| `expanded_v14_laplace10` | `PROTOCOL.md` (450c1f76) | item B | Laplace on 10 to 12 networks per dataset | R2 H1 |
| `expanded_v15_scale_widegrid` | `PROTOCOL.md` (41e645e2) | B1, B2 | Laplace against a one-parameter recalibration; wide prior grid | R2 H3, H4 |
| `expanded_v16_round1` | `PROTOCOL.md` (150be162), `PROTOCOL_ADDENDUM.md` | C1, C2, C6 | few-label and similarity-weighted conformal; conformal under the linear schedule | R2 H7 |
| `expanded_v17_gin_linear` | `PROTOCOL.md` (48b2ae3c) | B3 | is the GIN classification gap a schedule difference | R1 H2 |
| `expanded_v18_round2` | `PROTOCOL.md` (b7961f11) | C3, C4 | evidence-based prior; Laplace on every ensemble member | R2 H4, H5 |

`expanded_v11_chemberta_sched` and `expanded_v12_headinit` hold the ChemBERTa schedule and head-initialisation arms (R1 H2) without a separate protocol; their settings are given in Report 1, section 4. The early stages `expanded_v2` and `expanded_v3` (first CPU expansion, random-split algorithm, active learning) are not part of the reported findings.

## Hypotheses and outcomes

| Hypothesis (written before the run) | Outcome |
|---|---|
| Conformal on Uni-Mol: classification coverage at least 5 points below target; regression within 0.88 to 0.93; no contrast reaches 6 of 8 | supported |
| Cross-model claims: larger models worse calibrated; method choice matters more for larger models (6 of 8 datasets) | not supported |
| Laplace improves on at least 6 of 8 datasets for every backbone (ten or more networks) | rejected (GIN 4 of 8) |
| Ten-network verdicts match the three-repeat verdicts on at least 7 of 8 datasets | rejected (GIN and Uni-Mol 6) |
| The Laplace gain shrinks under a linear schedule (ratio below 0.5) | supported (0.41) |
| Laplace beats a one-scalar recalibration on at least 6 of 8 datasets for at most two of four backbones | supported (none) |
| A wide prior grid leaves verdicts unchanged (at least 7 of 8) | supported (8 of 8, all backbones) |
| GIN with the linear schedule is consistent with the authors' runs on at least 18 of 20 comparisons | supported (20 of 20) |
| Balanced few-label calibration reaches 0.88 at k <= 50 on two of three backbones | rejected (one); target-only calibration at k = 20 does reach it |
| Similarity weighting does not rescue classification coverage | rejected (0.86 to 0.91) |
| Classification coverage stays below 0.88 under the linear schedule | supported (0.82) |
| Evidence prior larger and worse by at least 0.02 NLL on 3 of 4 backbones; still improves on 4 of 8 datasets for 2 backbones | both supported |
| Laplace on every ensemble member improves on at least 6 of 8 datasets for 3 backbones; ensemble gain smaller than single-network gain | first rejected (at most 5), second supported |
