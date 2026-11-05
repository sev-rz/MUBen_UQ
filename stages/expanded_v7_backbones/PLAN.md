# Plan: widening the reproduction to GIN (2D graph) and Uni-Mol (3D) -- DRAFT, not frozen; nothing has been downloaded or run for Uni-Mol
Drafted 2026-10-10. Student's request: add runs for a 2D graph model (GIN) and a 3D model (Uni-Mol), referring back to the authors' paper (MU.pdf, arXiv 2306.10060v4) and code (Yinghao-Li/MUBen, commit f0ec556, cloned in work/repos/MUBen).
This file is the plan and the list of decisions; it becomes a frozen protocol (PROTOCOL.md + sha256, as in Tier B and v6) only after the student answers section 9.

## 1. What changes compared with Tier B (lessons applied)
- Tier B used our own reimplementation of the DNN and ChemBERTa pipelines (n0057: it matches the authors' code on DNN QM7 and ESOL). For GIN and Uni-Mol the authors' OWN code is the primary pipeline (run/run.py --descriptor_type 2D; run/unimol.py), with their metric functions (muben.utils.io.load_results, muben.utils.metrics.regression_metrics / classification_metrics) and the settings stated in the paper (Appendix B), not the example config.
- Every deviation from the paper's settings is recorded; nothing is tuned; the agreement rule is the Tier-B one (|reproduced - published| <= k x published SD, k = 1, 2, rows with SD > 0; scaffold ensembles have no published SD).
- A timing pilot comes before scale decisions (paper_reproduction playbook); the student chooses the scale.

## 2. Targets from the paper and the released reports
| Item | GIN (2D) | Uni-Mol (3D) | Source |
|---|---|---|---|
| Input | 2D molecular graph (PyG), authors' featurisation | 10 sets of 3D conformations plus the 2D graph per molecule; one of the 11 sampled per epoch for training, logits averaged over all 11 at inference | paper Appendix B; code muben/dataset/dataset_2d, dataset_unimol |
| Model | 5 GIN layers, 128 hidden units, dropout 0.1; 0.26 M parameters (we ran the authors' code: 260,226) | pre-trained on 209M conformations, checkpoint as initialisation; 47.59 M parameters | paper Table 5, Appendix B |
| Training | batch 128, 200 epochs, early-stopping tolerance 50, constant lr 1e-4 | batch 128, 100 epochs, tolerance 40, linear scheduler with 0.1 warm-up, peak lr 5e-5, dropout 0.1 | paper Appendix B (the code's argument default for Uni-Mol batch size is 32, so the paper's value must be passed explicitly) |
| Cost per step (authors' GPU, BBBP, batch 128) | 7.21 ms | 392.55 ms | paper Table 5 |
| Published scaffold results | reports/primary/GIN-<dataset>.csv (14 datasets; methods none, MCDropout, SWAG, BBP, SGLD, DeepEnsembles; plus TemperatureScaling, FocalLoss for classification) | reports/primary/Uni-Mol-<dataset>.csv (same methods) | repo |
| Published random-split results | NONE for GIN | reports/random/Uni-Mol-<dataset>.csv, 8 core datasets | repo |
| Published frozen-backbone results | none (not pre-trained) | reports/frozen: ChemBERTa, GROVER, Uni-Mol | repo |
Consequence: GIN can be compared with published numbers on scaffold only; GIN random-split runs would be an extension without a published target. Uni-Mol has published targets on scaffold, random and frozen.
Observation to carry forward (not a conclusion): published QM7 scaffold NLL is 5.25 for GIN and 4.62 for Uni-Mol but 10.30 for DNN, while our DNN reproduction (5.48) and the authors' code (5.47) agree with the level of the other backbones; the published DNN QM7 value looks like the outlier.

## 3. Paper claims these runs can test (to be extracted in full at registration)
(a) "models with lower expressiveness tend to exhibit better calibration" and larger models such as Uni-Mol are more confident, with subpar calibration especially for regression (paper results discussion, Figure 2 and Figure 4); (b) UQ method choice matters more for larger models (larger gap between Deterministic and the best UQ method for Uni-Mol than for DNN, Tables 1-2).
With DNN and ChemBERTa already reproduced, GIN and Uni-Mol give four backbones on the same eight datasets, enough to test (a) and (b) within our scope. Claims to be listed with number, table and metric before any run, as the playbook requires.

## 4. Feasibility (estimates, NOT measurements)
Upper bounds in GPU-hours from the real training-set sizes (bace 1210, bbbp 1631, clintox 1182, tox21 6264, esol 902, freesolv 513, lipo 3360, qm7 5464) and the Table-5 step times, ignoring validation passes and early stopping (ESTIMATE.txt):
- GIN, full Tier-B layout (about 17-20 training fits per dataset on scaffold, about 16-19 over the three random splits): at most 2.3 GPU-h in total. Cheap; GPU or CPU.
- Uni-Mol, full layout: at most 63 GPU-h (32.5 scaffold, 30.7 random). Reduced layouts: Deterministic x 3 seeds on scaffold at most 5.4 GPU-h (MC dropout and temperature scaling are post-hoc); plus a 10-member ensemble pool at most 18.0; the full scaffold layout at most 32.5.
These come from the authors' GPU; our RTX 5000 Ada will differ. A timing pilot (one dataset, one seed) replaces them before any scale is chosen. Storage (3 TB) is no issue.
Environment: the authors' environment is already built (work/repos/MUBen environment, torch 2.1.2, torch_geometric 2.4.0, torch_scatter and torch_cluster from prebuilt wheels, rich added). The authors' GIN code ran in it (2-epoch smoke, no downloads). Uni-Mol is NOT yet testable: it needs the files in section 7.

## 5. Stages and gates
Stage 7.0 (no compute): extract the claims table and published rows into a note; registration note with the agreement rule and what would count against reproduction; student answers section 9; freeze this plan as PROTOCOL.md.
Stage 7.1 GIN: authors' code, paper settings, seeds 0, 1, 2 (their seeds; they run their code, so we use theirs), 8 datasets on the scaffold files; all methods published for GIN (none, MC dropout, SWAG, BBP, SGLD, ensembles; temperature scaling and focal loss for classification). Compare per row with reports/primary. Random-split GIN: only if the student wants the extension (no published target); needs random-split CSV folders written with the Tier-B algorithm (random_state 0, 1, 2), because the authors' loader reads data/files/<dataset>/{train,valid,test}.csv.
Stage 7.2 Uni-Mol: after approval of the downloads (section 7): (i) pilot for timing and a smoke test of the conformer pipeline; (ii) gate: student chooses the scale from section 4 (suggested: Deterministic x 3 seeds plus the post-hoc methods on scaffold first, extended only if the pilot allows); (iii) runs checkpointed and resumable (<= 8 h GPU slices); (iv) comparison with reports/primary and, if run, reports/random.
Stage 7.3 Laplace on the new backbones (extension, own frozen protocol as stages 1-2): last layer on the saved authors'-code checkpoints (model_best.ckpt); same endpoints and sign rules; gated on 7.1/7.2 being done.
Stage 7.4 Synthesis: four-backbone table (agreement counts per backbone x split x dataset; claims (a) and (b)); report sections; handover.

## 6. Decision rules (proposed, fixed at freezing)
Agreement counting, claims check and "reproduced / partly / not" per target as in Tier B (TIERB_SUMMARY.md structure); a gap is named with its likely causes, never tuned away; negative results reported; the Laplace stage keeps the all-repeats sign rule and the 6-of-8 rule.

## 7. Downloads that need the student's approval (file, source, size)
1. Uni-Mol molecular-property datasets with the pre-computed conformations: molecular_property_prediction.tar.gz, 3,764,044,131 bytes (3.76 GB), https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz (the authors' README link; a server in China, so speed and availability are uncertain; the file is on the Uni-Mol side, not the MUBen repository).
2. Uni-Mol pre-trained checkpoint: mol_pre_no_h_220816.pt, 190,540,187 bytes (190.5 MB), https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt (the authors' README link).
Both would be fetched with bench.fetch (url, size and sha256 recorded in the journal) into the project, and unpacked in their own empty folders. Nothing else needs a download for GIN.

## 8. Risks and open points
- Uni-Mol conformations: the authors used the conformations prepared by the Uni-Mol repository. If the archive cannot be fetched, generating conformers with RDKit would be a deviation (different conformers) and would need its own decision.
- Random-split Uni-Mol results need conformers for exactly the molecules in each random split; they come from the same Uni-Mol files by SMILES, the authors' conversion script is assist/dataset_build_from_unimol.py.
- Time: Uni-Mol dominates; the pilot decides. The GPU partition has one GPU per student.
- The Tier-B random-split algorithm is ours (expanded_v3); published random-split Uni-Mol numbers used the authors' own random splits, so a split difference is a possible cause of gaps and will be stated.
- The authors' example config is not the paper's settings (lesson from the DNN check, n0057); paper settings are used and every argument is logged.

## 9. Decisions needed from the student
(a) Approve the two downloads in section 7? (b) GIN random splits as an extension without a published target: yes or no? (c) Uni-Mol scale: decide after the pilot; confirm that a pilot may run once the files are approved. (d) Laplace on GIN and Uni-Mol (stage 7.3): wanted, and gated as above? (e) Seeds: the authors' 0, 1, 2 for these backbones (recommended) rather than Tier B's 0/100/200?

## 10. REVISION 2026-10-10 (student's decision; supersedes sections 1 and 5 where they say the authors' code is the primary pipeline)
Decision: continue with our own Tier-B framework (tierb_* pipeline: units, seeds, FROZEN markers, tierb_eval metrics, Laplace and conformal stages) for the new backbones, not the authors' trainer.
Reason given: the earlier results did not differ much. What the evidence actually covers: DNN, Deterministic only, scaffold, ESOL and QM7 (n0057); ChemBERTa, other methods, other datasets and random splits were NOT cross-checked.
Practical form (proposed): "our pipeline, their model and data classes". Re-writing Uni-Mol (47.6 M parameters, 3D conformer input, atom-pair attention, a pre-trained checkpoint) from scratch would be a large and error-prone job, so the new backbone adapters import the authors' muben.model.gin / muben.model.unimol and muben.dataset.dataset_2d / dataset_unimol (as Tier B already imports their SWAG, BBP and SGLD modules), while training loop, selection, methods, seeds (0/100/200, random-0..2), metrics and Laplace stay ours. GIN could be a plain PyG reimplementation if the import proves awkward.
Consequences to state in the report: (1) all four backbones go through one pipeline, which makes the cross-backbone comparison cleaner; (2) fidelity of GIN and Uni-Mol is bounded by the authors' model code but not by their training code, so a bridge check is still wanted: GIN against the authors' trainer on 1-2 datasets (cheap, runs already verified) and Uni-Mol on one dataset at the pilot; (3) a gap against published numbers could now come from our training loop, which the DNN case suggests is acceptable but is not proven for these backbones; (4) seeds, the 8-vs-9-layer DNN depth and our random splits remain as in Tier B.
Unchanged: section 7 downloads still need approval before any Uni-Mol work; section 9 questions (a)-(e) still open; the plan is not frozen.
