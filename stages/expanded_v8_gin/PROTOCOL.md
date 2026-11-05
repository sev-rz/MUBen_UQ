# GIN (2D graph) reproduction in the Tier-B pipeline: protocol (FROZEN 2026-10-10, before any GIN unit was trained)
Student's instruction (2026-10-10): proceed with GIN; keep it simple; our pipeline with the authors' model and data classes; an efficient plan that does not weaken the statistics. Decision n0067. Sources: paper MU.pdf (arXiv 2306.10060v4) Appendix B and Table 5; MUBen code commit f0ec556.

## 1. Question and scope
How closely do MUBen's published GIN rows reproduce on the eight core datasets (bace, bbbp, clintox, tox21, esol, freesolv, lipo, qm7) on the authors' scaffold splits, with all of the paper's UQ methods?
Extension (no published GIN target exists: the released reports have GIN on scaffold only): the same design on the three random splits, so that all four backbones (DNN, ChemBERTa, GIN, later Uni-Mol) have the same design.
Methods exactly as the published rows: classification Deterministic, Temperature scaling, Focal Loss, MC dropout, SWAG, BBP, SGLD, Ensembles; regression Deterministic, MC dropout, SWAG, BBP, SGLD, Ensembles.

## 2. Pipeline
Our Tier-B framework (structure of tierb_chemberta.py: units, seeds, masked loss, validation selection, post-hoc and Bayesian methods, FROZEN markers, tierb_eval metrics) with the authors' pieces imported, not rewritten: the GIN encoder of muben.model.gin.GIN (atom embedding, torch_geometric GIN with jk="cat", global add pooling) and the featuriser muben.utils.chem.smiles_to_2d_graph. Our own output head (one Linear layer, Tier-B initialisation), our training loop and seeds.
Paper settings (Appendix B): 5 GIN layers, 128 hidden units, dropout 0.1, batch 128, constant learning rate 1e-4, 200 epochs, early-stopping tolerance 50, AdamW with weight decay 0.01 (as all Tier-B models). Parameter count must match the paper's 0.26 M (authors' code: 260,226 for a one-task regression head); checked at start.
MC dropout: dropout layers active, normalisation layers in evaluation mode. Any difference from the authors' implementation of a method is recorded in DECISIONS.md.

## 3. Design (identical to Tier B; nothing is reduced)
Scaffold: authors' split files, single-model seeds 0, 100, 200, one pooled 10-member ensemble (seeds 0..9). Random splits (extension): random-0..2 with the Tier-B algorithm, seed 100k, ensemble of 3. All eight datasets, all methods above, three repeats each. Cost is small (paper Table 5: 7.21 ms per step; upper bound about 2.3 GPU-hours for the full layout), so no seed, dataset or method is dropped to save time.

## 4. Targets and agreement rule (as Tier B, fixed now)
Targets: reports/primary/GIN-<dataset>.csv (mean and SD over the authors' three seeds). A row agrees at k if |reproduced mean - published mean| <= k x published SD, k = 1 and 2, over rows with published SD > 0; scaffold ensembles have no published SD and are not counted. Reported per dataset and overall, with a claims check on the same claim list as Tier B. A gap is named with its likely causes, never tuned away. Random-split GIN numbers are compared with no published value.

## 5. Bridge check against the authors' trainer (registered now)
The authors' unmodified run/run.py, descriptor_type 2D, paper settings above, uncertainty_method none, seeds 0-2, on ESOL and QM7 scaffold, metrics from their load_results and regression_metrics, against our pipeline's GIN Deterministic on the same datasets. Consistent if all four metrics (RMSE, MAE, NLL, CE) differ by less than 2 x the larger of the two seed SDs; otherwise the difference is reported as a pipeline difference and decision n0067 is revisited. Cheap (minutes) and run before the main grid.

## 6. Test isolation
Selection by validation only. A dataset's test set is evaluated only after all its units for that split are complete (FROZEN marker carrying this protocol's sha256 and the code digest); code changes after freezing invalidate the marker. No hyperparameter changes after seeing test results.

## 7. Statistics
No hypothesis tests. Descriptive agreement counts and sign rules as in Tier B and v6; all datasets and methods reported; no row dropped or re-run for an unfavourable result; the random-split part is labelled an extension; multiple comparisons are stated, not corrected. Cross-backbone claims (less expressive models better calibrated; method choice matters more for larger models) are specified before they are run, in a later registration, once Uni-Mol is decided.

## 8. Compute and gate
GPU Slurm jobs, resumable (per-unit completion markers). A timing pilot (two datasets) precedes the full run; the full run starts only if the projected cost is at most 4 GPU-hours. Outputs in expanded_v8_gin/ (models, results, frozen, DECISIONS.md, OPERATIONS_LOG.md). Laplace on GIN (last layer) is stage 7.3 with its own frozen protocol after this stage.
