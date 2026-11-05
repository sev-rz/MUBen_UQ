# Fidelity matrix and interpretation rules

Reference: MUBen arXiv2306.10060v4 and source commit f0ec55667c1c4cc7b63369018fd011a96f31fdc4. Values below describe the expanded experiment; the earlier study is preserved separately.

| Component | Reference | Expanded implementation | Status |
|---|---|---|---|
| Datasets | Broad molecular benchmark | Full BACE, BBBP, ESOL supplied splits | Reduced task coverage; regression added |
| Representation | Multiple pretrained and non-pretrained models | Normalized RDKit200 descriptor DNN | No claims about pretrained backbones |
| DNN architecture | Appendix B:8hidden128/GELU; current defaults:9total/ReLU | Main8/GELU plus9/ReLU deterministic sensitivity runs | Paper aligned; source discrepancy tested separately |
| Descriptor handling | DescriptaStorus rdkit2dnormalized, NaNs to0 | Same family and policy; versions locked | Version-dependent chemistry retained and disclosed |
| Training | AdamW, lr2e-4,wd.01,batch256,max400,patience50 | Same; ties accepted; stop at>50failures | Closer to source than prior earliest-tie implementation |
| Randomization | Authors' original seeds/RNG/software | Groups0..4,100..104,200..204; independent loader RNG | Not exact numerical replication |
| Ensembles | 10members for these scaffold datasets | 5members,3independent groups | User-requested reduction; larger than prior3 |
| MC dropout | 30samples; probabilities averaged | 30samples from first member in each group | Sampling count aligned |
| Temperature | Source20epochs AdamWlr.01, validation data | Same initialized scalar and frozen backbone | Replaces prior bounded scalar minimization |
| Focal Loss | torchvision defaults alpha.25,gamma2 | Same, independently trained | Additional original classification method |
| SWAG | 20snapshots/30samples; blockwise source sampler | Pinned SWAModel; additional AdamWlr1e-4 | CPU copy avoids first-snapshot aliasing |
| Regression | Standardize targets; Gaussian mean/softplus variance head | Train-only mean/SD; same loss and inverse variance scaling | New task type |
| Regression ensembles/MC/SWAG | Average means and individual variances | Same, no between-model variance term | Paper's simplified aggregation retained |
| Classification metrics | BCE, Brier, AUC, trapezoidal PRC,15-bin positive ECE/MCE | Same reported; old clipped NLL auxiliary only | No cross-definition ECE comparison |
| Regression NLL | PyTorch GaussianNLLLoss without constant | Same; full NLL saved separately | Like-for-like reference column |
| Regression calibration | Source all-pairs broadcast; intended per-example PIT | Both separately reported; exact source output checked | Source behavior is not silently repaired |
| Laplace | Not part of original MUBen | Verified library classifier; conditional analytic Gaussian mean-head extension | Extension, not reproduced original method |
| Novelty Laplace | Not part of original MUBen | Prespecified variance correction and2ablation controls | New tested project hypothesis; no novelty claim |
| Evaluation access | Original published tests | Existing BACE/BBBP exploratory, ESOL newly held out | No test-driven retuning |

The paper/source difference alone does not establish which architecture generated a published table. The sensitivity experiment changes depth and activation together; it cannot isolate their individual causal effects. Our constructor also does not reproduce the original wrapper's complete RNG initialization call order.

Classification Laplace prior selection reuses the verified first-stage implementation with NLL clipping1e-7; extension lambda selection uses BCE without extra clipping. Save all grids and identify saturation if present. This is a retained implementation detail, not hidden equivalence.

Original tables supply SD for most methods but not the published ensembles. Missing SD remains blank. Three repeated initializations on one test set are not three independent dataset replications or proof of statistical significance.
