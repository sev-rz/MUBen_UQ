# MUBen CPU expansion: frozen experimental protocol

Date: 4 October 2026. Supersedes the proposed expansion scope, not the original completed study. User authorized this expansion with the existing 1 CPU / 4 GB allocation, no GPU, and persistent Lustre storage.

## Questions and scope

Use the full supplied scaffold splits of BACE (activity classification), BBBP (permeability classification), and ESOL (aqueous solubility regression). These cover two therapeutic-screening endpoints and a distinct task type with small CPU-feasible datasets. They are benchmarks, not experimental therapeutic validation. No ChemBERTa or GPU runs.

Main model follows paper Appendix B: 200 normalized RDKit descriptors; eight total hidden layers, width128, GELU, dropout0.1. AdamW lr0.0002, wd0.01, batch256, max400epochs. Match upstream checkpoint ties (accept equality) and stop after more than50 consecutive non-improving validations. Xavier uniform and bias0.01; RNG call order is not claimed identical to upstream. Classification selects validation ROC-AUC; regression validation RMSE, using training-only target mean/population SD.

Three independent repetition groups: (0..4), (100..104), (200..204). Five-member ensembles, as requested; paper uses ten members for these scaffold datasets. Group first member supplies other post-hoc methods. Train Focal Loss independently with the same first seeds, alpha0.25/gamma2, using torchvision. Additional 9-layer/ReLU deterministic runs at first seeds isolate the pinned repository's architecture defaults; these are architecture sensitivity runs, not an assertion that those defaults produced the published results.

Methods: deterministic, MC dropout30, ensemble5, SWAG20 extraepochs and30samples; classification additionally temperature scaling and Focal Loss. Temperature follows source validation fitting: scalar initialized1, AdamWlr0.01/wd0.01,20epochs, frozen network. SWAG uses pinned source blockwise sampler, AdamWlr0.0001,20snapshots, no early stopping; copied snapshot inputs avoid CPU parameter-storage aliasing. This does not claim equivalence to full cross-layer covariance. ESOL Gaussian mean/softplus variance head, +1e-6 in training loss only, matching source. MC/ensemble/SWAG average predictive means and aleatoric variances as paper does; do not silently add between-model variance.

## Laplace and proposed extension

Classification: existing verified full129parameter affine-head Laplace, laplace-torch0.2.3, Bernoulli likelihood, GLM probit, alpha grid10^-4..10^4, validation NLL. ESOL: conditional Laplace over the mean head with frozen representation and frozen predicted noise; H=Phi^T diag(1/noise) Phi, covariance=(H+alpha I)^-1. Analytic implementation; verify Hessian with autograd. This differs from a joint mean-and-noise posterior. Posterior centers remain early-stopped estimates, not exact prior-specific MAP solutions.

Novelty distance d(x)=1-max_train Tanimoto(Morgan radius2,2048bits,chiralityfalse). Reference fingerprints use only training molecules. No dense all-pairs matrix is retained. Proposed extra variance lambda*d^2; lambda grid[0,.01,.1,1,10,100]. Classification adds to logit variance before probit approximation. ESOL adds to standardized predictive variance. Select lambda by validation NLL, after selecting baseline Laplace alpha. Save all grid results and choices before test access. Equal six-candidate controls: novelty-only (no Laplace variance) and constant inflation on top of Laplace. Lambda0 recovers the relevant base model exactly. No settings selected from tests.

Primary extension endpoint: per-dataset paired NLL difference vs ordinary Laplace. Also Brier/AUC and entropy-ranked AURC for classification; regression RMSE/MAE, paired Gaussian PIT calibration, 90% coverage and interval width, and variance-ranked retained MSE. Prespecified novelty subgroup is nearest-training similarity below0.4; always report counts. Improvements are hypotheses; publish negative and null results.

## Evaluation and provenance

All fit stages across all datasets finish before evaluation. BACE/BBBP have previously inspected tests: extension results there are exploratory. ESOL is a newly untouched task and its evaluation is held out from method design/tuning. Three seeds quantify initialization variation only.

Classification reports upstream BCE without extra clipping, positive-class15-bin ECE, MCE, ROC-AUC, trapezoidal PRC-AUC, Brier; auxiliary AURC. ESOL reports Gaussian NLL without normalizing constant for upstream comparison and full NLL separately. Audit upstream regression CE: current source broadcasts all labels against all predictive distributions. Save that exact reference-compatible value separately from the scientifically intended paired PIT calibration error. Never label them interchangeable. Match original result CSV metrics and report missing ensemble SD as unavailable.

Audit split hashes/counts/masks, invalid SMILES, canonical overlap, and chiral/nonchiral scaffold overlap. Fail on invalid molecules/non-finite features, replace NaN descriptors by0 as source. Save checkpoints, optimizer/RNG resume state, histories, cached features, fitted settings, individual predictions, checks, runtime and dependency lock. Source commit f0ec55667c1c4cc7b63369018fd011a96f31fdc4. Exact source defaults and paper settings remain separately labeled.

Persistent location: /l/users/sevinch.zoda/schub/projects/muben-laplace/work/expanded_v2. All execution goes through the SC Hub journal in existing job217442; no login-node training. Preserve the original completed package. Create a portable archive and SHA256 manifest on Lustre and copy the deliverables locally. Lustre survives allocation termination; institutional storage retention/backups are not guaranteed by this project.

References: https://arxiv.org/html/2306.10060v4 ; https://github.com/Yinghao-Li/MUBen ; https://github.com/AlexImmer/Laplace . The novelty correction is a project hypothesis, not a verified new research method.
