MUBen on a CPU:expanded reproduction andchemical-novelty-aware Laplace

Course technical report | 4 October 2026 | 1 CPU, 4 GB RAM, no GPU

1. Objective and scope

MUBen benchmarks uncertainty quantification (UQ) across molecular representations and property tasks [1]. We reproduce a larger, explicitly bounded descriptor-DNN subset and test whether adding chemical unfamiliarity to last-layer Laplace improves predictive uncertainty. BACE models activity, BBBP models blood-brain-barrier permeability, and ESOL models aqueous solubility. These are screening-relevant benchmarks, not evidence of therapeutic efficacy.

This expansion adds regression, Focal Loss, SWAG, five-member ensembles, source-aligned temperature fitting and checkpoint selection, an architecture sensitivity study, and controlled extension ablations. The original two-dataset study is preserved. No pretrained model is fine-tuned; conclusions about pretrained molecular representations remain untested.

| Dataset | Task | Train / valid / test | Novel test count* |

| BACE | Classification | 1210 / 151 / 152 | 33 |

| BBBP | Classification | 1631 / 204 / 204 | 101 |

| ESOL | Regression | 902 / 113 / 113 | 68 |

*Prespecified subgroup: maximum Morgan-fingerprint similarity to training molecules below 0.4. Full supplied splits are retained; no test-driven resplitting.

Measured findings

BACE. Lowest mean NLL: Novelty only (0.618). Ordinary Laplace: 0.742; novelty-aware Laplace: 0.623; paired mean difference -0.118 (negative favors the extension).

BBBP. Lowest mean NLL: Focal Loss (0.662). Ordinary Laplace: 0.951; novelty-aware Laplace: 0.925; paired mean difference -0.027 (negative favors the extension).

ESOL. Lowest mean NLL: Laplace constant (0.353). Ordinary Laplace: 0.506; novelty-aware Laplace: 0.356; paired mean difference -0.151 (negative favors the extension).

The study produced 60 trained models and 93 evaluated dataset/method/repeat rows. All reported metrics were recomputed from saved predictions. Three seeds quantify initialization variation on fixed test sets; they do not establish population significance. BACE/BBBP analyses are exploratory because earlier test results informed the extension hypothesis; ESOL was newly held out.


---

2. Method and reproduction fidelity

The main DNN follows Appendix B: 200 normalized RDKit/DescriptaStorus features, eight hidden layers of width 128, GELU and dropout 0.1. Train with AdamW (learning rate 0.0002, weight decay 0.01), batch 256, at most 400 epochs. Checkpoints accept tied validation improvements and training stops after more than 50 unsuccessful validations, matching source logic. Validation selects ROC-AUC for classification and RMSE for regression. Target mean and population SD are fitted on ESOL training labels only.

| Component | Implementation and remaining deviation |

| Repeated ensembles | Three groups: seeds 0-4, 100-104, 200-204; five members per group. Original scaffold ensembles use ten [1, App.C.1]. |

| Temperature / Focal Loss | Temperature: validation-only scalar, initialized at 1, AdamW lr 0.01, 20 epochs. Focal: independent training, torchvision alpha 0.25/gamma 2. |

| MC dropout / SWAG | 30 samples each; SWAG: 20 extra epochs at lr 0.0001, pinned blockwise low-rank-plus-diagonal sampler. Copy snapshots to avoid CPU storage aliasing. |

| ESOL likelihood | Mean and softplus variance head, Gaussian NLL; inverse-scale mean and variance. Ensemble/MC/SWAG average variances as in the paper; no between-model term. |

| Architecture sensitivity | Separate nine-layer/ReLU deterministic variant reflects pinned defaults. Main8/GELU follows paper; published-run arguments are not reconstructed from defaults. |

Software versions, RNG initialization order and selected seed groups differ from the authors; this is not bitwise replication. FIDELITY.md records the complete matrix. The source commit is f0ec55667c1c4cc7b63369018fd011a96f31fdc4 [2].

Laplace and chemical novelty

Freeze the representation and fit a full 129-parameter affine mean/logit head approximation. Classification uses laplace-torch with a fixed-zero reference logit, verified against Bernoulli curvature. ESOL uses conditional Gaussian curvature with the learned noise head fixed:

Sigma = (H + alpha I)^(-1);  v(x) = phi(x)^T Sigma phi(x)H_class = sum p_i(1-p_i) phi_i phi_i^TH_reg = sum phi_i phi_i^T / noise_i

Let d(x)=1-max training Tanimoto similarity, with Morgan radius 2, 2048 bits and chirality disabled. The proposed correction is v_new=v+lambda*d(x)^2. Classification predicts sigmoid[m/sqrt(1+pi*v_new/8)]; regression adds v_new to the predicted noise variance. This is a heuristic correction, not an exact Bayesian posterior.

Select alpha from 10^-4 to 10^4 and lambda from {0,.01,.1,1,10,100} using validation NLL only. Compare ordinary Laplace, novelty-only variance, constant inflation plus Laplace, and novelty-aware Laplace with equal six-candidate lambda grids. Posterior centers are early-stopped estimates, not prior-specific MAP optima. ESOL models only conditional mean-head uncertainty.


---

3. Comparison with the authors' results

Each row compares the paper-specified DNN implementation here with the authors' pinned primary result CSV [2]. Values are mean +/- SD across three initialization groups here; published ensemble SD is unavailable. Classification primary metric is ROC-AUC (higher better); ESOL is RMSE (lower better). NLL is lower better. All metrics, SDs and absolute differences are in results/original_comparison.csv.

Classification

| Dataset / method | Original primary | Our primary | Original NLL | Our NLL |

| BACE / Deterministic | 0.819 +/- 0.016 | 0.812 +/- 0.009 | 0.895 +/- 0.080 | 0.837 +/- 0.068 |

| BACE / Temperature | 0.819 +/- 0.016 | 0.812 +/- 0.009 | 0.741 +/- 0.047 | 0.766 +/- 0.054 |

| BACE / MC dropout | 0.817 +/- 0.012 | 0.809 +/- 0.008 | 0.767 +/- 0.045 | 0.808 +/- 0.053 |

| BACE / Ensemble 5 | 0.821 (SD n/a) | 0.815 +/- 0.003 | 0.799 (SD n/a) | 0.796 +/- 0.025 |

| BACE / Focal Loss | 0.819 +/- 0.011 | 0.809 +/- 0.010 | 0.691 +/- 0.010 | 0.713 +/- 0.014 |

| BACE / SWAG | 0.817 +/- 0.012 | 0.812 +/- 0.011 | 0.873 +/- 0.064 | 0.868 +/- 0.076 |

| BBBP / Deterministic | 0.672 +/- 0.013 | 0.686 +/- 0.014 | 3.815 +/- 4.078 | 0.965 +/- 0.054 |

| BBBP / Temperature | 0.672 +/- 0.013 | 0.686 +/- 0.014 | 2.134 +/- 1.896 | 1.025 +/- 0.121 |

| BBBP / MC dropout | 0.667 +/- 0.015 | 0.686 +/- 0.012 | 2.170 +/- 1.748 | 0.936 +/- 0.048 |

| BBBP / Ensemble 5 | 0.702 (SD n/a) | 0.691 +/- 0.009 | 0.904 (SD n/a) | 0.931 +/- 0.011 |

| BBBP / Focal Loss | 0.669 +/- 0.005 | 0.686 +/- 0.004 | 0.892 +/- 0.342 | 0.662 +/- 0.002 |

| BBBP / SWAG | 0.678 +/- 0.014 | 0.691 +/- 0.016 | 4.841 +/- 5.541 | 1.033 +/- 0.036 |

Regression

| Dataset / method | Original primary | Our primary | Original NLL | Our NLL |

| ESOL / Deterministic | 0.905 +/- 0.029 | 0.913 +/- 0.002 | 1.003 +/- 0.376 | 0.675 +/- 0.129 |

| ESOL / MC dropout | 0.913 +/- 0.021 | 0.914 +/- 0.007 | 1.025 +/- 0.370 | 0.638 +/- 0.152 |

| ESOL / Ensemble 5 | 0.869 (SD n/a) | 0.871 +/- 0.017 | 0.567 (SD n/a) | 0.545 +/- 0.133 |

| ESOL / SWAG | 0.916 +/- 0.024 | 0.933 +/- 0.005 | 0.946 +/- 0.343 | 0.700 +/- 0.079 |

These differences are descriptive, not evidence that our DNN improves on the paper. Ensemble size, architecture/source ambiguity, seeds, software and numerical details remain different. Large published BBBP NLL SD indicates unstable probability scores; close AUC does not imply close calibration. The published ensemble CSV does not provide variability across repeated ensembles.

Classification comparison uses unclipped PyTorch BCE, binary Brier, trapezoidal PRC-AUC and positive-class15-bin ECE/MCE. ESOL NLL omits the Gaussian normalization constant, matching source; adding 0.91894 gives full NLL. Neither is interchangeable with classification NLL or comparable in magnitude across differently scaled regression targets.


---

4. Extension results and ablations

All hyperparameters were frozen before this version's test evaluation. The primary comparison is novelty-aware versus ordinary Laplace NLL. Constant inflation tests whether merely softening/widening predictions is sufficient; novelty-only tests whether the posterior variance is needed. Means +/- sample SD use three seeds.

| Dataset / method | NLL | Selective risk* | Selected lambda (3 seeds) |

| BACE / Laplace | 0.742 +/- 0.042 | 0.269 +/- 0.013 | - |

| BACE / Novelty only | 0.618 +/- 0.016 | 0.230 +/- 0.012 | 100.0, 100.0, 100.0 |

| BACE / Laplace constant | 0.636 +/- 0.018 | 0.246 +/- 0.011 | 10.0, 10.0, 10.0 |

| BACE / Novelty Laplace | 0.623 +/- 0.016 | 0.241 +/- 0.017 | 100.0, 100.0, 100.0 |

| BBBP / Laplace | 0.951 +/- 0.053 | 0.292 +/- 0.009 | - |

| BBBP / Novelty only | 0.883 +/- 0.098 | 0.289 +/- 0.008 | 10.0, 0.0, 1.0 |

| BBBP / Laplace constant | 0.948 +/- 0.055 | 0.292 +/- 0.009 | 0.1, 0.0, 0.0 |

| BBBP / Novelty Laplace | 0.925 +/- 0.037 | 0.288 +/- 0.008 | 1.0, 0.0, 1.0 |

| ESOL / Laplace | 0.506 +/- 0.090 | 0.507 +/- 0.017 | - |

| ESOL / Novelty only | 0.404 +/- 0.038 | 0.532 +/- 0.019 | 0.1, 0.1, 0.1 |

| ESOL / Laplace constant | 0.353 +/- 0.014 | 0.507 +/- 0.017 | 0.1, 0.1, 0.1 |

| ESOL / Novelty Laplace | 0.356 +/- 0.028 | 0.523 +/- 0.017 | 0.1, 0.1, 0.1 |

*Area under retained-error curve for classification and retained-MSE curve for regression; lower better. Different task scales cannot be averaged meaningfully.

Figure 1. Each point is one seed's NLL difference from ordinary Laplace; bars show the three-seed mean. Zero means no change. These are paired initialization comparisons on the same molecules, not independent test samples.

BACE: extension NLL decreased by 0.1183. Constant-inflation control NLL 0.636; novelty-only 0.618. 

BBBP: extension NLL decreased by 0.0267. Constant-inflation control NLL 0.948; novelty-only 0.883. 

ESOL: extension NLL decreased by 0.1508. Constant-inflation control NLL 0.353; novelty-only 0.404. 

The ablations do not establish that combining chemical novelty with Laplace is the best strategy. Novelty alone has lower NLL on both classification tasks; constant inflation slightly outperforms the combination on ESOL. The combination improves classification AURC versus Laplace, but worsens ESOL retained-MSE AURC. BACE lambda selections reach the upper grid boundary; no post-test grid expansion was made.


---

5. Reliability, selection and source checks

Figure 2. Error among retained predictions, averaged across seeds. Classification ranks by predictive entropy; regression by predictive variance. Curves show 20-100% coverage, while saved AURC includes all nonzero coverages.

| Dataset / method | Novel count | Novel NLL | Familiar NLL |

| BACE / Laplace | 33 | 0.551 | 0.795 |

| BACE / Novelty Laplace | 33 | 0.561 | 0.641 |

| BBBP / Laplace | 101 | 1.039 | 0.865 |

| BBBP / Novelty Laplace | 101 | 1.001 | 0.850 |

| ESOL / Laplace | 68 | 0.632 | 0.316 |

| ESOL / Novelty Laplace | 68 | 0.433 | 0.238 |

The similarity threshold 0.4 was fixed before evaluation. The BACE novel subgroup NLL worsens slightly, despite improved overall NLL; its gain comes from more familiar molecules. BBBP and ESOL novel-subgroup NLL improve. Counts are small and these comparisons are descriptive.

Regression calibration: two different quantities

The pinned upstream regression_calibration_error broadcasts N labels against N predictive distributions, producing an N-by-N matrix. That is different from pairing each molecule's prediction with its own label. We reproduce that source value only for reference comparison and additionally report the paired probability-integral-transform (PIT) calibration error. A direct numerical check verifies exact source agreement; neither definition is silently substituted.

| ESOL method | Source-compatible CE | Paired PIT CE | 90%coverage / width |

| Deterministic | 0.046 +/- 0.002 | 0.006 +/- 0.003 | 0.696 / 1.802 |

| Ensemble 5 | 0.046 +/- 0.003 | 0.005 +/- 0.002 | 0.705 / 1.847 |

| Laplace | 0.044 +/- 0.002 | 0.004 +/- 0.002 | 0.755 / 2.082 |

| Novelty Laplace | 0.037 +/- 0.002 | 0.001 +/- 0.000 | 0.870 / 2.518 |

Coverage is the empirical fraction in a Gaussian 90% interval; width is in original ESOL units. A wider interval can improve coverage without improving ranking or usefulness. Paired PIT error uses 19 cumulative probabilities between 0 and 1. Source-compatible CE does not validate per-molecule calibration.

Checks completed

Recomputed all 93 result rows from stored predictions. Checked classifier curvature against analytic/autograd Hessians and library predictive probabilities; Gaussian mean-head Hessian against autograd; lambda = 0 against its base model; all checkpoint reloads; source Gaussian loss, target-scaler roundtrip and exact regression CE. Train/test canonical overlaps: BACE=0, BBBP=0, ESOL=0.


---

6. Compute, limitations and conclusions

The 60 saved training runs completed 7,802 epochs in aggregate. Summed measured training invocation time was 184.4s. The successful fitting stage took 222.5s; evaluation took 10.1s. Peak process RSS was 458MiB. These are measurements on one allocated CPU; process RSS excludes other job processes. Setup, downloads, report authoring and failed attempts are excluded from these stage timings.

Lustre stores raw inputs, feature caches, checkpoints, optimizer/RNG restart states, uncertainty parameters, predictions, code, logs and hashes. Allocation expiry stops computation but does not delete this project directory. Institutional retention policies still apply; the downloadable archive provides a second copy. Raw inference timings are saved but differ in loading overhead, so no inference-speedup claim is made.

Architecture sensitivity

| Dataset | Paper 8/GELU primary | Source 9/ReLU primary | NLL: paper / source |

| BACE | 0.812 +/- 0.009 | 0.806 +/- 0.008 | 0.837 / 0.873 |

| BBBP | 0.686 +/- 0.014 | 0.686 +/- 0.007 | 0.965 / 1.133 |

| ESOL | 0.913 +/- 0.002 | 0.842 +/- 0.022 | 0.675 / 0.452 |

Depth and activation change together, so this is sensitivity analysis rather than isolation of either effect. Source defaults cannot prove the historical configuration behind published scores. Software/RNG differences and five-versus-ten ensemble members remain limitations.

What is and is not reproduced

Supported within scope: MUBen concludes that ensembles improve on deterministic predictions [1, Sec.6]. Our five-member ensembles improve both NLL and AUC/RMSE on all three tasks. Partially reproduced: the paper recommends temperature scaling and MC dropout for classification; MC dropout improves NLL on both tasks here, but temperature scaling improves BACE and worsens BBBP. Not tested: the claims about pretrained backbones, BBP/SGLD for regression, and random versus scaffold splits. These small fixed tests cannot overturn the broader benchmark.

The extension reduces NLL relative to ordinary Laplace on all three tasks, but simpler controls prevent a claim that their combination is superior. BACE/BBBP remain exploratory. Small reused validation sets, boundary hyperparameters and frozen representations constrain inference. ESOL coverage improves toward 90% while intervals widen and uncertainty ranking worsens. The contribution is a verified reproduction and controlled extension study, not a universal improvement or established research novelty.

Reproduce and inspect

Persistent root: /l/users/sevinch.zoda/schub/projects/muben-laplace/work/expanded_v2Run run_expansion.py --stage fit, then evaluate_saved.py, then analyze_expansion.py in an allocated CPU environment. README.md explains frozen-run protection and fresh reproduction. FROZEN.json, FIDELITY.md, original_comparison.csv and MANIFEST.json provide traceability.

[1] Li et al. MUBen: Benchmarking the Uncertainty of Molecular Representation Models. arXiv:2306.10060v4, 2024. https://arxiv.org/html/2306.10060v4

[2] Authors' source and primary result CSVs, commit f0ec55667c1c4cc7b63369018fd011a96f31fdc4. https://github.com/Yinghao-Li/MUBen

[3] Daxberger et al. Laplace Redux, NeurIPS 2021; laplace-torch 0.2.3. https://github.com/AlexImmer/Laplace

[4] Related work on molecular uncertainty under shift: CoDrug, NeurIPS 2023. https://arxiv.org/abs/2310.12033
