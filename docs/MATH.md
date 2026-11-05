# Derivations used in this project

Notation: `phi(x) = [h(x); 1]` is the penultimate representation plus a bias; `theta` the `D` trained last-layer weights (one set per task); `m_i in {0,1}` masks missing labels; `alpha` the prior precision, `theta ~ N(0, alpha^-1 I)`.

## 1. Last-layer Laplace

The posterior over the last layer is approximated by a Gaussian at the trained weights: `p(theta | data) ~ N(theta*, (H + alpha I)^-1)`, where `H` is the Hessian of the negative log-likelihood of the observed training labels. The network up to the last layer is a fixed feature extractor. The networks are early-stopped, so `theta*` is not an exact MAP point (stated everywhere).

**Classification** (independent affine Bernoulli model per task), `s_i = sigmoid(theta^T phi_i)`:

```
H = sum_i m_i s_i (1 - s_i) phi_i phi_i^T
v(x) = phi(x)^T (H + alpha I)^-1 phi(x)
p(y=1 | x) ~ sigmoid( z / sqrt(1 + pi v(x) / 8) ),   z = theta^T phi(x)      (probit approximation)
```

**Regression** (conditional Laplace over the mean head; the noise head is frozen): `mu = theta^T phi`, `sigma^2 = softplus(theta_n^T phi)`.

```
H = sum_i m_i phi_i phi_i^T / sigma_i^2
y | x ~ N( mu(x), sigma^2(x) + phi(x)^T (H + alpha I)^-1 phi(x) )
```

The predictive mean does not change; only the variance grows. RMSE and MAE are therefore unchanged by construction, and the NLL gain comes from wider variances. Targets are standardised on the training labels and converted back (mean times scale plus shift, variance times scale squared).

**Why NLL improves.** Gaussian NLL per molecule is `0.5 [log sigma^2 + (y - mu)^2 / sigma^2]`. An overconfident network pays a large second term where it errs; widening `sigma^2` there removes the penalty, with the optimum at `sigma^2 ~ squared error`.

**Choosing alpha.**
- *Validation grid* (primary, `expanded_v6` to `v17`): per task, the minimiser of the Laplace validation NLL over `10^-4 ... 10^4` (nine values; `10^-8 ... 10^4` in the sensitivity analyses). An upper-edge pick means "no posterior effect"; a lower-edge pick means the optimum may lie below the grid.
- *Evidence* (`expanded_v18`): the Laplace marginal likelihood, up to terms independent of `alpha`,

```
E(alpha) = (D/2) log alpha - (alpha/2) ||theta||^2 - (1/2) sum_k log(lambda_k + alpha)
```

with `lambda_k` the eigenvalues of the data Hessian on the training set; `alpha_ev = argmax E` over 121 log-spaced values. No validation or test data enter the choice. Fixed point (MacKay): `alpha = gamma / ||theta||^2`, `gamma = sum_k lambda_k / (lambda_k + alpha)`.

**One-parameter recalibration baseline** (`expanded_v15`): regression, a variance multiplier `s = mean_i (y_i - mu_i)^2 / sigma_i^2` over observed validation labels (the closed-form maximum-likelihood value); classification, a temperature `T` minimising validation NLL, `p = sigmoid(z / T)`.

## 2. Split conformal prediction

Hold out a calibration set of `n` labelled molecules (here the validation split). With a nonconformity score `s(x, y)`:

```
q = the ceil((n + 1)(1 - a))-th smallest calibration score     (infinite if that rank exceeds n)
C(x) = { y : s(x, y) <= q }
```

- Classification, **LAC**: `s(x, y) = 1 - p(y | x)`; for binary tasks the set is empty, `{0}`, `{1}` or `{0,1}`.
- Regression, **normalised**: `s(x, y) = |y - mu(x)| / sigma(x)`, interval `mu +/- q sigma`.

**Guarantee.** If calibration and test points are exchangeable, `1 - a <= P(y in C(x)) <= 1 - a + 1/(n + 1)`, marginally, for any model. Scaffold shift makes validation scaffolds and test scaffolds non-exchangeable, so the guarantee does not apply; validation scaffolds are typically easier, which gives the systematic coverage shortfall seen for classification. Regression holds because the score divides by the predicted SD, which grows on hard molecules.

**Mondrian conformal.** The quantile is computed within groups (here novel versus familiar, by maximum Tanimoto similarity to the training set below 0.4), giving a group-conditional guarantee.

**Few-label calibration** (`expanded_v16`, C1). Labelled target molecules `L` (whole scaffold groups) are added to the calibration scores: *pooled* (validation and labelled scores together), *balanced* (weighted quantile with half the mass on each), *target-only* (labelled scores alone, `|L| >= 20`). The weighted quantile uses weights `w_i` normalised to total mass `n/(n+1)` plus a point mass `1/(n+1)` at infinity.

**Similarity-weighted conformal** (`expanded_v16`, C2; Tibshirani et al. 2019). Weights `w(x) = p(test | x) / p(validation | x)` from a cross-fitted L2 logistic regression on Morgan bits (features only), clipped odds. For each test molecule `x_j`, the quantile is the smallest score `s` with `sum_{s_i <= s} w_i >= (1 - a)(sum w + w(x_j))`; if none, `q = infinity` (the trivial full label set).

## 3. Statistics

- **R10 rule** (per dataset, many networks): "improves" if at least 90 % of the networks have a negative paired NLL difference and the 95 % t-interval of the mean lies below zero; an overall statement needs 6 of 8 datasets.
- **Exact sign-flip test** over the eight dataset statistics `z_d`: enumerate the `2^8` sign patterns, `p = P(sum z' <= sum z_obs)`; the smallest attainable one-sided `p` is `1/256 = 0.0039`.
- **Holm correction** over `m` tests: sort `p_(1) <= ... <= p_(m)`, `p_adj(k) = max_{j<=k} min(1, (m - j + 1) p_(j))`.
- **Bootstrap over test molecules** (2,000 resamples, jointly across tasks) for the paired per-molecule NLL difference; `z = estimate / bootstrap SE`.
- **Agreement with published tables:** a row agrees at `k` if `|ours - published| <= k x SD_published` (rows with SD > 0). Under pure noise with three seeds on each side, a perfect reproduction would match about 58 to 65 % of rows within 1 SD and 82 to 87 % within 2 SD (own simulation).
- **Bridge check:** `d = (mean_authors - mean_ours) / pooled SD` with a Welch 95 % interval; consistent if `|d| < 1`, different if `|d| >= 1` and the interval excludes zero.
