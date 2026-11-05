"""Toy demonstration of the two ideas of this project, on synthetic data, numpy only, a few seconds on a laptop CPU.

A binary classifier is trained on 'familiar chemistry' (cluster A). Test molecules come from a 'new scaffold' (cluster B) where the labels are noisier and the features lie
further from the training data, so the network is wrong more often than it says (distribution shift).

 1. last-layer Laplace: analytic Hessian of the logistic last layer, probit predictive, prior strength alpha chosen on validation NLL (as in stage expanded_v6_laplace),
    compared with the plain network and a one-parameter temperature scaling (as in expanded_v15_scale_widegrid);
 2. split conformal prediction (LAC score, alpha 0.1): coverage when the calibration set looks like the training data, versus calibration on a few labelled molecules
    of the new scaffold (as in expanded_v16_round1).

Run:  python toy/toy_laplace_conformal.py
"""
import math
import numpy as np

rng = np.random.default_rng(0)
D, N_TR, N_VA, N_TE = 20, 1500, 400, 600


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make(n, shifted, w_true):
    """Features are a random projection of latent chemistry; shifted molecules have a different mean and noisier labels."""
    x = rng.normal(size=(n, D)) + (1.6 * np.ones(D) / math.sqrt(D) * (3.0 if shifted else 0.0))
    logit = x @ w_true * (0.35 if shifted else 1.0)            # weaker signal on the new scaffold -> more errors
    y = (rng.random(n) < sigmoid(logit)).astype(float)
    return x, y


w_true = rng.normal(size=D) / math.sqrt(D) * 2.0
Xtr, ytr = make(N_TR, False, w_true)
Xva, yva = make(N_VA, False, w_true)                           # validation = calibration set, same chemistry as training
Xte, yte = make(N_TE, True, w_true)                            # test = new scaffold


def phi(x):                                                    # last-layer features with a bias column
    return np.hstack([x, np.ones((len(x), 1))])


def fit_logistic(P, y, l2=1e-6, iters=60):
    theta = np.zeros(P.shape[1])
    for _ in range(iters):                                     # Newton steps
        p = sigmoid(P @ theta)
        H = P.T @ (P * (p * (1 - p))[:, None]) + l2 * np.eye(P.shape[1])
        theta -= np.linalg.solve(H, P.T @ (p - y) + l2 * theta)
    return theta


def nll(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


Ptr, Pva, Pte = phi(Xtr), phi(Xva), phi(Xte)
theta = fit_logistic(Ptr, ytr, l2=0.0)                         # an overfit 'trained network' (no prior)
H = Ptr.T @ (Ptr * (sigmoid(Ptr @ theta) * (1 - sigmoid(Ptr @ theta)))[:, None])   # data Hessian of the last layer


def laplace_probs(P, alpha):
    cov = np.linalg.inv(H + alpha * np.eye(len(theta)))        # posterior covariance N(theta, (H + alpha I)^-1)
    v = np.einsum("ij,jk,ik->i", P, cov, P)
    return sigmoid((P @ theta) / np.sqrt(1 + math.pi * v / 8))  # probit approximation


grid = [10.0 ** a for a in range(-4, 5)]
alpha = min(grid, key=lambda a: nll(yva, laplace_probs(Pva, a)))   # prior strength on validation NLL only
temps = np.exp(np.linspace(math.log(0.05), math.log(20), 81))
T = min(temps, key=lambda t: nll(yva, sigmoid((Pva @ theta) / t)))  # one-parameter recalibration on validation only

p_det, p_lap, p_tmp = sigmoid(Pte @ theta), laplace_probs(Pte, alpha), sigmoid((Pte @ theta) / T)
print("TEST on the new scaffold (lower NLL is better)")
print(f"  plain network        NLL {nll(yte, p_det):.3f}")
print(f"  last-layer Laplace   NLL {nll(yte, p_lap):.3f}   (alpha {alpha:g}, chosen on validation)")
print(f"  temperature scaling  NLL {nll(yte, p_tmp):.3f}   (T {T:.2f}, chosen on validation)")


def qhat(scores, a=0.1):
    s = np.sort(scores)
    k = math.ceil((len(s) + 1) * (1 - a))
    return np.inf if k > len(s) else s[k - 1]


def lac_scores(p, y):
    return np.where(y == 1, 1 - p, p)                          # LAC: 1 - probability of the true label


def coverage(p, y, q):
    in1, in0 = (1 - p) <= q, p <= q
    return float(np.where(y == 1, in1, in0).mean()), float((in1.astype(int) + in0).mean())


pva = sigmoid(Pva @ theta)
q_val = qhat(lac_scores(pva, yva))
cov_val, size_val = coverage(p_det, yte, q_val)
print("\nCONFORMAL (target coverage 0.90)")
Xva2, yva2 = make(N_TE, False, w_true)                         # sanity: same-chemistry test set
cov_iid, _ = coverage(sigmoid(phi(Xva2) @ theta), yva2, q_val)
print(f"  calibrated on validation, test = same chemistry : coverage {cov_iid:.3f}")
print(f"  calibrated on validation, test = new scaffold    : coverage {cov_val:.3f}  (average set size {size_val:.2f})   <- the promise breaks")
for k in (20, 50):
    idx = rng.permutation(N_TE)
    lab, ev = idx[:k], idx[k:]
    q_t = qhat(lac_scores(p_det[lab], yte[lab]))               # 'target-only' arm of expanded_v16_round1
    c, s = coverage(p_det[ev], yte[ev], q_t)
    print(f"  calibrated on {k:>2d} labelled new-scaffold molecules  : coverage {c:.3f}  (average set size {s:.2f})")
