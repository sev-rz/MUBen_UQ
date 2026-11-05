"""Unit tests of the conformal and evidence code used in the project (stages/expanded_v16_round1/cp_common.py, stages/expanded_v18_round2/r2_common.py).
Tests that need RDKit or PyTorch are skipped when those packages are not installed."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stages" / "expanded_v16_round1"))
sys.path.insert(0, str(ROOT / "stages" / "expanded_v18_round2"))


@pytest.fixture(scope="module")
def cp():
    pytest.importorskip("rdkit")
    import cp_common
    return cp_common


def test_qhat_rank(cp):
    s = np.arange(1.0, 10.0)                    # n = 9: rank ceil(10 * 0.9) = 9 -> the largest score
    assert cp.qhat(s, 0.1) == 9.0
    assert cp.qhat(s[:8], 0.1) == np.inf        # n = 8: rank 9 > n -> infinite threshold (full set)
    assert cp.qhat(np.arange(1.0, 101.0), 0.1) == 91.0   # n = 100: rank ceil(101 * 0.9) = 91


def test_weighted_quantile_reduces_to_unweighted(cp):
    rng = np.random.default_rng(1)
    for n in (20, 57, 150):                       # (n + 1)(1 - a) is not an integer here
        s = rng.random(n)
        assert cp.wqhat(s, np.ones(n)) == cp.qhat(s)


def test_weighted_quantile_integer_boundary(cp):
    """Known floating-point edge: when (n + 1)(1 - a) is an integer (n = 9, a = 0.1) the cumulative weight 9 x 0.1 can fall
    just below 0.9, so wqhat returns infinity (the conservative full set) where qhat returns the largest score. The reported
    results were produced with this code; the effect is conservative and confined to exact integer boundaries."""
    s = np.arange(1.0, 10.0)
    assert cp.qhat(s) == 9.0
    assert cp.wqhat(s, np.ones(9)) in (9.0, np.inf)


def test_split_conformal_marginal_coverage(cp):
    """Exchangeable calibration and test data: coverage is at least 1 - alpha on average (finite-sample guarantee)."""
    rng = np.random.default_rng(2)
    covs = []
    for _ in range(300):
        cal, test = rng.random(150), rng.random(400)       # scores from the same distribution
        covs.append(float((test <= cp.qhat(cal)).mean()))
    assert np.mean(covs) >= 0.89


def test_coverage_breaks_under_shift(cp):
    """If test scores are stochastically larger than calibration scores, coverage falls below the target."""
    rng = np.random.default_rng(3)
    cal = rng.random(300)
    test = rng.random(5000) ** 0.6                          # larger scores
    assert float((test <= cp.qhat(cal)).mean()) < 0.85


def test_lac_sets_and_regression_interval(cp):
    p = np.array([0.95, 0.5, 0.05])
    y = np.array([1.0, 0.0, 0.0])
    assert np.allclose(cp.cls_scores(p, y), [0.05, 0.5, 0.05])
    cov, size = cp.cls_cov(p, y, 0.1)                       # q = 0.1: only confident labels are kept
    assert cov == pytest.approx(2 / 3) and size == pytest.approx(2 / 3)
    cov, width = cp.reg_cov(np.array([0.0, 0.0]), np.array([1.0, 4.0]), np.array([1.5, 1.5]), 1.0)
    assert cov == 0.5 and width == pytest.approx(2 * (1 + 2) / 2)


def test_evidence_alpha_matches_numerical_optimum():
    torch = pytest.importorskip("torch")
    import r2_common
    lam, th = 50.0, 1.3
    H = torch.tensor([[lam]], dtype=torch.float64)
    theta = torch.tensor([th], dtype=torch.float64)
    a = r2_common.evidence_alpha(torch, H, theta)
    grid = np.logspace(-8, 4, 20001)
    E = 0.5 * np.log(grid) - 0.5 * grid * th ** 2 - 0.5 * np.log(lam + grid)
    best = grid[int(np.argmax(E))]
    assert abs(math.log10(a) - math.log10(best)) < 0.06   # one step of the 121-point grid is 0.1 in log10


def test_signflip_and_verdict():
    pytest.importorskip("scipy")
    import r2_common
    assert r2_common.signflip(np.full(8, -2.0)) == pytest.approx(1 / 256)   # smallest attainable one-sided p
    assert r2_common.signflip(np.zeros(8)) == 1.0
    v, mean, lo, hi, frac, t = r2_common.verdict(-np.ones(12) + np.random.default_rng(0).normal(0, 0.05, 12), "improves", "worsens")
    assert v == "improves" and frac == 1.0
