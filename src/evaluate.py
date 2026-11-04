"""Loss functions and Diebold-Mariano tests."""
import numpy as np
import statsmodels.api as sm
from scipy import stats


def qlike(v, f):
    """QLIKE up to a constant: log f + v/f. Robust to noise in the variance proxy (Patton, 2011)."""
    return np.log(f) + v / f


def mse(v, f):
    return (v - f) ** 2


def dm_test(loss_bench, loss_model, h):
    """H0: equal expected loss. d = L_bench - L_model; mean(d) > 0 => the model is better.
    HAC (Newey-West) variance with max(5, 2h) lags. Returns (mean_d, t, two-sided p)."""
    d = np.asarray(loss_bench) - np.asarray(loss_model)
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": max(5, 2 * h)})
    t = float(res.tvalues[0])
    return float(d.mean()), t, float(2 * (1 - stats.norm.cdf(abs(t))))
