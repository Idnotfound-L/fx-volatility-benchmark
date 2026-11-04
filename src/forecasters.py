"""Variance forecasters. All are refit on a semi-annual schedule using only data that was fully observed
(targets included) CUTOFF_DAYS before the refit date, then applied unchanged until the next refit."""
import warnings
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from arch import arch_model
import lightgbm as lgb

warnings.filterwarnings("ignore")
CUTOFF_DAYS = 10
HAR_COLS = ["x1", "x5", "x22"]
ML_COLS_NOGLOBAL = ["x1", "x5", "x22", "x63", "xmax22"]
ML_COLS = ML_COLS_NOGLOBAL + ["g5", "g22"]
MIN_REL = 1e-3  # floor for tree models (Poisson outputs are already > 0)
HAR_FLOOR = 0.1  # OLS can output ~0 or negative variances; truncate at 10% of the trailing-year mean variance


def refit_dates(start, end):
    return list(pd.date_range(start, end, freq="6MS")) + [pd.Timestamp(end) + pd.Timedelta(days=1)]


# ---------------- simple benchmarks (no fitting) ----------------
def roll20(r):
    return (r ** 2).rolling(20).mean()


def ewma(r, lam=0.94):
    return (r ** 2).ewm(alpha=1 - lam, adjust=False).mean()


# ---------------- GARCH(1,1) ----------------
def garch_forecasts(r, blocks):
    """Returns (f1, f5): variance forecasts for the next day and the mean of the next 5 days."""
    f1 = pd.Series(np.nan, index=r.index); f5 = f1.copy()
    r2 = r.values ** 2
    for T0, T1 in zip(blocks[:-1], blocks[1:]):
        train = r[r.index <= T0 - pd.Timedelta(days=CUTOFF_DAYS)]
        if len(train) < 1000:
            continue
        res = arch_model(train.values, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
        w, a, b = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
        h0 = train.values.var()
        u = lfilter([1.0], [1.0, -b], w + a * r2, zi=[b * h0])[0]       # u_t = sigma^2_{t+1}
        pers = min(a + b, 0.9999); V = w / (1 - pers)
        m = (r.index >= T0) & (r.index < T1)
        f1[m] = u[m]
        f5[m] = V + (u[m] - V) * (1 - pers ** 5) / (5 * (1 - pers))
    return f1, f5


# ---------------- HAR-style regression on squared returns (Corsi, 2009) ----------------
def har_forecasts(P, blocks):
    out = {1: pd.Series(np.nan, index=P.index), 5: pd.Series(np.nan, index=P.index)}
    for T0, T1 in zip(blocks[:-1], blocks[1:]):
        tr = P[P.index <= T0 - pd.Timedelta(days=CUTOFF_DAYS)]
        m = (P.index >= T0) & (P.index < T1)
        for h in (1, 5):
            d = tr.dropna(subset=[f"y{h}"] + HAR_COLS)
            if len(d) < 500:
                continue
            X = np.c_[np.ones(len(d)), d[HAR_COLS].values]
            beta = np.linalg.lstsq(X, d[f"y{h}"].values, rcond=None)[0]
            Xp = np.c_[np.ones(m.sum()), P.loc[m, HAR_COLS].values]
            out[h][m] = np.maximum(Xp @ beta, HAR_FLOOR) * P.loc[m, "s"].values
    return out[1], out[5]


# ---------------- pooled gradient boosting ----------------
LGB_PARAMS = dict(objective="poisson", n_estimators=150, learning_rate=0.05, num_leaves=8, min_child_samples=300,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=0,
                  n_jobs=1, verbose=-1)


def lgbm_forecasts(panel, blocks, cols, return_models=False):
    """One model pooled across all currencies (features are scale-free), refit every block."""
    out = {h: {c: pd.Series(np.nan, index=P.index) for c, P in panel.items()} for h in (1, 5)}
    last_models = {}
    for T0, T1 in zip(blocks[:-1], blocks[1:]):
        cut = T0 - pd.Timedelta(days=CUTOFF_DAYS)
        for h in (1, 5):
            tr = pd.concat([P[P.index <= cut].dropna(subset=[f"y{h}"] + cols) for P in panel.values()])
            if len(tr) < 5000:
                continue
            mdl = lgb.LGBMRegressor(**LGB_PARAMS).fit(tr[cols], tr[f"y{h}"])
            last_models[h] = mdl
            for c, P in panel.items():
                m = (P.index >= T0) & (P.index < T1)
                if m.any():
                    out[h][c][m] = np.maximum(mdl.predict(P.loc[m, cols]), MIN_REL) * P.loc[m, "s"].values
    return (out, last_models) if return_models else out
