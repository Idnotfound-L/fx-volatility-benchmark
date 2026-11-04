"""Leakage-free features and targets. Everything on row t uses information up to and including day t.
Variance proxy = squared daily return (only closes are available). To make a single model comparable across
currencies with very different volatility levels, features/targets are divided by s_t, the trailing 1-year
mean squared return."""
import numpy as np
import pandas as pd


def build_panel(returns):
    panel, rel5, rel22 = {}, {}, {}
    for c, r in returns.items():
        r2 = r ** 2
        s = r2.rolling(252, min_periods=126).mean()
        f = pd.DataFrame(index=r.index)
        f["s"] = s
        f["x1"] = r2 / s
        f["x5"] = r2.rolling(5).mean() / s
        f["x22"] = r2.rolling(22).mean() / s
        f["x63"] = r2.rolling(63).mean() / s
        f["xmax22"] = r2.rolling(22).max() / s
        f["v1"] = r2.shift(-1)                          # next-day variance proxy
        f["v5"] = r2.rolling(5).mean().shift(-5)        # mean variance proxy over next 5 days
        panel[c] = f
        rel5[c], rel22[c] = f["x5"], f["x22"]
    # Global FX-volatility state: cross-sectional mean of relative short-term variance (ffilled, causal).
    union = sorted(set().union(*[f.index for f in panel.values()]))
    g5 = pd.DataFrame(rel5).reindex(union).ffill().mean(axis=1)
    g22 = pd.DataFrame(rel22).reindex(union).ffill().mean(axis=1)
    for c, f in panel.items():
        f["g5"], f["g22"] = g5.reindex(f.index), g22.reindex(f.index)
        f["y1"], f["y5"] = f["v1"] / f["s"], f["v5"] / f["s"]
        panel[c] = f.dropna(subset=["s", "x63", "g22"])  # rows without a full history are unusable
    return panel
