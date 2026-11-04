"""Walk-forward benchmark of volatility forecasters on 9 USD exchange rates (OOS: 2010 -> latest).
Models: 20-day rolling variance, EWMA (RiskMetrics), GARCH(1,1), HAR-style regression, pooled LightGBM
(with / without a cross-currency 'global vol' state), and an equal-weight combination.
Evaluation: QLIKE + MSE on a squared-return proxy; Diebold-Mariano tests vs. GARCH(1,1)."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_returns
from features import build_panel
import forecasters as F
from evaluate import qlike, mse, dm_test

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
OOS_START = sys.argv[1] if len(sys.argv) > 1 else "2010-01-01"
MODELS = ["roll20", "ewma", "garch", "har", "lgbm_noglobal", "lgbm", "combo"]
BENCH = "garch"


def main():
    returns = load_returns()
    panel = build_panel(returns)
    end = max(P.index.max() for P in panel.values())
    blocks = F.refit_dates(OOS_START, end)
    print("OOS", OOS_START, "->", end.date(), "| refits:", len(blocks) - 1, flush=True)

    fc = {m: {1: {}, 5: {}} for m in MODELS}
    for c, P in panel.items():
        r = returns[c]
        g1, g5 = F.garch_forecasts(r, blocks)
        h1, h5 = F.har_forecasts(P, blocks)
        rl, ew = F.roll20(r), F.ewma(r)
        for h, (g, hr) in {1: (g1, h1), 5: (g5, h5)}.items():
            fc["garch"][h][c] = g.reindex(P.index); fc["har"][h][c] = hr
            fc["roll20"][h][c] = rl.reindex(P.index); fc["ewma"][h][c] = ew.reindex(P.index)
        print("garch/har done", c, flush=True)
    for name, cols in [("lgbm_noglobal", F.ML_COLS_NOGLOBAL), ("lgbm", F.ML_COLS)]:
        res, models = F.lgbm_forecasts(panel, blocks, cols, return_models=True)
        for h in (1, 5):
            fc[name][h] = res[h]
        if name == "lgbm":
            final_lgbm = models
        print("lgbm done:", name, flush=True)
    for h in (1, 5):
        for c in panel:
            fc["combo"][h][c] = (fc["garch"][h][c] + fc["har"][h][c] + fc["lgbm"][h][c]) / 3

    # ---- assemble long table ----
    rows = []
    for c, P in panel.items():
        for h in (1, 5):
            df = pd.DataFrame({m: fc[m][h][c] for m in MODELS})
            df["v"] = P[f"v{h}"]; df["currency"] = c; df["h"] = h
            rows.append(df[(df.index >= OOS_START)].dropna())
    L = pd.concat(rows)
    L.to_csv(os.path.join(OUT, "oos_forecasts.csv.gz"))
    print("OOS obs:", len(L) // 2, "per horizon", flush=True)

    summary, dm_rows, cur_rows = [], [], []
    periods = {"all": ("2010-01-01", "2100-01-01"), "2010-2019": ("2010-01-01", "2019-12-31"), "2020+": ("2020-01-01", "2100-01-01")}
    for h in (1, 5):
        Lh = L[L.h == h]
        for pname, (a, b) in periods.items():
            Lp = Lh[(Lh.index >= a) & (Lh.index <= b)]
            if Lp.empty:
                continue
            groups = {c: d for c, d in Lp.groupby("currency")}
            for m in MODELS:
                ql, qb, ms, mb = {}, {}, {}, {}
                better = worse = 0
                for c, d in groups.items():
                    v, fm, fb = d["v"].values, d[m].values, d[BENCH].values
                    ql[c], qb[c] = qlike(v, fm).mean(), qlike(v, fb).mean()
                    ms[c], mb[c] = mse(v, fm).mean(), mse(v, fb).mean()
                    md, t, p = dm_test(qlike(v, fb), qlike(v, fm), h)
                    if pname == "all":
                        dm_rows.append({"h": h, "model": m, "currency": c, "qlike_gain_vs_garch": md, "t": t, "p": p})
                        cur_rows.append({"h": h, "model": m, "currency": c, "qlike": ql[c], "mse": ms[c]})
                    if m != BENCH and p < 0.05:
                        better += md > 0; worse += md < 0
                summary.append({"h": h, "period": pname, "model": m,
                                "mean_qlike": np.mean(list(ql.values())),
                                "qlike_gain_vs_garch": np.mean([qb[c] - ql[c] for c in ql]),
                                "mse_ratio_vs_garch": np.mean([ms[c] / mb[c] for c in ms]),
                                "n_ccy_sig_better": int(better), "n_ccy_sig_worse": int(worse)})
    S = pd.DataFrame(summary); S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.DataFrame(dm_rows).to_csv(os.path.join(OUT, "dm_vs_garch.csv"), index=False)
    pd.DataFrame(cur_rows).to_csv(os.path.join(OUT, "by_currency.csv"), index=False)
    pd.set_option("display.width", 200)
    print(S[S.period == "all"].round(4).to_string(index=False), flush=True)
    print(S[S.period != "all"].round(4).to_string(index=False), flush=True)

    # ---- Figure 1: QLIKE gain vs GARCH heatmap (positive = better than GARCH) ----
    D = pd.DataFrame(dm_rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, h in zip(axes, (1, 5)):
        pv = D[D.h == h].pivot(index="model", columns="currency", values="qlike_gain_vs_garch").loc[MODELS]
        im = ax.imshow(pv.values, cmap="RdYlGn", vmin=-np.abs(pv.values).max(), vmax=np.abs(pv.values).max(), aspect="auto")
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        pp = D[D.h == h].pivot(index="model", columns="currency", values="p").loc[MODELS]
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.values[i, j]:.3f}" + ("*" if pp.values[i, j] < .05 and MODELS[i] != BENCH else ""),
                        ha="center", va="center", fontsize=7)
        ax.set_title(f"QLIKE gain vs GARCH(1,1), h={h}d (green=better, *=DM p<0.05)", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "qlike_gain_heatmap.png"), dpi=150); plt.close()

    # ---- Figure 2: LightGBM feature importance (final model, h=1, gain) ----
    mdl = final_lgbm[1]
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=F.ML_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    (100 * imp / imp.sum()).plot.barh(ax=ax, color="tab:blue"); ax.set_xlabel("share of total gain (%)")
    ax.set_title("LightGBM feature importance (h=1, final refit)"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lgbm_feature_importance.png"), dpi=150); plt.close()
    imp.to_csv(os.path.join(OUT, "lgbm_feature_importance.csv"))

    # ---- Figure 3: INR example, 22-day-smoothed volatility (annualised %) ----
    d = L[(L.currency == "INR") & (L.h == 1)]
    ann = lambda x: np.sqrt(252 * x.rolling(22).mean())
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(d.index, ann(d["v"]), c="k", lw=1, label="realised (squared-return proxy)")
    ax.plot(d.index, ann(d["garch"]), lw=1, label="GARCH(1,1)"); ax.plot(d.index, ann(d["lgbm"]), lw=1, label="LightGBM")
    ax.set_ylabel("annualised vol (%)"); ax.legend(); ax.grid(alpha=.3); ax.set_title("USD/INR: 1-day-ahead volatility forecasts (22d smoothed)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "inr_forecasts.png"), dpi=150); plt.close()


if __name__ == "__main__":
    main()
