"""Robustness check: how much of the average QLIKE gain over GARCH comes from USD/INR alone?
Reads results/dm_vs_garch.csv (written by run_benchmark.py) and writes results/gain_ex_inr.csv."""
import os
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
d = pd.read_csv(os.path.join(OUT, "dm_vs_garch.csv"))
rows = []
for (h, m), g in d.groupby(["h", "model"]):
    rows.append({"h": h, "model": m,
                 "gain_all_9": g["qlike_gain_vs_garch"].mean(),
                 "gain_ex_INR_8": g[g.currency != "INR"]["qlike_gain_vs_garch"].mean(),
                 "gain_INR": g[g.currency == "INR"]["qlike_gain_vs_garch"].iloc[0]})
R = pd.DataFrame(rows)
order = ["roll20", "ewma", "garch", "har", "lgbm_noglobal", "lgbm", "combo"]
R["o"] = R.model.map(order.index); R = R.sort_values(["h", "o"]).drop(columns="o")
R.to_csv(os.path.join(OUT, "gain_ex_inr.csv"), index=False)
print(R.round(4).to_string(index=False))
