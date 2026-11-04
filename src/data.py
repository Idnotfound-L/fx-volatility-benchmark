"""Daily FX rates from the Federal Reserve H.10 release, mirrored at github.com/datasets/exchange-rates."""
import os, urllib.request
import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"
PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fed_h10_daily.csv")
CURRENCIES = {"Euro": "EUR", "Japan": "JPY", "United Kingdom": "GBP", "Switzerland": "CHF", "Canada": "CAD",
              "Australia": "AUD", "India": "INR", "Mexico": "MXN", "South Korea": "KRW"}


def load_returns(start="1999-01-04"):
    """Dict {ticker: Series of daily log returns in percent}. Holidays (missing values) are dropped, so a
    return spans consecutive observed fixings."""
    if not os.path.exists(PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        urllib.request.urlretrieve(URL, PATH)
    d = pd.read_csv(PATH, parse_dates=["Date"])
    d.columns = ["date", "country", "rate"]
    d["rate"] = pd.to_numeric(d["rate"], errors="coerce")
    out = {}
    for country, tick in CURRENCIES.items():
        s = d[d.country == country].set_index("date")["rate"].dropna()
        s = s[s.index >= start]
        out[tick] = (100 * np.log(s).diff()).dropna()
    return out
