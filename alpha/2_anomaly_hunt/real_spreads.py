"""
Moon Dev: real quoted spread on the Binance perp from the few bookTicker days we have.
Used to anchor the spread proxy for claim (d) and the "is the anomaly smaller than the spread" check.
"""
import os, glob, json, zipfile
import numpy as np
import pandas as pd

BT = "/Volumes/Moon 26/tick_data/binance/futures_bookTicker"
HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(BT, "*.zip")))
if not files:
    raise FileNotFoundError(f"Moon Dev: no bookTicker zips in {BT}")
out = []
for f in files:
    with zipfile.ZipFile(f) as z:
        with z.open(z.namelist()[0]) as fh:
            d = pd.read_csv(fh, usecols=["best_bid_price", "best_ask_price", "transaction_time"], engine="pyarrow")
    d["ts"] = pd.to_datetime(d["transaction_time"], unit="ms")
    s = (d["best_ask_price"] - d["best_bid_price"]) / ((d["best_ask_price"] + d["best_bid_price"]) / 2) * 1e4
    # sample the book once a second (last quote in each second) so busy moments do not dominate
    sec = s.groupby(d["ts"].dt.floor("1s")).last()
    name = os.path.basename(f).replace(".zip", "")
    out.append({"file": name, "median_spread_bp": round(float(sec.median()), 4), "mean_spread_bp": round(float(sec.mean()), 4),
                "seconds": int(len(sec))})
    print("Moon Dev:", out[-1])
json.dump(out, open(os.path.join(HERE, "cache", "real_spreads.json"), "w"), indent=1)
