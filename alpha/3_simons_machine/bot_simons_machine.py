"""
🌙 Moon Dev's Simons Machine bot (DRY RUN ONLY)

Jim Simons ~0:12: "look at a lot of data and you try to simulate different predictive schemes
until you get better and better at it."

What it does, every 4h bar close:
  1. pull the last ~62 days of 1h perp bars from Binance's public API (no keys)
  2. build the same weak features the backtest used (machine.py)
  3. score every coin with the trained model in model/ (made by report.py)
  4. smooth the score, long the top third, short the bottom third, dollar neutral
  5. print the book. It does NOT send orders. The order call goes in place_order().

Retrain monthly: rerun prep.py, run_models.py, report.py ("keep on top of the business", ~0:13).
Built by Moon Dev.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
import machine as M

DRY_RUN = True           # Moon Dev: keep this True. This bot is an example, not a live trader.
ACCOUNT_USD = 10_000     # Moon Dev: capital the book is sized on
BINANCE = "https://fapi.binance.com"
HERE = Path(__file__).parent
META = json.load(open(HERE / "model" / "meta.json"))


def get_bars(sym):
    """last 1500 closed 1h bars, stamped at close time (same as prep.py)."""
    r = requests.get(f"{BINANCE}/fapi/v1/klines", params={"symbol": sym, "interval": "1h", "limit": 1500}, timeout=20)
    r.raise_for_status()
    k = pd.DataFrame(r.json()).iloc[:, :11]
    k.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
                 "taker_buy_volume", "taker_buy_quote_volume"]
    k = k.astype({c: float for c in ["open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]})
    k["trades"] = k["trades"].astype(int)
    k["t"] = pd.to_datetime(k["open_time"], unit="ms", utc=True) + pd.Timedelta(hours=1)
    k = k[k["t"] <= pd.Timestamp.now(tz="UTC")]  # drop the bar that is still open
    k["sym"] = sym
    return k[["t", "sym", "open", "high", "low", "close", "quote_volume", "trades", "taker_buy_quote_volume"]]


def get_funding(sym):
    r = requests.get(f"{BINANCE}/fapi/v1/fundingRate", params={"symbol": sym, "limit": 60}, timeout=20)
    r.raise_for_status()
    f = pd.DataFrame(r.json())
    return pd.DataFrame({"ts": pd.to_datetime(f["fundingTime"], unit="ms", utc=True), "rate": f["fundingRate"].astype(float), "sym": sym})


def place_order(sym, usd_change):
    """🌙 Moon Dev: THIS is where the real exchange order call would go (e.g. a reduce/increase market or limit order
    for usd_change dollars of sym). It only prints while DRY_RUN is True."""
    if DRY_RUN:
        print(f"🌙 Moon Dev DRY RUN: would {'BUY ' if usd_change > 0 else 'SELL'} ${abs(usd_change):,.0f} of {sym}")
        return
    raise RuntimeError("Moon Dev: live trading is not built here on purpose")


def main():
    h = META["horizon_hours"]
    print(f"🌙 Moon Dev Simons Machine: model={META['model']} horizon={h}h coins={len(META['universe'])}")
    bars = pd.concat([get_bars(s) for s in META["universe"]], ignore_index=True)
    fu = pd.concat([get_funding(s) for s in META["funding_coins"]], ignore_index=True)
    end = bars["t"].max().floor(f"{h}h")
    W = M.wide_from_bars(bars[bars.t <= end], h, end)
    F, R, ctx, valid, fwd, yz = M.features(W, h, fu)
    syms = list(valid.columns)
    X = np.stack([R[f].reindex(columns=syms).values for f in META["features"]], axis=2)
    rows = valid.index[valid.sum(axis=1) >= 6][-60:]           # last 60 scored bars, for the EMA smoothing
    S = pd.DataFrame(np.nan, index=rows, columns=syms)
    booster = lgb.Booster(model_file=str(HERE / "model" / "simons_model.txt")) if META["model"] == "gbm" else None
    for t in rows:
        i = valid.index.get_loc(t)
        ok = valid.loc[t].values
        x = np.nan_to_num(X[i][ok])
        if META["model"] == "gbm":
            c = np.array([[ctx[n].loc[t] for n in META["context"]]] * ok.sum())
            s = booster.predict(np.hstack([x, c]))
        elif META["model"] == "ridge":
            s = x @ np.array(META["coef"])
        else:
            s = x @ np.array(META["signs"]) / len(META["signs"])
        S.loc[t, np.array(syms)[ok]] = s
    if META["smooth_halflife_bars"] > 0:
        S = S.ewm(halflife=META["smooth_halflife_bars"], ignore_na=True).mean().where(S.notna())
    last = S.iloc[-1].dropna().sort_values()
    k = max(len(last) // 3, 1)
    lev = META["median_leverage_in_backtest"]   # backtest vol-target leverage (10%/yr). Live: compute from your own pnl.
    book = pd.Series(0.0, index=last.index)
    book[last.index[-k:]] = 0.5 / k * lev
    book[last.index[:k]] = -0.5 / k * lev
    print(f"🌙 Moon Dev: bar closed {rows[-1]}  leverage {lev}x  gross ${abs(book).sum() * ACCOUNT_USD:,.0f}")
    print("🌙 Moon Dev LONG book:")
    for s in book[book > 0].index[::-1]:
        print(f"   {s:10s} score {last[s]:+.4f}  ${book[s] * ACCOUNT_USD:,.0f}")
    print("🌙 Moon Dev SHORT book:")
    for s in book[book < 0].index:
        print(f"   {s:10s} score {last[s]:+.4f}  ${book[s] * ACCOUNT_USD:,.0f}")
    for s, w in book.items():
        if w != 0:
            place_order(s, w * ACCOUNT_USD)  # a real bot would trade only the CHANGE vs current position, and skip small changes (band)


if __name__ == "__main__":
    main()
