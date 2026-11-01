"""
Moon Dev: build the bar cache for Jim Simons' 20 day trend test.
Real data only. If a file is missing, this script crashes on purpose.

Sources:
  crypto perps: /Volumes/Moon 26/tick_data/binance/futures_klines_1m/<SYM>/<SYM>_1m_<start>_<end>.parquet
  crypto spot : /Volumes/Moon 26/tick_data/binance/spot_klines_1m/<SYM>/... (longer history for BTC, ETH, BNB, XRP, LINK, DOGE, ZEC)
  commodities / FX / index futures: moon-dev-trading-bots/data/ge_data/*_1d.csv (Yahoo daily, 2000 -> 2024-09)
  SPY: moon-dev-trading-bots/data/yahoo_daily_50/SPY.csv (Yahoo daily, 2010 -> 2026-09)
  TSLA: /Volumes/Moon 26/tick_data/derived/tsla/lit_rth_hist.parquet (Nasdaq lit prints, RTH, 2024-07 -> 2026-06)
"""
import glob, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TICK = "/Volumes/Moon 26/tick_data"
BOTS = "/Users/md/Dropbox/dev/github/moon-dev-trading-bots/data"

PERPS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT", "AVAXUSDT",
         "CRVUSDT", "FILUSDT", "ZECUSDT", "ARBUSDT", "SUIUSDT", "TIAUSDT", "STRKUSDT", "INITUSDT"]
SPOT_LONG = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LINKUSDT", "DOGEUSDT", "ZECUSDT"]

TRAD = {  # Moon Dev: name -> (file, group)
    "CL (crude oil fut)": ("ge_data/CL_F_future_1d.csv", "commodities"),
    "GC (gold fut)": ("ge_data/GC_F_future_1d.csv", "commodities"),
    "SI (silver fut)": ("ge_data/SI_F_future_1d.csv", "commodities"),
    "NG (nat gas fut)": ("ge_data/NG_F_future_1d.csv", "commodities"),
    "EURUSD": ("ge_data/EURUSD_X_forex_1d.csv", "fx"),
    "GBPUSD": ("ge_data/GBPUSD_X_forex_1d.csv", "fx"),
    "USDJPY": ("ge_data/USDJPY_X_forex_1d.csv", "fx"),
    "AUDUSD": ("ge_data/AUDUSD_X_forex_1d.csv", "fx"),
    "ES (S&P fut)": ("ge_data/ES_F_future_1d.csv", "index_futures"),
    "NQ (Nasdaq fut)": ("ge_data/NQ_F_future_1d.csv", "index_futures"),
    "YM (Dow fut)": ("ge_data/YM_F_future_1d.csv", "index_futures"),
}

AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def one_minute_file(kind, sym):
    files = glob.glob(f"{TICK}/binance/{kind}/{sym}/{sym}_1m_*.parquet")
    assert len(files) == 1, f"Moon Dev: expected one all-in-one 1m file for {kind}/{sym}, got {files}"
    return files[0]


def crypto_bars(kind, sym):
    path = one_minute_file(kind, sym)
    m = pd.read_parquet(path, columns=["open_time", "open", "high", "low", "close"]).set_index("open_time").sort_index()
    m = m[~m.index.duplicated()]
    h1 = m.resample("1h").agg(AGG).dropna()
    h4 = m.resample("4h").agg(AGG).dropna()
    d1 = m.resample("1D").agg(AGG).dropna()
    # Moon Dev: drop the last daily bar if it is not a full day
    if m.index[-1] < d1.index[-1] + pd.Timedelta(hours=23, minutes=59):
        d1 = d1.iloc[:-1]
    print(f"🌙 Moon Dev: {kind} {sym}: {len(m):,} 1m bars {m.index[0].date()} -> {m.index[-1].date()} | {len(d1)} days")
    return path, h1, h4, d1


def clean_daily(d, name, fx=False):
    """Moon Dev: Yahoo has a few bad prints (EURUSD +17% on 2008-12-08 then back, AUDUSD 4.57 on 2006-12-25,
    CL -37.63 on 2020-04-20). Drop closes <= 0 (only CL, 2020-04-20). For FX only (major FX pairs never move 8%
    in a day and snap back), drop one-day spike-and-revert bars. Commodities and indexes keep every real move
    (silver squeeze, March 2020). We print every bar we drop."""
    bad = d.index[d["close"] <= 0]
    if len(bad):
        print(f"   Moon Dev: {name}: dropping close <= 0 bars {[str(x.date()) for x in bad]}")
    d = d[d["close"] > 0].copy()
    for _ in range(3 if fx else 0):
        r = d["close"].pct_change()
        spike = (r.abs() > 0.08) & (r.shift(-1) * r < 0) & (r.shift(-1).abs() > 0.8 * r.abs() / (1 + r))
        if not spike.any():
            break
        print(f"   Moon Dev: {name}: dropping spike bars {[str(x.date()) for x in d.index[spike]]}")
        d = d[~spike]
    return d


def main():
    os.makedirs(CACHE, exist_ok=True)
    sources = {}
    for sym in PERPS:
        path, h1, h4, d1 = crypto_bars("futures_klines_1m", sym)
        h1.to_parquet(f"{CACHE}/perp_{sym}_1h.parquet")
        h4.to_parquet(f"{CACHE}/perp_{sym}_4h.parquet")
        d1.to_parquet(f"{CACHE}/perp_{sym}_1d.parquet")
        sources[f"{sym}-perp"] = dict(path=path, group="crypto_perps", start=str(d1.index[0].date()), end=str(d1.index[-1].date()))
    for sym in SPOT_LONG:
        path, h1, h4, d1 = crypto_bars("spot_klines_1m", sym)
        d1.to_parquet(f"{CACHE}/spot_{sym}_1d.parquet")
        sources[f"{sym}-spot"] = dict(path=path, group="crypto_spot_long", start=str(d1.index[0].date()), end=str(d1.index[-1].date()))

    for name, (f, group) in TRAD.items():
        path = f"{BOTS}/{f}"
        d = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")[["open", "high", "low", "close"]].sort_index()
        d = clean_daily(d[~d.index.duplicated()], name, fx=(group == "fx"))
        key = name.split(" ")[0]
        d.to_parquet(f"{CACHE}/trad_{key}_1d.parquet")
        sources[name] = dict(path=path, group=group, start=str(d.index[0].date()), end=str(d.index[-1].date()), file=f"trad_{key}_1d.parquet")
        print(f"🌙 Moon Dev: {name}: {len(d)} days {d.index[0].date()} -> {d.index[-1].date()}")

    path = f"{BOTS}/yahoo_daily_50/SPY.csv"
    d = pd.read_csv(path, parse_dates=["Date"], index_col="Date").rename(columns=str.lower)[["open", "high", "low", "close"]].sort_index()
    d = clean_daily(d, "SPY")
    d.to_parquet(f"{CACHE}/trad_SPY_1d.parquet")
    sources["SPY"] = dict(path=path, group="stocks", start=str(d.index[0].date()), end=str(d.index[-1].date()), file="trad_SPY_1d.parquet")
    print(f"🌙 Moon Dev: SPY: {len(d)} days {d.index[0].date()} -> {d.index[-1].date()}")

    # Moon Dev: TSLA daily OHLC from 139M Nasdaq lit prints, read one row group at a time
    path = f"{TICK}/derived/tsla/lit_rth_hist.parquet"
    pf = pq.ParquetFile(path)
    parts = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["ns", "date", "price"]).to_pandas().sort_values("ns")
        parts.append(t.groupby("date")["price"].agg(["first", "max", "min", "last"]).assign(n=t.groupby("date").size()))
    p = pd.concat(parts)
    # a day can span two row groups: merge in order
    p = p.groupby(level=0).agg({"first": "first", "max": "max", "min": "min", "last": "last", "n": "sum"})
    d = p.rename(columns={"first": "open", "max": "high", "min": "low", "last": "close"})[["open", "high", "low", "close"]]
    d.index = pd.to_datetime(d.index)
    d.to_parquet(f"{CACHE}/trad_TSLA_1d.parquet")
    sources["TSLA"] = dict(path=path, group="stocks", start=str(d.index[0].date()), end=str(d.index[-1].date()), file="trad_TSLA_1d.parquet")
    print(f"🌙 Moon Dev: TSLA: {len(d)} days {d.index[0].date()} -> {d.index[-1].date()}")

    pd.Series({k: v for k, v in sources.items()}).to_json(f"{CACHE}/sources.json", indent=1)
    print("🌙 Moon Dev: bar cache done")


if __name__ == "__main__":
    main()
