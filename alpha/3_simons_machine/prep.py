"""
Moon Dev's Simons machine, step 1: turn Binance 1 minute perp bars into 1 hour bars.
Real data only. If a file is missing, this crashes and names it.
Bars are stamped at their CLOSE time (open_time + 1h), so a feature at time t only uses data up to t.
"""
from pathlib import Path
import pandas as pd

KLINES = Path("/Volumes/Moon 26/tick_data/binance/futures_klines_1m")
FUNDING = Path("/Volumes/Moon 26/tick_data/binance/funding")
LIQ = Path("/Users/md/Dropbox/dev/github/moon-dev-trading-bots/data/moondev_api")
CACHE = Path(__file__).parent / "cache"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LINKUSDT", "ZECUSDT", "DOGEUSDT", "CRVUSDT",
           "SOLUSDT", "AVAXUSDT", "FILUSDT", "ARBUSDT", "SUIUSDT", "TIAUSDT", "STRKUSDT", "INITUSDT"]


def one_symbol(sym):
    files = sorted((KLINES / sym).glob(f"{sym}_1m_*_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Moon Dev: no all-in-one 1m file in {KLINES / sym}")
    df = pd.read_parquet(files[-1])
    print(f"Moon Dev: {sym} {len(df):,} 1m bars from {files[-1].name}")
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated()]
    h = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                               "quote_volume": "sum", "trades": "sum", "taker_buy_quote_volume": "sum"})
    h["n_min"] = df["close"].resample("1h").count()
    h.index = h.index + pd.Timedelta(hours=1)  # stamp at bar CLOSE
    h.index.name = "t"
    h = h[h["n_min"] >= 30]  # drop hours that are mostly holes (Binance outages)
    h["sym"] = sym
    return h.reset_index()


def funding():
    out = []
    for f in sorted(FUNDING.glob("*_funding.csv")):
        d = pd.read_csv(f)
        d["ts"] = pd.to_datetime(d["ts"], format="ISO8601", utc=True)
        d["sym"] = f.name.split("_")[0]
        out.append(d)
    if not out:
        raise FileNotFoundError(f"Moon Dev: no funding files in {FUNDING}")
    return pd.concat(out)


def liqs():
    out = []
    for c in ["BTC", "ETH", "SOL", "XRP"]:
        f = LIQ / f"liq_clean_{c}_5m.csv"  # the ONLY allowed liq files (data_sources.md bans liq_data_*_ohlcv)
        d = pd.read_csv(f, parse_dates=["datetime"])
        d["t"] = d["datetime"].dt.tz_localize("UTC") + pd.Timedelta(minutes=5)  # stamp at bar close
        d = d.set_index("t")[["forced_sell_usd", "forced_buy_usd"]].resample("1h", label="right", closed="right").sum()
        d["sym"] = c + "USDT"
        out.append(d.reset_index())
    return pd.concat(out)


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    bars = pd.concat([one_symbol(s) for s in SYMBOLS], ignore_index=True)
    bars.to_parquet(CACHE / "bars_1h.parquet")
    funding().to_parquet(CACHE / "funding.parquet")
    liqs().to_parquet(CACHE / "liq_1h.parquet")
    print(f"Moon Dev: saved {len(bars):,} hourly bars, {bars.t.min()} -> {bars.t.max()}")
