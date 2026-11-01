"""
Moon Dev anomaly hunt (package 2): Jim Simons, TED ~0:13.
"any one anomaly might be a random thing. However, if you have enough data, you can tell that it's not."

Runs the 224 pre-declared short-term tests in notes.md on Moon Dev's real data.
Each test = a signal (+1 / -1 / 0) at a bar close and the forward return in bp.
Stats per test are cached to cache/tests.pkl. Events for |IS t| >= 2 go to cache/events/<id>.npz.
Built by Moon Dev.
"""
import os, sys, glob, pickle, zlib
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
EVDIR = os.path.join(CACHE, "events")
os.makedirs(EVDIR, exist_ok=True)

KL = "/Volumes/Moon 26/tick_data/binance/futures_klines_1m"
FUND = "/Volumes/Moon 26/tick_data/binance/funding"
LIQ = "/Users/md/Dropbox/dev/github/moon-dev-trading-bots/data/moondev_api"
TSLA = "/Volumes/Moon 26/tick_data/derived/tsla"

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "SOLUSDT",
         "LINKUSDT", "AVAXUSDT", "ZECUSDT", "CRVUSDT", "FILUSDT"]
CRYPTO_SPLIT = pd.Timestamp("2024-01-01")
N_PERM = 300
RNG_SEED = 7


def need(path):
    # Moon Dev: real data only. Missing file = loud failure with its name.
    if not os.path.exists(path):
        raise FileNotFoundError(f"Moon Dev: missing data file {path}")
    return path


def sgn(x):
    return np.sign(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0))


# ---------------------------------------------------------------- core test evaluation
def evaluate(meta, ts, sig, fwd, split, hold_min):
    """ts: datetime64 array (naive, UTC or ET for TSLA). sig: +1/-1/0. fwd: forward return in bp."""
    ts = pd.DatetimeIndex(ts).tz_localize(None) if getattr(ts, "tz", None) is not None else pd.DatetimeIndex(ts)
    sig = np.asarray(sig, dtype=np.float64)
    fwd = np.asarray(fwd, dtype=np.float64)
    ok = (sig != 0) & np.isfinite(fwd)
    ev_ts, ev_raw, ev_sig = ts[ok], (sig * fwd)[ok], sig[ok]
    is_ev = ev_ts < split
    if is_ev.sum() < 30 or (~is_ev).sum() < 30:
        print(f"Moon Dev: skip {meta['id']} (too few events: IS {is_ev.sum()}, OOS {(~is_ev).sum()})")
        return None
    d = 1.0 if ev_raw[is_ev].mean() >= 0 else -1.0
    pnl = d * ev_raw  # bp per trade in the IS-chosen direction
    days = ev_ts.normalize()
    daily = pd.Series(pnl).groupby(days).sum()
    dcount = pd.Series(1, index=range(len(pnl))).groupby(days).sum()
    dIS, dOOS = daily[daily.index < split], daily[daily.index >= split]

    def tstat(x):
        return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0

    # IS cut into 4 equal time chunks (Jim's "persisted for a sufficiently long time")
    t0, t1 = dIS.index.min(), dIS.index.max()
    edges = pd.date_range(t0, t1 + pd.Timedelta(days=1), periods=5)
    chunk_t = [tstat(dIS[(dIS.index >= a) & (dIS.index < b)]) for a, b in zip(edges[:-1], edges[1:])]

    # quarterly effect in bp per trade (for decay)
    q = pd.Series(pnl, index=ev_ts).groupby(ev_ts.to_period("Q"))
    quarters = pd.DataFrame({"bp": q.mean(), "n": q.size()})
    years = pd.Series(pnl, index=ev_ts).groupby(ev_ts.year)
    yearly = pd.DataFrame({"bp": years.mean(), "n": years.size(), "t": daily.groupby(daily.index.year).apply(tstat)})

    t_is = tstat(dIS)
    rec = dict(meta)
    rec.update(dict(
        direction="follow" if d > 0 else "fade", n_is=int(is_ev.sum()), n_oos=int((~is_ev).sum()),
        days_is=len(dIS), days_oos=len(dOOS),
        is_bp=float(pnl[is_ev].mean()), oos_bp=float(pnl[~is_ev].mean()),
        is_t=t_is, oos_t=tstat(dOOS),
        is_sr_daily=float(dIS.mean() / dIS.std()) if dIS.std() > 0 else 0.0,
        is_skew=float(dIS.skew()), is_kurt=float(dIS.kurt() + 3.0),
        oos_sr_daily=float(dOOS.mean() / dOOS.std()) if dOOS.std() > 0 else 0.0,
        trades_per_day_is=float(dcount[dcount.index < split].mean()),
        trades_per_day_oos=float(dcount[dcount.index >= split].mean()),
        chunk_t=chunk_t, quarters=quarters, yearly=yearly, hold_min=hold_min,
        split=str(split.date()), first=str(ev_ts.min()), last=str(ev_ts.max()),
        daily=daily.astype("float32"), daily_n=dcount.astype("int32"),
    ))

    # circular shift permutation on the IS part (keeps each series' own autocorrelation)
    rec["perm_p"] = np.nan
    if abs(t_is) >= 2:
        isg = ts < split
        s_is, f_is = sig[isg], np.nan_to_num(fwd[isg], nan=0.0)
        f_ok = np.isfinite(fwd[isg])
        obs = abs(ev_raw[is_ev].mean())
        rng = np.random.default_rng(RNG_SEED + zlib.crc32(meta["id"].encode()))
        n = len(s_is)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            k = int(rng.integers(n // 10, n - n // 10))
            sr = np.roll(s_is, k)
            m = (sr != 0) & f_ok
            null[i] = abs((sr[m] * f_is[m]).mean())
        rec["perm_p"] = float((1 + (null >= obs).sum()) / (1 + N_PERM))
        np.savez_compressed(os.path.join(EVDIR, meta["id"] + ".npz"),
                            ts=ev_ts.values.astype("datetime64[ns]").astype(np.int64),
                            pnl=pnl.astype(np.float32), side=(d * ev_sig).astype(np.int8),
                            fwd=fwd[ok].astype(np.float32))
    print(f"Moon Dev: {meta['id']:<34} IS t {t_is:+6.2f}  OOS t {rec['oos_t']:+6.2f}  "
          f"IS {rec['is_bp']:+7.2f} bp  OOS {rec['oos_bp']:+7.2f} bp  n {len(pnl)}")
    return rec


def M(id_, market, symbol, tf, scheme, family, horizon):
    return dict(id=id_, market=market, symbol=symbol, timeframe=tf, scheme=scheme, family=family, horizon=horizon)


# ---------------------------------------------------------------- crypto loaders
def load_1m(sym):
    f = glob.glob(os.path.join(KL, sym, f"{sym}_1m_*_*.parquet"))
    if not f:
        raise FileNotFoundError(f"Moon Dev: no all-in-one 1m kline file in {os.path.join(KL, sym)}")
    df = pd.read_parquet(need(f[0])).drop_duplicates("open_time").set_index("open_time").sort_index()
    df.index = df.index.tz_convert("UTC")
    grid = pd.date_range(df.index[0], df.index[-1], freq="1min")
    return df.reindex(grid)


def bars(df, rule):
    agg = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum", "taker_buy_volume": "sum", "trades": "sum", "quote_volume": "sum"})
    empty = df["close"].resample(rule).count() == 0
    agg.loc[empty, :] = np.nan
    return agg


def lr(c):
    return np.log(c).diff() * 1e4


def fwdk(c, k):
    return (np.log(c.shift(-k)) - np.log(c)) * 1e4


def crypto_coin(sym):
    """Families A-D, H-M (and E-G for BTC/ETH) for one coin. Also liquidity stats for claim (d)."""
    out, s = [], sym.replace("USDT", "")
    df = load_1m(sym)
    tfs = {"1m": df}
    for rule, name in [("5min", "5m"), ("15min", "15m"), ("1h", "1h")]:
        tfs[name] = bars(df, rule)
        tfs[name].to_parquet(os.path.join(CACHE, f"{sym}_{name}.parquet"))
    hold = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
    fam = {"1m": "A", "5m": "B", "15m": "C", "1h": "D"}
    for name, b in tfs.items():
        r = lr(b["close"])
        out.append(evaluate(M(f"{fam[name]}_{s}_{name}_lastbar", "crypto", s, name, "last bar return -> next bar",
                              fam[name], name), b.index, sgn(r), r.shift(-1), CRYPTO_SPLIT, hold[name]))
    b5, b1h = tfs["5m"], tfs["1h"]
    r5 = lr(b5["close"])
    spike = b5["volume"] / b5["volume"].rolling(288, min_periods=200).median().shift(1) > 4
    sig = sgn(r5) * spike.values
    out.append(evaluate(M(f"H_{s}_5m_volspike", "crypto", s, "5m", "volume spike 4x -> next 5m", "H", "5m"),
                        b5.index, sig, r5.shift(-1), CRYPTO_SPLIT, 5))
    out.append(evaluate(M(f"I_{s}_5m_volspike_1h", "crypto", s, "5m", "volume spike 4x -> next 1h", "I", "1h"),
                        b5.index, sig, fwdk(b5["close"], 12), CRYPTO_SPLIT, 60))
    for b, w, name, fm, hm in [(b5, 288, "5m", "J", 5), (b1h, 168, "1h", "K", 60)]:
        imb = 2 * b["taker_buy_volume"] / b["volume"] - 1
        z = (imb - imb.rolling(w, min_periods=w // 2).mean().shift(1)) / imb.rolling(w, min_periods=w // 2).std().shift(1)
        sig = sgn(z) * (z.abs() > 2).values
        out.append(evaluate(M(f"{fm}_{s}_{name}_takerimb", "crypto", s, name, "taker buy imbalance |z|>2 -> next bar",
                              fm, name), b.index, sig, lr(b["close"]).shift(-1), CRYPTO_SPLIT, hm))
    if sym != "BTCUSDT":
        for name, fm, hm in [("1m", "L", 1), ("5m", "M", 5)]:
            btc = pd.read_parquet(os.path.join(CACHE, f"BTC_close_{name}.parquet"))["close"]
            b = tfs[name]
            rb = lr(btc).reindex(b.index)
            ra = lr(b["close"])
            out.append(evaluate(M(f"{fm}_{s}_{name}_btclead", "crypto", s, name, "BTC last bar sign -> alt next bar",
                                  fm, name), b.index, sgn(rb), ra.shift(-1), CRYPTO_SPLIT, hm))
    if sym in ("BTCUSDT", "ETHUSDT"):
        out += seasonality(s, df, b1h)
    # claim (d): liquidity and spread proxy per year (Abdi-Ranaldo 2017 close-high-low estimator on 1m bars)
    c, eta = np.log(df["close"]), (np.log(df["high"]) + np.log(df["low"])) / 2
    s2 = 4 * (c - eta) * (c - eta.shift(-1))
    liq = pd.DataFrame({"s2": s2, "trades": df["trades"], "qv": df["quote_volume"]})
    liq["month"] = liq.index.tz_localize(None).to_period("M")
    mon = liq.groupby("month").agg(s2=("s2", "mean"), trades=("trades", "median"), qv=("qv", "sum"))
    mon["spread_bp"] = np.sqrt(mon["s2"].clip(lower=0)) * 1e4
    mon["year"] = mon.index.year
    ly = mon.groupby("year").agg(spread_bp=("spread_bp", "mean"), med_trades_per_min=("trades", "median"),
                                 quote_vol_bn_per_month=("qv", "median"))
    ly["quote_vol_bn_per_month"] /= 1e9
    return [o for o in out if o is not None], (s, ly)


def seasonality(s, df, b1h):
    out = []
    r1h = lr(b1h["close"])
    nxt_hour = (b1h.index.hour + 1) % 24
    for h in range(24):
        out.append(evaluate(M(f"E_{s}_hour{h:02d}", "crypto", s, "1h", f"long the {h:02d}:00 UTC hour", "E", "1h"),
                            b1h.index, (nxt_hour == h).astype(float), r1h.shift(-1), CRYPTO_SPLIT, 60))
    cD = df["close"].resample("1D").last()
    rD = lr(cD)
    nxt_dow = (cD.index.dayofweek + 1) % 7
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for k in range(7):
        out.append(evaluate(M(f"F_{s}_dow_{names[k]}", "crypto", s, "1d", f"long {names[k]} (UTC day)", "F", "1d"),
                            cD.index, (nxt_dow == k).astype(float), rD.shift(-1), CRYPTO_SPLIT, 1440))
    # US cash session in New York time. The 1m bar at 09:29 closes at 9:30, the 15:59 bar closes at 16:00.
    et = df["close"].copy()
    et.index = et.index.tz_convert("America/New_York")
    wk = et.index.dayofweek < 5
    o = et[wk & (et.index.hour == 9) & (et.index.minute == 29)]
    c = et[wk & (et.index.hour == 15) & (et.index.minute == 59)]
    o.index, c.index = o.index.normalize(), c.index.normalize()
    j = pd.DataFrame({"o": o, "c": c}).dropna()
    us = (np.log(j["c"]) - np.log(j["o"])) * 1e4
    night = (np.log(j["o"].shift(-1)) - np.log(j["c"])) * 1e4  # 16:00 -> next weekday 9:30
    ts_us = j.index.tz_localize(None) + pd.Timedelta(hours=9, minutes=30)
    out.append(evaluate(M(f"G_{s}_us_session", "crypto", s, "session", "long 9:30-16:00 New York", "G", "6.5h"),
                        ts_us, np.ones(len(j)), us.values, CRYPTO_SPLIT, 390))
    out.append(evaluate(M(f"G_{s}_outside_us", "crypto", s, "session", "long 16:00 -> next 9:30 New York", "G", "overnight"),
                        j.index.tz_localize(None) + pd.Timedelta(hours=16), np.ones(len(j)), night.values, CRYPTO_SPLIT, 1050))
    fri = cD[cD.index.dayofweek == 4]  # close of Friday UTC day
    sun = cD[cD.index.dayofweek == 6]
    fri.index = fri.index + pd.Timedelta(days=2)
    w = pd.DataFrame({"f": fri, "s": sun}).dropna()
    wr = (np.log(w["s"]) - np.log(w["f"])) * 1e4
    out.append(evaluate(M(f"G_{s}_weekend", "crypto", s, "session", "long Sat+Sun (UTC)", "G", "2d"),
                        w.index.tz_localize(None) - pd.Timedelta(days=2), np.ones(len(w)), wr.values, CRYPTO_SPLIT, 2880))
    return out


# ---------------------------------------------------------------- funding + liquidations
def funding_tests():
    out = []
    for s in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
        f = pd.read_csv(need(os.path.join(FUND, f"{s}USDT_funding.csv")))
        f["ts"] = pd.to_datetime(f["ts"], utc=True, format="mixed").dt.floor("min")
        f = f.drop_duplicates("ts").set_index("ts").sort_index()
        px = load_1m(f"{s}USDT")["close"]
        entry = px.reindex(f.index - pd.Timedelta(minutes=1)).values
        exit_ = px.reindex(f.index + pd.Timedelta(hours=8) - pd.Timedelta(minutes=1)).values
        fwd = (np.log(exit_) - np.log(entry)) * 1e4
        rate = f["rate"]
        hi = rate > rate.rolling(90, min_periods=60).quantile(0.9).shift(1)
        lo = rate < rate.rolling(90, min_periods=60).quantile(0.1).shift(1)
        sig = hi.astype(float).values - lo.astype(float).values
        days = f.index.normalize().unique()
        split = days[int(len(days) * 0.6)].tz_localize(None)
        idx = f.index.tz_localize(None)
        out.append(evaluate(M(f"N_{s}_funding_extreme", "crypto", s, "8h", "funding above 90th / below 10th pct -> next 8h",
                              "N", "8h"), idx, sig, fwd, split, 480))
        out.append(evaluate(M(f"O_{s}_funding_change", "crypto", s, "8h", "funding change sign -> next 8h", "O", "8h"),
                            idx, sgn(rate.diff()), fwd, split, 480))
    return out


def liq_tests():
    out = []
    for s in ["BTC", "ETH", "SOL", "XRP"]:
        d = pd.read_csv(need(os.path.join(LIQ, f"liq_clean_{s}_5m.csv")), parse_dates=["datetime"]).set_index("datetime")
        thr = d["liq_usd"].rolling(2016, min_periods=1000).quantile(0.99).shift(1)
        spike = (d["liq_usd"] > thr) & (d["liq_usd"] > 0) & (~d["coinm_suspect"])
        sig = sgn(d["liq_imbalance"]) * spike.values
        r = lr(d["close"])
        days = d.index.normalize().unique()
        split = days[int(len(days) * 0.6)]
        out.append(evaluate(M(f"P_{s}_liqspike_5m", "crypto", s, "5m", "liq spike (99th pct) side -> next 5m", "P", "5m"),
                            d.index, sig, r.shift(-1), split, 5))
        out.append(evaluate(M(f"Q_{s}_liqspike_1h", "crypto", s, "5m", "liq spike (99th pct) side -> next 1h", "Q", "1h"),
                            d.index, sig, fwdk(d["close"], 12), split, 60))
    return out


# ---------------------------------------------------------------- TSLA
def tsla_1m():
    path = os.path.join(CACHE, "tsla_1m.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(need(os.path.join(TSLA, "lit_rth_hist.parquet")))
    parts = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["date", "tod", "price"]).to_pandas()
        t["minute"] = (t["tod"] // 60).astype(np.int16)
        g = t.groupby(["date", "minute"], sort=True)["price"]
        parts.append(pd.DataFrame({"open": g.first(), "high": g.max(), "low": g.min(), "close": g.last(), "n": g.size()}))
        print(f"Moon Dev: TSLA row group {i + 1}/{pf.num_row_groups}")
    a = pd.concat(parts)
    g = a.groupby(level=[0, 1], sort=True)
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "n": g["n"].sum()}).reset_index()
    b = b[(b["minute"] >= 570) & (b["minute"] < 960)]
    # full 390 minute grid per day, carry the last price over empty minutes
    full = pd.MultiIndex.from_product([b["date"].unique(), range(570, 960)], names=["date", "minute"])
    b = b.set_index(["date", "minute"]).reindex(full)
    b["n"] = b["n"].fillna(0)
    b["close"] = b.groupby(level=0)["close"].ffill()
    b = b.reset_index()
    b["ts"] = pd.to_datetime(b["date"]) + pd.to_timedelta(b["minute"].astype(int), unit="min")
    b.to_parquet(path)
    return b


def tsla_tests():
    out = []
    b = tsla_1m()
    days = np.sort(b["date"].unique())
    split = pd.Timestamp(days[int(len(days) * 0.6)])
    cost_price = float(b["close"].median())
    for k, fm in [(1, "R"), (5, "R"), (15, "R")]:
        bk = b[(b["minute"] - 570) % k == k - 1].copy()  # close of each k minute block
        bk["r"] = bk.groupby("date")["close"].transform(lambda x: np.log(x).diff() * 1e4)
        bk["f"] = bk.groupby("date")["r"].shift(-1)
        out.append(evaluate(M(f"R_TSLA_{k}m_lastbar", "tsla", "TSLA", f"{k}m", "last bar return -> next bar", "R", f"{k}m"),
                            bk["ts"].values, sgn(bk["r"]), bk["f"].values, split, k))
    piv = b.pivot(index="date", columns="minute", values="close")
    dopen = b[b["minute"] == 570].set_index("date")["open"].reindex(piv.index)
    dclose = piv[959]
    lp = np.log
    for j in range(13):
        start = 570 + 30 * j
        p0 = dopen if j == 0 else piv[start - 1]
        r = (lp(piv[start + 29]) - lp(p0)) * 1e4
        hh, mm = divmod(start, 60)
        out.append(evaluate(M(f"S_TSLA_{hh:02d}{mm:02d}", "tsla", "TSLA", "30m", f"long {hh:02d}:{mm:02d}-{hh:02d}:{mm + 29:02d} block",
                              "S", "30m"), pd.to_datetime(piv.index) + pd.Timedelta(minutes=start), np.ones(len(r)), r.values,
                            split, 30))
    oc = (lp(dclose) - lp(dopen)) * 1e4
    dts = pd.to_datetime(piv.index)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for k in range(5):
        out.append(evaluate(M(f"T_TSLA_dow_{names[k]}", "tsla", "TSLA", "1d", f"long {names[k]} open -> close", "T", "6.5h"),
                            dts + pd.Timedelta(minutes=570), (dts.dayofweek == k).astype(float), oc.values, split, 390))
    first30 = (lp(piv[599]) - lp(dopen)) * 1e4
    rest = (lp(dclose) - lp(piv[599])) * 1e4
    out.append(evaluate(M("U_TSLA_open_drift", "tsla", "TSLA", "1d", "sign 9:30-10:00 -> 10:00-16:00", "U", "6h"),
                        dts + pd.Timedelta(minutes=600), sgn(first30), rest.values, split, 360))
    upto = (lp(piv[929]) - lp(dopen)) * 1e4
    last30 = (lp(dclose) - lp(piv[929])) * 1e4
    out.append(evaluate(M("V_TSLA_close_drift", "tsla", "TSLA", "1d", "sign 9:30-15:30 -> 15:30-16:00", "V", "30m"),
                        dts + pd.Timedelta(minutes=930), sgn(upto), last30.values, split, 30))
    on = (lp(dopen) - lp(dclose.shift(1))) * 1e4
    out.append(evaluate(M("W_TSLA_overnight", "tsla", "TSLA", "1d", "long overnight close -> open", "W", "17.5h"),
                        dts - pd.Timedelta(hours=8), np.ones(len(on)), on.values, split, 1050))
    out.append(evaluate(M("W_TSLA_intraday", "tsla", "TSLA", "1d", "long intraday open -> close", "W", "6.5h"),
                        dts + pd.Timedelta(minutes=570), np.ones(len(oc)), oc.values, split, 390))
    out.append(evaluate(M("W_TSLA_overnight_to_intraday", "tsla", "TSLA", "1d", "sign overnight -> intraday", "W", "6.5h"),
                        dts + pd.Timedelta(minutes=570), sgn(on), oc.values, split, 390))
    # block prints: direction = sign(print price vs last lit price) -> lit 1m close 5 minutes later
    bp_ = pd.read_parquet(need(os.path.join(TSLA, "big_prints_hist.parquet")))
    bp_ = bp_[bp_["dev_bp"].notna() & (bp_["dev_bp"] != 0)]
    bp_["minute"] = (bp_["tod"] // 60).astype(int)
    closes = b.set_index(["date", "minute"])["close"]
    for name, sel in [("trf_10k", (bp_["publisher_id"] != 81) & (bp_["size"] >= 10000)),
                      ("trf_50k", (bp_["publisher_id"] != 81) & (bp_["size"] >= 50000)),
                      ("lit_10k", (bp_["publisher_id"] == 81) & (bp_["size"] >= 10000))]:
        x = bp_[sel & (bp_["minute"] + 5 < 960)]
        ex = closes.reindex(pd.MultiIndex.from_arrays([x["date"], x["minute"] + 5])).values
        fwd = (np.log(ex) - np.log(x["last_lit"].values)) * 1e4
        ts = pd.to_datetime(x["date"]) + pd.to_timedelta(x["tod"], unit="s")
        out.append(evaluate(M(f"X_TSLA_block_{name}", "tsla", "TSLA", "tick", f"block print {name} direction -> 5 min later",
                              "X", "5m"), ts.values, sgn(x["dev_bp"]), fwd, split, 5))
    b["r"] = b.groupby("date")["close"].transform(lambda x: np.log(x).diff() * 1e4)
    b["f5"] = b.groupby("date")["close"].transform(lambda x: (np.log(x.shift(-5)) - np.log(x)) * 1e4)
    med = b["n"].rolling(390, min_periods=200).median().shift(1)
    sig = sgn(b["r"]) * (b["n"] > 4 * med).values
    out.append(evaluate(M("Y_TSLA_tradecount_spike", "tsla", "TSLA", "1m", "trade count spike 4x -> next 5 min", "Y", "5m"),
                        b["ts"].values, sig, b["f5"].values, split, 5))
    for o in out:
        if o is not None:
            o["cost_price"] = cost_price
    return [o for o in out if o is not None]


if __name__ == "__main__":
    print("Moon Dev: anomaly hunt starting. 224 pre-declared tests. Real data only.")
    # BTC closes first, for the lead-lag tests
    btc = load_1m("BTCUSDT")
    btc[["close"]].to_parquet(os.path.join(CACHE, "BTC_close_1m.parquet"))
    bars(btc, "5min")[["close"]].to_parquet(os.path.join(CACHE, "BTC_close_5m.parquet"))
    del btc
    with Pool(6) as pool:
        jobs = pool.map_async(crypto_coin, COINS)
        side = [funding_tests(), liq_tests(), tsla_tests()]
        crypto = jobs.get()
    recs, liq_stats = [], {}
    for r, (s, ly) in crypto:
        recs += r
        liq_stats[s] = ly
    for r in side:
        recs += r
    print(f"Moon Dev: {len(recs)} tests evaluated")
    with open(os.path.join(CACHE, "tests.pkl"), "wb") as fh:
        pickle.dump({"recs": recs, "liq_stats": liq_stats}, fh)
    print("Moon Dev: saved cache/tests.pkl")
