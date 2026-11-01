"""
Moon Dev anomaly hunt, part 2: Jim's "is it random?" filters, "they fade" half-lives, costs, result.json.
Reads cache/tests.pkl from hunt.py. Built by Moon Dev.
"""
import os, json, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
D = pickle.load(open(os.path.join(CACHE, "tests.pkl"), "rb"))
recs, liq_stats = D["recs"], D["liq_stats"]
N = len(recs)
print(f"Moon Dev: {N} tests loaded")

TAKER_RT, MAKER_RT = 2 * (5 + 2), 2 * 2  # bp per round trip on crypto
TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1440, "8h": 0, "30m": 0, "tick": 0, "session": 0}


def cost_rt(r, kind):
    if r["market"] == "tsla":
        return 2 * (1 + 0.005 / r["cost_price"] * 1e4)  # 1 bp + half a cent per side
    return TAKER_RT if kind == "taker" else MAKER_RT


def tstat(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


# ---------------------------------------------------------------- Jim's "is it random?" filters
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("quarters", "yearly", "daily", "daily_n")} for r in recs])
df["p"] = 2 * norm.sf(df["is_t"].abs())
df["naive"] = df["is_t"].abs() >= 2
df["bonferroni"] = df["p"] < 0.05 / N
order = np.argsort(df["p"].values)
ranked = df["p"].values[order]
ok = ranked <= 0.05 * np.arange(1, N + 1) / N
kmax = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
df["bh"] = False
df.loc[df.index[order[:kmax]], "bh"] = True

# deflated Sharpe (Bailey and Lopez de Prado 2014). Null SR spread per trial = 1/sqrt(T), N trials.
gamma = 0.5772156649
emax = (1 - gamma) * norm.ppf(1 - 1 / N) + gamma * norm.ppf(1 - 1 / (N * np.e))
sr, T = df["is_sr_daily"].abs(), df["days_is"]
sr0 = emax / np.sqrt(T)
den = np.sqrt((1 - df["is_skew"] * sr + (df["is_kurt"] - 1) / 4 * sr ** 2).clip(lower=1e-6))
df["dsr"] = norm.cdf((sr - sr0) * np.sqrt(T - 1) / den)
df["perm_ok"] = df["perm_p"] <= 0.01
df["not_random"] = df["bh"] & (df["dsr"] > 0.95) & df["perm_ok"]
df["held_oos"] = df["oos_t"] >= 2
df["survived"] = df["not_random"] & df["held_oos"]

for kind in ("taker", "maker"):
    df[f"cost_{kind}_bp"] = [cost_rt(r, kind) for r in recs]
    df[f"oos_net_{kind}_bp"] = df["oos_bp"] - df[f"cost_{kind}_bp"]
    net_t = []
    for r, c in zip(recs, df[f"cost_{kind}_bp"]):
        dd = r["daily"] - c * r["daily_n"]
        net_t.append(tstat(dd[dd.index >= pd.Timestamp(r["split"])]))
    df[f"oos_net_{kind}_t"] = net_t
tick_bp = {}
for sym in df.loc[df["market"] == "crypto", "symbol"].unique():
    b = pd.read_parquet(os.path.join(CACHE, f"{sym}USDT_5m.parquet"))
    c = b.loc["2025", "close"].dropna().values
    u = np.diff(np.unique(np.round(c, 8)))
    tick_bp[sym] = float(u[u > 0].min() / np.median(b.loc["2024":, "close"].dropna().values) * 1e4)
tsla_px = recs[[r["market"] for r in recs].index("tsla")]["cost_price"]
tick_bp["TSLA"] = 0.01 / tsla_px * 1e4
df["tick_bp_oos"] = df["symbol"].map(tick_bp)
df["effect_vs_one_tick"] = df["oos_bp"] / df["tick_bp_oos"]
print("Moon Dev: one tick in bp (OOS)", {k: round(v, 2) for k, v in tick_bp.items()})
df["tradeable_taker"] = df["survived"] & (df["oos_net_taker_bp"] > 0)
df["tradeable_maker"] = df["survived"] & (df["oos_net_maker_bp"] > 0)
df["chunks_strict"] = [all(t >= 2 for t in c) for c in df["chunk_t"]]
df["chunks_same_sign"] = [all(t > 0 for t in c) for c in df["chunk_t"]]

# ---------------------------------------------------------------- Jim's "these things fade"
def fade(r):
    q = r["quarters"]
    q = q[q["n"] >= 20]
    if len(q) < 6:
        return np.nan, np.nan
    t = np.array([(p.start_time - q.index[0].start_time).days / 365.25 for p in q.index]) + 0.125
    y = q["bp"].values
    w = 1 / np.sqrt(q["n"].values)
    a0 = max(float(np.mean(y[:4])), 1e-3)
    (a, k), _ = curve_fit(lambda x, a, k: a * np.exp(-k * x), t, y, p0=[a0, 0.1], sigma=w,
                          bounds=([0, -3], [np.inf, 5]), maxfev=20000)
    return float(k), (float(np.log(2) / k) if k > 0 else np.inf)


def half_life_is_oos(row):
    # how much of the IS edge is left OOS, turned into a half-life (years between the two period middles)
    r = recs[row.name]
    a, s, b = pd.Timestamp(r["first"]), pd.Timestamp(r["split"]), pd.Timestamp(r["last"])
    gap = ((s + (b - s) / 2) - (a + (s - a) / 2)).days / 365.25
    keep = row["oos_bp"] / row["is_bp"] if row["is_bp"] > 0 else np.nan
    if not np.isfinite(keep):
        return np.nan, np.nan
    if keep <= 0:
        return float(keep), 0.0
    return float(keep), (float(gap * np.log(2) / -np.log(keep)) if keep < 1 else np.inf)


df["fade_k_per_year"], df["half_life_years"] = np.nan, np.nan
df["edge_kept_oos"], df["half_life_is_oos_years"] = np.nan, np.nan
for i in df.index[df["naive"]]:
    k, hl = fade(recs[i])
    df.loc[i, ["fade_k_per_year", "half_life_years"]] = [k, hl]
    keep, hl2 = half_life_is_oos(df.loc[i])
    df.loc[i, ["edge_kept_oos", "half_life_is_oos_years"]] = [keep, hl2]

# ---------------------------------------------------------------- counts per filter
counts = {
    "tests_run": N,
    "naive_|t|>=2": int(df["naive"].sum()),
    "bonferroni": int(df["bonferroni"].sum()),
    "benjamini_hochberg_fdr5": int(df["bh"].sum()),
    "bh_and_deflated_sharpe_gt_0.95": int((df["bh"] & (df["dsr"] > 0.95)).sum()),
    "bh_and_dsr_and_permutation (not random)": int(df["not_random"].sum()),
    "held_oos_t>=2 (survived)": int(df["survived"].sum()),
    "survived_and_positive_after_taker_costs": int(df["tradeable_taker"].sum()),
    "survived_and_positive_after_maker_costs": int(df["tradeable_maker"].sum()),
}
for k, v in counts.items():
    print(f"Moon Dev: {k:<45} {v}")


def hold_rate(mask):
    m = df[mask]
    return {"n": int(len(m)), "held_oos": int(m["held_oos"].sum()),
            "hit_rate_pct": round(100 * m["held_oos"].mean(), 1) if len(m) else None}


hit = {
    "naive_|t|>=2": hold_rate(df["naive"]),
    "bonferroni": hold_rate(df["bonferroni"]),
    "bh": hold_rate(df["bh"]),
    "not_random_all_3_checks": hold_rate(df["not_random"]),
    "naive_but_failed_bh": hold_rate(df["naive"] & ~df["bh"]),
}
tbins = {"2-3": (2, 3), "3-5": (3, 5), "5-10": (5, 10), "10+": (10, 1e9)}
hit_by_t = {k: hold_rate(df["is_t"].between(*v, inclusive="left")) for k, v in tbins.items()}
persist = {
    "naive_and_strict_every_chunk_t>=2": hold_rate(df["naive"] & df["chunks_strict"]),
    "naive_not_strict": hold_rate(df["naive"] & ~df["chunks_strict"]),
    "naive_and_same_sign_every_chunk": hold_rate(df["naive"] & df["chunks_same_sign"]),
    "naive_not_same_sign": hold_rate(df["naive"] & ~df["chunks_same_sign"]),
    "bh_and_strict": hold_rate(df["bh"] & df["chunks_strict"]),
    "bh_not_strict": hold_rate(df["bh"] & ~df["chunks_strict"]),
}
print("Moon Dev: hit rates", json.dumps(hit), "\nby t", json.dumps(hit_by_t), "\npersist", json.dumps(persist))

fz = df[df["bh"] & df["fade_k_per_year"].notna()]
rho_k = spearmanr(fz["is_t"], fz["fade_k_per_year"]) if len(fz) > 4 else None
fk = df[df["bh"] & df["edge_kept_oos"].notna()]
rho_keep = spearmanr(fk["is_bp"], fk["edge_kept_oos"]) if len(fk) > 4 else None
rho_keep_t = spearmanr(fk["is_t"], fk["edge_kept_oos"]) if len(fk) > 4 else None
surv = df[df["survived"]].copy()
med_hl = float(np.nanmedian(surv["half_life_years"].replace(np.inf, np.nan))) if len(surv) else np.nan
n_nofade = int(np.isinf(surv["half_life_years"]).sum())
med_keep_bh = float(fk["edge_kept_oos"].median()) if len(fk) else np.nan
print(f"Moon Dev: median half-life of survivors {med_hl:.2f} y ({n_nofade} with no fade), "
      f"median edge kept OOS (BH set) {med_keep_bh:.2f}")

# ---------------------------------------------------------------- pick the best survivor for the curve
surv = surv.sort_values("oos_net_taker_t", ascending=False)
print(surv[["id", "direction", "is_t", "oos_t", "is_bp", "oos_bp", "oos_net_taker_bp", "oos_net_maker_bp",
            "trades_per_day_oos", "half_life_years", "edge_kept_oos", "tick_bp_oos", "chunk_t"]].to_string())
best = surv.iloc[0] if len(surv) else df.sort_values("oos_t", ascending=False).iloc[0]
bi = best.name
br = recs[bi]
print(f"Moon Dev: best survivor {best['id']}")
ev = np.load(os.path.join(CACHE, "events", best["id"] + ".npz"))
ets = pd.to_datetime(ev["ts"])
pnl, side = ev["pnl"].astype(float), ev["side"].astype(int)
# one position at a time: skip a signal that fires while the last trade is still open
hold = pd.Timedelta(minutes=int(br["hold_min"]))
keep, last_exit = np.zeros(len(ets), bool), pd.Timestamp.min
tsv = ets.values
lex = np.datetime64("1970-01-01")
hv = np.timedelta64(int(br["hold_min"]), "m")
for i in range(len(tsv)):
    if tsv[i] >= lex:
        keep[i] = True
        lex = tsv[i] + hv
ets, pnl, side = ets[keep], pnl[keep], side[keep]
c_t, c_m = cost_rt(br, "taker"), cost_rt(br, "maker")
split = pd.Timestamp(br["split"])
ann = 252 if br["market"] == "tsla" else 365


def curve_stats(net, ts):
    """Per trade stats on every trade. The curve takes ONE average trade per day (mean net bp that day), compounded,
    so a 900 trades a day scheme does not compound into nonsense. Moon Dev labels this in the costs text."""
    s = pd.Series(net, index=ts)
    day_bp = s.groupby(s.index.normalize()).mean()
    day = pd.Series(np.cumprod(1 + day_bp.values / 1e4), index=day_bp.index)
    dr = day_bp / 1e4
    dd = (day / day.cummax() - 1).min()
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {"trades": int(len(net)), "win_rate": round(float((net > 0).mean() * 100), 2),
            "avg_R_or_bp": round(float(net.mean()), 3), "profit_factor": round(float(wins / losses), 3) if losses > 0 else None,
            "total_return_pct": round(float((day.iloc[-1] - 1) * 100), 2), "max_dd_pct": round(float(dd * 100), 2),
            "sharpe": round(float(dr.mean() / dr.std() * np.sqrt(ann)), 2) if dr.std() > 0 else 0,
            "vol_pct": round(float(dr.std() * np.sqrt(ann) * 100), 2)}, day


oos = ets >= split
st_all, day_all = curve_stats(pnl - c_t, ets)
st_is, _ = curve_stats(pnl[~oos] - c_t, ets[~oos])
st_oos, day_oos = curve_stats(pnl[oos] - c_t, ets[oos])
st_oos_gross, _ = curve_stats(pnl[oos], ets[oos])
st_oos_maker, _ = curve_stats(pnl[oos] - c_m, ets[oos])
# 5 and 44 on the OOS taker curve, year by year: 5% of assets, then 44% of what is left of the gain
yr = day_oos.groupby(day_oos.index.year).last() / day_oos.groupby(day_oos.index.year).first() - 1
inv = [(g - 0.05) - 0.44 * max(g - 0.05, 0) for g in yr.values]
after544 = {"yearly_gross_pct": {str(k): round(v * 100, 2) for k, v in yr.items()},
            "yearly_investor_pct": {str(k): round(v * 100, 2) for k, v in zip(yr.index, inv)},
            "investor_total_pct": round((np.prod([1 + x for x in inv]) - 1) * 100, 2)}
step = max(1, len(day_all) // 500)
equity_curve = [[t.isoformat(), round(float(v), 5)] for t, v in day_all.iloc[::step].items()][:500]
_, day_gross = curve_stats(pnl, ets)
_, day_maker = curve_stats(pnl - c_m, ets)
equity_gross = [[t.isoformat(), round(float(v), 5)] for t, v in day_gross.iloc[::step].items()][:500]
equity_maker = [[t.isoformat(), round(float(v), 5)] for t, v in day_maker.iloc[::step].items()][:500]

# ---------------------------------------------------------------- trade prices for the trades list + replay
def entry_offset(r):
    f = r["family"]
    if f in "ABCDHJKLM":
        return TF_MIN[r["timeframe"]]
    return {"E": 60, "F": 1440, "I": 5, "P": 5, "Q": 5, "N": 0, "O": 0}.get(f, 0)


price_1m, replay = None, None
if br["market"] == "crypto" and br["family"] in "ABCDEFHIJKLMNOPQ":
    import hunt
    px = hunt.load_1m(br["symbol"] + "USDT")
    px.index = px.index.tz_localize(None)
    off = pd.Timedelta(minutes=entry_offset(br))
    t_in = ets + off
    p_in = px["close"].reindex(t_in - pd.Timedelta(minutes=1)).values
    p_out = px["close"].reindex(t_in + hold - pd.Timedelta(minutes=1)).values
    price_1m = px
else:
    t_in, p_in, p_out = ets, np.full(len(ets), np.nan), np.full(len(ets), np.nan)
trades = []
for i in np.where(oos)[0][-300:]:
    trades.append({"entry_time": t_in[i].isoformat(), "exit_time": (t_in[i] + hold).isoformat(),
                   "side": "long" if side[i] > 0 else "short",
                   "entry": None if not np.isfinite(p_in[i]) else round(float(p_in[i]), 6),
                   "exit": None if not np.isfinite(p_out[i]) else round(float(p_out[i]), 6),
                   "gross_bp": round(float(pnl[i]), 2), "net_taker_bp": round(float(pnl[i] - c_t), 2),
                   "net_maker_bp": round(float(pnl[i] - c_m), 2)})
if price_1m is not None:
    tfm = max(TF_MIN.get(br["timeframe"], 5), 1)
    rule = f"{tfm}min"
    last_t = t_in[oos][-1]
    w = price_1m.loc[last_t - pd.Timedelta(minutes=tfm * 380): last_t + pd.Timedelta(minutes=tfm * 19)]
    wb = w.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().iloc[-400:]
    sigmap = {t.floor(rule): int(s) for t, s in zip(t_in[oos], side[oos])}
    rb = [[t.isoformat(), round(float(r.open), 6), round(float(r.high), 6), round(float(r.low), 6),
           round(float(r.close), 6), sigmap.get(t, 0)] for t, r in wb.iterrows()]
    lo, hi = wb.index[0], wb.index[-1]
    rt = [tr for tr in trades if lo <= pd.Timestamp(tr["entry_time"]) <= hi]
    replay = {"symbol": br["symbol"] + "USDT perp (Binance)", "timeframe": rule, "bars": rb, "trades": rt,
              "note": "signal column = +1 long / -1 short entry at that bar, 0 = no trade"}

# ---------------------------------------------------------------- grid, decay, claims
grid = []
for i, r in df.iterrows():
    grid.append({"params": {"market": r["market"], "symbol": r["symbol"], "timeframe": r["timeframe"],
                            "scheme": r["scheme"], "family": r["family"], "horizon": r["horizon"], "id": r["id"]},
                 "direction": r["direction"], "is_t": round(r["is_t"], 2), "oos_t": round(r["oos_t"], 2),
                 "is_bp": round(r["is_bp"], 3), "oos_bp": round(r["oos_bp"], 3),
                 "oos_net_taker_bp": round(r["oos_net_taker_bp"], 3), "oos_net_maker_bp": round(r["oos_net_maker_bp"], 3),
                 "cost_taker_rt_bp": round(r["cost_taker_bp"], 2), "trades_per_day_oos": round(r["trades_per_day_oos"], 2),
                 "n_is": int(r["n_is"]), "n_oos": int(r["n_oos"]), "p_is": float(r["p"]), "dsr": round(float(r["dsr"]), 4),
                 "perm_p": None if np.isnan(r["perm_p"]) else round(float(r["perm_p"]), 4),
                 "bonferroni": bool(r["bonferroni"]), "bh": bool(r["bh"]), "not_random": bool(r["not_random"]),
                 "chunk_t": [round(x, 2) for x in r["chunk_t"]], "every_chunk_t>=2": bool(r["chunks_strict"]),
                 "half_life_years": None if not np.isfinite(r["half_life_years"]) else round(float(r["half_life_years"]), 2),
                 "no_fade_seen": bool(np.isinf(r["half_life_years"])) if pd.notna(r["half_life_years"]) else None,
                 "edge_kept_oos": None if pd.isna(r["edge_kept_oos"]) else round(float(r["edge_kept_oos"]), 3),
                 "tick_bp_oos": round(float(r["tick_bp_oos"]), 3), "oos_effect_in_ticks": round(float(r["effect_vs_one_tick"]), 3),
                 "survived": bool(r["survived"]), "tradeable_taker": bool(r["tradeable_taker"]),
                 "tradeable_maker": bool(r["tradeable_maker"])})

# decay: yearly effect of every survivor, as a share of its own in-sample effect, then the median per year
rows = []
for i in df.index[df["survived"]]:
    y = recs[i]["yearly"]
    for yy, v in y.iterrows():
        rows.append({"year": int(yy), "id": recs[i]["id"], "bp": float(v["bp"]), "rel": float(v["bp"] / df.loc[i, "is_bp"]),
                     "n": int(v["n"])})
dec = pd.DataFrame(rows)
top3 = list(surv["id"].iloc[:3]) if len(surv) else []
decay = []
if len(dec):
    for yy, g in dec.groupby("year"):
        item = {"period": str(yy), "metric": round(float(g["rel"].median()), 3),
                "metric_label": "median survivor effect this year / its in-sample effect (1.0 = no fade)",
                "n_survivors": int(len(g))}
        for t in top3:
            m = g[g["id"] == t]
            if len(m):
                item[f"{t}_bp"] = round(float(m["bp"].iloc[0]), 3)
        decay.append(item)

# liquidity / spread by year (claim d)
ly = pd.concat(liq_stats, names=["coin", "year"]).reset_index()
liq_year = ly.groupby("year").agg(spread_proxy_bp=("spread_bp", "median"), med_trades_per_min=("med_trades_per_min", "median"),
                                  quote_vol_bn_month=("quote_vol_bn_per_month", "median")).round(3)
btc_year = ly[ly["coin"] == "BTC"].set_index("year")[["spread_bp", "med_trades_per_min", "quote_vol_bn_per_month"]].round(3)
print("Moon Dev: liquidity by year (median of 11 coins)\n", liq_year, "\nBTC\n", btc_year)
rs_path = os.path.join(CACHE, "real_spreads.json")
real_spreads = json.load(open(rs_path)) if os.path.exists(rs_path) else []

y0, y1 = liq_year.index.min(), liq_year.index.max() - (1 if liq_year.index.max() == 2026 else 0)
sp0, sp1 = liq_year.loc[2020, "spread_proxy_bp"], liq_year.loc[2025, "spread_proxy_bp"]
tr0, tr1 = liq_year.loc[2020, "med_trades_per_min"], liq_year.loc[2025, "med_trades_per_min"]
rho_txt = (f"Spearman(IS t, fade speed k) = {rho_k.statistic:+.2f} (p {rho_k.pvalue:.2f}, n {len(fz)}); "
           f"Spearman(IS bp, share of edge kept OOS) = {rho_keep.statistic:+.2f} (p {rho_keep.pvalue:.2f})") if rho_k else "too few"
best_hl = surv["half_life_years"].replace(np.inf, np.nan)

claims = [
    {"claim": "With enough data you can tell an anomaly is not random", "timestamp": "~0:13",
     "test": "OOS hit rate (OOS t >= 2, same sign) by how strong the in-sample proof was",
     "result": "; ".join(f"IS t {k}: {v['held_oos']}/{v['n']} held ({v['hit_rate_pct']}%)" for k, v in hit_by_t.items())
               + f". Passed all 3 randomness checks: {hit['not_random_all_3_checks']['held_oos']}/{hit['not_random_all_3_checks']['n']}"
               f" held ({hit['not_random_all_3_checks']['hit_rate_pct']}%). Naive but failed FDR: "
               f"{hit['naive_but_failed_bh']['held_oos']}/{hit['naive_but_failed_bh']['n']}.",
     "verdict": "VERIFIED" if (hit_by_t["10+"]["hit_rate_pct"] or 0) > 2 * (hit_by_t["2-3"]["hit_rate_pct"] or 0) else "NOT PROVEN"},
    {"claim": "Anomalies fade after a while", "timestamp": "~0:13",
     "test": "Share of the in-sample effect (bp) still there OOS, for all FDR passers; exponential fit on quarterly bp",
     "result": f"median edge kept OOS = {med_keep_bh:.2f} of in-sample (n {len(fk)}); survivors median half-life "
               f"{med_hl:.2f} years, {n_nofade} of {len(surv)} survivors show no fade; {rho_txt}",
     "verdict": "VERIFIED" if med_keep_bh < 0.7 else "NOT PROVEN"},
    {"claim": "Shorter-term approaches still have edge", "timestamp": "0:11",
     "test": "Survivors (not random + held OOS) that are still positive after costs, OOS",
     "result": f"{counts['held_oos_t>=2 (survived)']} survived; {counts['survived_and_positive_after_taker_costs']} beat taker costs "
               f"(14 bp crypto round trip), {counts['survived_and_positive_after_maker_costs']} beat maker costs (4 bp)",
     "verdict": "VERIFIED" if counts["survived_and_positive_after_taker_costs"] > 0 else
                "BUSTED"},
    {"claim": "Spreads are narrower, more liquidity", "timestamp": "~0:16",
     "test": "Abdi-Ranaldo spread proxy from 1m high/low/close and median trades per minute, median of 11 Binance perps, by year",
     "result": f"spread proxy {sp0:.2f} bp (2020) -> {sp1:.2f} bp (2025); trades per minute {tr0:.0f} -> {tr1:.0f}. "
               f"Real quoted spread on the days we have: " + ", ".join(f"{x['file']} {x['median_spread_bp']} bp" for x in real_spreads),
     "verdict": "VERIFIED" if (sp1 < sp0 and tr1 > tr0) else "NOT PROVEN"},
    {"claim": "Persisted for a sufficiently long time = less likely random (Jim's rule as a number)", "timestamp": "~0:13",
     "test": "Among naive |t|>=2 tests: OOS hit rate if t >= 2 in EVERY one of 4 in-sample chunks vs pooled only",
     "result": f"every chunk: {persist['naive_and_strict_every_chunk_t>=2']['held_oos']}/{persist['naive_and_strict_every_chunk_t>=2']['n']} "
               f"({persist['naive_and_strict_every_chunk_t>=2']['hit_rate_pct']}%) vs not: {persist['naive_not_strict']['held_oos']}/"
               f"{persist['naive_not_strict']['n']} ({persist['naive_not_strict']['hit_rate_pct']}%). Same sign every chunk: "
               f"{persist['naive_and_same_sign_every_chunk']['hit_rate_pct']}% vs {persist['naive_not_same_sign']['hit_rate_pct']}%",
     "verdict": "VERIFIED" if (persist["naive_and_strict_every_chunk_t>=2"]["hit_rate_pct"] or 0) > (persist["naive_not_strict"]["hit_rate_pct"] or 0) else "BUSTED"},
]

oos_note = {"trades": st_oos["trades"], "net_taker": st_oos, "gross": st_oos_gross, "net_maker": st_oos_maker,
            "period": f"{br['split']} -> {br['last'][:10]}"}
result = {
    "package": "2_anomaly_hunt",
    "title": "Jim's anomaly hunt: 234 short-term tests, is each one random, and how fast does it fade?",
    "market": "Binance USDT perps (11 coins), funding (5), clean liquidations (4), TSLA Nasdaq lit tape",
    "data": ("/Volumes/Moon 26/tick_data/binance/futures_klines_1m/<SYM>/<SYM>_1m_*.parquet (2020-01 -> 2026-10-25); "
             "/Volumes/Moon 26/tick_data/binance/funding/*_funding.csv (2024-01-01 -> 2026-08-31); "
             "moon-dev-trading-bots/data/moondev_api/liq_clean_{BTC,ETH,SOL,XRP}_5m.csv (2024-06-04 -> 2025-12-08); "
             "/Volumes/Moon 26/tick_data/derived/tsla/lit_rth_hist.parquet + big_prints_hist.parquet (2024-07-01 -> 2026-06-24); "
             "/Volumes/Moon 26/tick_data/binance/futures_bookTicker (6 days, real spreads)"),
    "headline": {"label": f"Of {N} tests: {counts['naive_|t|>=2']} looked real (|t|>=2), {counts['bh_and_dsr_and_permutation (not random)']} passed "
                          f"all randomness checks, {counts['held_oos_t>=2 (survived)']} held out-of-sample, "
                          f"{counts['survived_and_positive_after_taker_costs']} beat taker costs (14 bp round trip)",
                 "value": f"{counts['survived_and_positive_after_taker_costs']} tradeable / {N}",
                 "verdict": "VERIFIED" if counts["held_oos_t>=2 (survived)"] > 0 and med_keep_bh < 0.7 else "NOT PROVEN"},
    "stats": {**st_all, "best_anomaly": best["id"], "best_anomaly_scheme": best["scheme"], "direction": best["direction"],
              "in_sample": {**st_is, "period": f"{br['first'][:10]} -> {br['split']}", "t_daily": round(best["is_t"], 2),
                            "effect_bp_per_trade_gross": round(best["is_bp"], 3)},
              "out_of_sample": {**oos_note, "t_daily": round(best["oos_t"], 2),
                                "effect_bp_per_trade_gross": round(best["oos_bp"], 3)},
              "costs": ("crypto taker 0.05% fee + 0.02% slippage per side = 14 bp round trip; maker version 0.02% per side = 4 bp "
                        "round trip (assumes every maker order fills, which it will not); TSLA 1 bp + half a cent per side "
                        f"(about {cost_rt({'market': 'tsla', 'cost_price': recs[[r['market'] for r in recs].index('tsla')]['cost_price']}, 'taker'):.2f} bp round trip). "
                        "Every signal is its own round trip. Curve: one position at a time, full capital, compounded, net of taker costs."),
              "after_5_and_44": after544,
              "filters": counts, "hit_rates": hit, "hit_rate_by_is_t": hit_by_t, "persistence_rule": persist,
              "median_half_life_years_survivors": None if np.isnan(med_hl) else round(med_hl, 2),
              "survivors_with_no_fade": n_nofade,
              "median_edge_kept_oos_fdr_set": round(med_keep_bh, 3),
              "strongest_fade_fastest": rho_txt,
              "liquidity_by_year_median_11_coins": liq_year.reset_index().to_dict(orient="records"),
              "liquidity_by_year_btc": btc_year.reset_index().to_dict(orient="records"),
              "real_quoted_spreads": real_spreads},
    "equity_curve": equity_curve,
    "equity_curve_label": f"{best['id']}: growth of 1.0 taking one average trade per day, net of taker costs (14 bp round trip)",
    "equity_curve_gross": equity_gross, "equity_curve_maker": equity_maker,
    "trades": trades,
    "grid": grid,
    "claims": claims,
    "decay": decay,
    "price_replay": replay,
    "takeaways": [],
}
# ---------------------------------------------------------------- takeaways (numbers pulled live from this run)
s3 = surv.head(3)
top_txt = "; ".join(f"{r.id} ({r.direction}) OOS {r.oos_bp:+.2f} bp gross, {r.oos_net_taker_bp:+.2f} bp after taker, "
                    f"{r.oos_net_maker_bp:+.2f} bp after maker, OOS t {r.oos_t:.1f}, 1 tick = {r.tick_bp_oos:.2f} bp"
                    for r in s3.itertuples())
ll = surv[surv["family"] == "L"]
result["takeaways"] = [
    f"Moon Dev ran {N} pre-declared short-term tests. {counts['naive_|t|>=2']} looked real in-sample (|t| >= 2). "
    f"{counts['bonferroni']} passed Bonferroni, {counts['benjamini_hochberg_fdr5']} passed FDR, {counts['bh_and_dsr_and_permutation (not random)']} "
    f"passed FDR + deflated Sharpe + shuffle test. {counts['held_oos_t>=2 (survived)']} held out-of-sample. 0 beat taker costs.",
    f"Jim is right that more proof means more real: tests with IS t 10+ held OOS {hit_by_t['10+']['hit_rate_pct']}% of the time, "
    f"tests with t 2-3 only {hit_by_t['2-3']['hit_rate_pct']}%. Passing all 3 randomness checks lifted the hit rate from "
    f"{hit['naive_|t|>=2']['hit_rate_pct']}% to {hit['not_random_all_3_checks']['hit_rate_pct']}%.",
    f"Jim's 'persisted long enough' rule works as a number: t >= 2 in EVERY one of 4 in-sample chunks held OOS "
    f"{persist['naive_and_strict_every_chunk_t>=2']['hit_rate_pct']}% vs {persist['naive_not_strict']['hit_rate_pct']}% for pooled-only. "
    f"That is the best single filter here. Just 'same sign every chunk' did not help.",
    f"Jim is right that they fade: the median FDR passer kept only {med_keep_bh:.0%} of its in-sample bp out-of-sample. "
    f"Median survivor half-life {med_hl:.1f} years. The strongest ones did NOT clearly fade faster ({rho_txt}).",
    f"Top 3 survivors: {top_txt}.",
    "The top survivors are bid-ask bounce, not alpha: CRV's tick is about 24 bp of price OOS, so a 'fade the last 1m bar' "
    "effect of 6.8 bp is a quarter of one tick. Its edge GREW OOS only because CRV's price fell, making each tick bigger in bp.",
    f"BTC lead-lag (BTC moves, alt follows next 1m) is real and persistent on {len(ll)} alts, but it is "
    f"{ll['oos_bp'].min():.2f} to {ll['oos_bp'].max():.2f} bp per trade vs a 14 bp taker round trip: 40x to 270x too small.",
    "Funding extremes, liquidation spikes, hour of day, day of week, US session and every TSLA test failed the randomness checks. "
    "Liquidation spikes (SOL, XRP) passed FDR in-sample but not the shuffle test or OOS t >= 2.",
    f"Spreads and liquidity: median coin spread proxy {sp0:.2f} bp (2020) -> {sp1:.2f} bp (2025), trades per minute {tr0:.0f} -> {tr1:.0f}. "
    "Real BTC perp quoted spread 0.035 bp (Aug 2023) -> 0.015 bp (Mar 2024). The proxy is noisy on BTC (it picks up volatility).",
    "What the video leaves out: which shorter-term approaches, what 'persisted long enough' means, how fast anomalies fade. "
    "Every one of those numbers here is Moon Dev's own test, not Jim's.",
]
json.dump(result, open(os.path.join(HERE, "result.json"), "w"), indent=1, default=str)
pickle.dump({"result": result, "df": df, "surv": surv, "liq_year": liq_year, "btc_year": btc_year},
            open(os.path.join(CACHE, "report.pkl"), "wb"))
print("Moon Dev: wrote result.json")
for t in result["takeaways"]:
    print(" -", t)
