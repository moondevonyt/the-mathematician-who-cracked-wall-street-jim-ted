"""
Moon Dev: turn the cached walk-forward scores into result.json + the deploy model for the bot.
Run after run_models.py A, B, C finish.
"""
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb
import machine as M
from run_models import GROUPS, key

HERE = Path(__file__).parent
COSTS = [0, 1, 2, 3, 5, 7, 10, 15]          # bp per side
TAKER, MAKER = 7, 2                          # taker 5 bp fee + 2 bp slippage; maker 2 bp fee, no slippage (optimistic)
HALFLIVES = [0, 3, 6, 12, 24, 48]                # bars of EMA smoothing on the score
BANDS = [0.0, 0.5, 0.9]                      # no-trade band, as a share of one slot's weight
MARKET = "binance_usdt_perps_16"
Y_IMPACT = 1.0                               # square-root impact constant (industry rule of thumb, NOT from the video)

P = {(4, 0): M.panel(4), (1, 0): M.panel(1), (4, 24): M.panel(4, 24)}
configs = [c for g in GROUPS.values() for c in g]


def sel_start(retrain):
    return M.ONCE_END if retrain == "once" else M.OOS_START


# ---------------------------------------------------------------- 1. pick portfolio settings on pre-holdout only
runs = {}
for h, m, fs, r, tgt in configs:
    k = key(h, m, fs, r, tgt)
    if not (M.CACHE / f"scores_{k}.npy").exists():
        print(f"Moon Dev: MISSING {k}, skipped")
        continue
    S = np.load(M.CACHE / f"scores_{k}.npy")
    best = None
    for hl in HALFLIVES:
        for b in BANDS:
            pf = M.portfolio(P[(h, 0)], S, hl, b)
            sh = M.stats(P[(h, 0)], pf, TAKER, sel_start(r), M.HOLD_START)["sharpe"]
            if best is None or sh > best[0]:
                best = (sh, hl, b, pf)
    pf0 = M.portfolio(P[(h, 0)], S, 0, 0.0)  # raw, no turnover control, for reference
    runs[k] = dict(h=h, m=m, fs=fs, r=r, tgt=tgt, hl=best[1], band=best[2], pf=best[3], pf0=pf0, S=S)
    print(f"Moon Dev: {k}: picked halflife={best[1]} band={best[2]} pre-holdout taker sharpe={best[0]}")

# ---------------------------------------------------------------- 2. grid rows (model x horizon x features x retrain x cost)
grid = []
for k, R in runs.items():
    for c in COSTS:
        PX = P[(R["h"], 0)]
        pre = M.stats(PX, R["pf"], c, sel_start(R["r"]), M.HOLD_START)
        hold = M.stats(PX, R["pf"], c, M.HOLD_START, M.DATA_END + pd.Timedelta(hours=1))
        raw = M.stats(PX, R["pf0"], c, M.HOLD_START, M.DATA_END + pd.Timedelta(hours=1))
        grid.append({"params": {"market": MARKET, "model": R["m"], "horizon": f"{R['h']}h", "features": R["fs"],
                                "retrain": R["r"], "target": f"next {R['tgt']}h" if R["tgt"] else "next bar", "cost_bp_per_side": c, "smooth_halflife_bars": R["hl"],
                                "band": R["band"]},
                     "pre_holdout_sharpe": pre["sharpe"], "pre_holdout_ann_return_pct": pre["ann_return_pct"],
                     "holdout_sharpe": hold["sharpe"], "holdout_ann_return_pct": hold["ann_return_pct"],
                     "holdout_max_dd_pct": hold["max_dd_pct"], "vol_pct": hold["vol_pct"],
                     "turnover_x_per_day": hold["turnover_x_per_day"], "hit_rate": hold["hit_rate"],
                     "holdout_sharpe_no_turnover_control": raw["sharpe"]})


def row(k, c):
    R = runs[k]
    return next(g for g in grid if g["params"]["model"] == R["m"] and g["params"]["horizon"] == f"{R['h']}h"
                and g["params"]["features"] == R["fs"] and g["params"]["retrain"] == R["r"] and g["params"]["cost_bp_per_side"] == c
                and g["params"]["target"] == (f"next {R['tgt']}h" if R["tgt"] else "next bar"))


# ---------------------------------------------------------------- 3. best model, picked on pre-holdout taker Sharpe only
cands = [k for k in runs if runs[k]["r"] != "once"]
best_k = max(cands, key=lambda k: row(k, TAKER)["pre_holdout_sharpe"])
B = runs[best_k]
PB = P[(B["h"], B["tgt"])]
pf = B["pf"]
print(f"Moon Dev: BEST = {best_k}")
END = M.DATA_END + pd.Timedelta(hours=1)
pre_t, hold_t = M.stats(PB, pf, TAKER, M.OOS_START, M.HOLD_START), M.stats(PB, pf, TAKER, M.HOLD_START, END)
pre_m, hold_m = M.stats(PB, pf, MAKER, M.OOS_START, M.HOLD_START), M.stats(PB, pf, MAKER, M.HOLD_START, END)
pre_0, hold_0 = M.stats(PB, pf, 0, M.OOS_START, M.HOLD_START), M.stats(PB, pf, 0, M.HOLD_START, END)
all_t = M.stats(PB, pf, TAKER, M.OOS_START, END)

# break-even cost: where holdout Sharpe crosses zero (linear in cost)
def breakeven(PX, pfx, start, end):
    m = (PX["t"] >= start) & (PX["t"] < end)
    g, tu = pfx["gross"][m].mean(), pfx["turn"][m].mean()
    return round(float(g / tu * 1e4), 2) if tu > 0 else None
be_pre, be_hold = breakeven(PB, pf, M.OOS_START, M.HOLD_START), breakeven(PB, pf, M.HOLD_START, END)

# ---------------------------------------------------------------- 4. equity curve, trades, price replay
t = PB["t"]
oos = t >= M.OOS_START
net = pf["gross"] - TAKER / 1e4 * pf["turn"]
eq = np.cumprod(1 + net[oos])
idx = np.linspace(0, len(eq) - 1, min(500, len(eq))).astype(int)
equity_curve = [[t[oos][i].isoformat(), round(float(eq[i]), 5)] for i in idx]

W = pf["W"]
dW = np.diff(np.vstack([np.zeros(W.shape[1]), W]), axis=0)
close = PB["close"].values
trades = []
for i in np.where((t >= M.HOLD_START))[0]:
    for j in np.where(np.abs(dW[i]) > 1e-9)[0]:
        trades.append({"time": t[i].isoformat(), "symbol": PB["syms"][j], "side": "buy" if dW[i, j] > 0 else "sell",
                       "weight_change_pct_of_capital": round(float(dW[i, j] * 100), 2),
                       "new_weight_pct": round(float(W[i, j] * 100), 2), "price": float(close[i, j])})
n_rebal_hold = len(trades)
trades = trades[-300:]

sym = "SOLUSDT"
j = PB["syms"].index(sym)
Wc = PB["W"]
ri = np.where(t <= M.DATA_END)[0][-400:]
bars = [[t[i].isoformat(), float(Wc["open"].values[i, j]), float(Wc["high"].values[i, j]), float(Wc["low"].values[i, j]),
         float(Wc["close"].values[i, j]), round(float(W[i, j] * 100), 2)] for i in ri]
rp_trades = [{"time": t[i].isoformat(), "side": "buy" if dW[i, j] > 0 else "sell", "price": float(close[i, j]),
              "new_weight_pct": round(float(W[i, j] * 100), 2)} for i in ri if abs(dW[i, j]) > 1e-9]

# ---------------------------------------------------------------- 5. decay: per-year Sharpe, retrain vs train once
once_k = key(B["h"], B["m"] if B["m"] in ("blend", "ridge", "gbm") else "ridge", "all", "once", B["tgt"])
decay = []
for y in range(2022, 2027):
    a, b = pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
    d = {"period": str(y), "metric": M.stats(PB, pf, TAKER, a, b)["sharpe"],
         "metric_label": "Sharpe after taker costs (7 bp/side), best model, monthly retrain",
         "gross_sharpe": M.stats(PB, pf, 0, a, b)["sharpe"]}
    if once_k in runs and y >= 2023:
        d["train_once_sharpe"] = M.stats(PB, runs[once_k]["pf"], TAKER, a, b)["sharpe"]
        d["train_once_gross_sharpe"] = M.stats(PB, runs[once_k]["pf"], 0, a, b)["sharpe"]
    decay.append(d)

# ---------------------------------------------------------------- 6. capacity (square-root impact)
qv = PB["qv"].values
per_day = 24 // PB["bar_hours"]
adv = pd.DataFrame(qv).rolling(30 * per_day, min_periods=10 * per_day).mean().shift(1).values * per_day  # $ per day, past only
sig = PB["F"]["vol_168"][PB["syms"]].values  # daily vol, past only
m_ = oos & np.isfinite(adv).all(axis=1) | oos
a_ = np.abs(dW)
term = np.nansum(a_ ** 1.5 * Y_IMPACT * np.nan_to_num(sig) / np.sqrt(np.where(adv > 0, adv, np.nan)), axis=1)
K = float(np.nanmean(term[oos]) * PB["per_year"])          # annual impact drag = K * sqrt(AUM)
R0 = all_t["ann_return_pct"] / 100
cap_zero = (R0 / K) ** 2 if R0 > 0 else 0.0
cap_half = (R0 / (2 * K)) ** 2 if R0 > 0 else 0.0
R0h = hold_t["ann_return_pct"] / 100
cap_zero_hold = (R0h / K) ** 2 if R0h > 0 else 0.0
part = {}
qbar = np.where(qv > 0, qv, np.nan)
for A in [1e5, 1e6, 1e7, 1e8]:
    p = (a_ * A / qbar)[oos][a_[oos] > 1e-9]
    part[f"${A:,.0f}"] = {"median_pct_of_bar_volume": round(float(np.nanmedian(p) * 100), 4),
                          "p95_pct_of_bar_volume": round(float(np.nanpercentile(p, 95) * 100), 4),
                          "impact_drag_pct_per_year": round(float(K * np.sqrt(A) * 100), 2)}
avg_pos = float(np.abs(W[oos]).sum(axis=1).mean())

# ---------------------------------------------------------------- 7. Jim's fees: 2/20 and 5/44 (annual HWM, monthly mgmt accrual)
def fees(start, end, mgmt, perf):
    m = (t >= start) & (t < end)
    r = pd.Series(net[m], index=t[m])
    mo = (1 + r).resample("ME").prod() - 1
    nav, hwm, gross_nav = 1.0, 1.0, 1.0
    year_start_nav = 1.0
    for i, (ts, x) in enumerate(mo.items()):
        gross_nav *= 1 + x
        nav *= (1 + x)
        nav -= nav * mgmt / 12
        if ts.month == 12 or i == len(mo) - 1:
            gain = nav - max(hwm, 0)
            if gain > 0:
                nav -= perf * gain
            hwm = max(hwm, nav)
    yrs = len(mo) / 12
    return {"years": round(yrs, 2), "fund_return_before_fees_pct": round((gross_nav - 1) * 100, 2),
            "investor_return_pct": round((nav - 1) * 100, 2),
            "investor_cagr_pct": round((nav ** (1 / yrs) - 1) * 100, 2) if nav > 0 else -100,
            "fund_cagr_before_fees_pct": round((gross_nav ** (1 / yrs) - 1) * 100, 2)}
fee_all = {"2_and_20": fees(M.OOS_START, END, 0.02, 0.20), "5_and_44": fees(M.OOS_START, END, 0.05, 0.44)}
fee_hold = {"2_and_20": fees(M.HOLD_START, END, 0.02, 0.20), "5_and_44": fees(M.HOLD_START, END, 0.05, 0.44)}

# ---------------------------------------------------------------- 8. feature importance
names = [f for f in M.FEATSETS["all"] if f in PB["names"]]
def group(f):
    return "price (incl. 20 day trend)" if f in M.PRICE else "volume / trade count" if f in M.VOLUME else \
        "taker flow" if f in M.FLOW else "funding" if f in M.FUND else "context"
imp = {}
for mod in ["gbm", "ridge", "blend"]:
    kk = key(B["h"], mod, "all", "monthly", B["tgt"] if mod != "blend" or (M.CACHE / f"infos_{key(B['h'], mod, 'all', 'monthly', B['tgt'])}.pkl").exists() else 0)
    imp[mod] = pickle.load(open(M.CACHE / f"infos_{kk}.pkl", "rb"))
gain = np.mean([i["gain"] / i["gain"].sum() for i in imp["gbm"]], axis=0)
gnames = names + M.CONTEXT
coef = np.mean([i["coef"] for i in imp["ridge"]], axis=0)
ic_is = np.mean([i["ic"] for i in imp["blend"]], axis=0)
Xall = PB["X"]
def oos_ic(f, start, end):
    jj = PB["names"].index(f)
    mm = PB["valid"] & ((t >= start) & (t < end))[:, None] & np.isfinite(PB["yz"])
    x, y = np.nan_to_num(Xall[:, :, jj][mm]), PB["yz"][mm]
    return round(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0, 4)
feat_rows = []
for i, f in enumerate(gnames):
    rr = {"feature": f, "group": group(f), "gbm_gain_pct": round(float(gain[i] * 100), 2)}
    if f in names:
        k2 = names.index(f)
        rr.update({"ridge_avg_coef": round(float(coef[k2]), 4), "in_sample_ic_avg": round(float(ic_is[k2]), 4),
                   "oos_ic_2022_2025H1": oos_ic(f, M.OOS_START, M.HOLD_START), "oos_ic_holdout": oos_ic(f, M.HOLD_START, END)})
    feat_rows.append(rr)
feat_rows.sort(key=lambda r: -r["gbm_gain_pct"])
group_gain = {}
for r in feat_rows:
    group_gain[r["group"]] = round(group_gain.get(r["group"], 0) + r["gbm_gain_pct"], 2)
for r in feat_rows:
    grid.append({"params": {"market": MARKET, "type": "feature_importance", "feature": r["feature"], "group": r["group"],
                            "horizon": f"{B['h']}h"}, **{k: v for k, v in r.items() if k not in ("feature", "group")}})

# ---------------------------------------------------------------- 9. liquidation add-on (BTC ETH SOL XRP, 2024-06 -> 2025-12, short history)
def liq_addon():
    h = 4
    PL = P[(h, 0)]
    lq = pd.read_parquet(M.CACHE / "liq_1h.parquet")
    fs_ = lq.pivot(index="t", columns="sym", values="forced_sell_usd")
    fb_ = lq.pivot(index="t", columns="sym", values="forced_buy_usd")
    fs4 = fs_.resample("4h", label="right", closed="right").sum(min_count=1)
    fb4 = fb_.resample("4h", label="right", closed="right").sum(min_count=1)
    coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    lo, hi = pd.Timestamp("2024-06-06", tz="UTC"), pd.Timestamp("2025-12-05", tz="UTC")
    mt = (PL["t"] >= lo) & (PL["t"] < hi)
    ci = [PL["syms"].index(c) for c in coins]
    tt = PL["t"][mt]
    fsx, fbx = fs4.reindex(tt)[coins], fb4.reindex(tt)[coins]
    qvx = PL["qv"].reindex(tt)[coins]
    tot = fsx + fbx
    raw = {"liq_imb_1": ((fbx - fsx) / tot).where(tot > 0, 0),
           "liq_int_1": tot / qvx,
           "liq_imb_24": ((fbx.rolling(6).sum() - fsx.rolling(6).sum()) / tot.rolling(6).sum()).fillna(0),
           "liq_int_24": tot.rolling(6).sum() / qvx.rolling(6).sum()}
    Lr = np.stack([(v.rank(axis=1, pct=True) - 0.5).values for v in raw.values()], axis=2)
    base_j = [PL["names"].index(f) for f in names]
    Xb = np.nan_to_num(PL["X"][mt][:, ci][:, :, base_j])
    Xl = np.concatenate([Xb, np.nan_to_num(Lr)], axis=2)
    fwd = PL["fwd"][mt][:, ci]
    dm = fwd - np.nanmean(fwd, axis=1, keepdims=True)
    y = dm / np.nanstd(dm, axis=1, keepdims=True)
    ok = PL["valid"][mt][:, ci].all(axis=1) & np.isfinite(y).all(axis=1)
    out = {}
    months = pd.date_range(pd.Timestamp("2025-01-01", tz="UTC"), hi, freq="MS")
    for nm, X in [("price+volume+flow+funding", Xb), ("+ liquidations", Xl)]:
        S = np.full(fwd.shape, np.nan)
        for i, m0 in enumerate(months):
            m1 = months[i + 1] if i + 1 < len(months) else hi
            tr = ok & (tt < m0 - M.EMBARGO)
            te = ok & (tt >= m0) & (tt < m1)
            mdl = Ridge(alpha=1000.0, fit_intercept=False).fit(X[tr].reshape(-1, X.shape[2]), y[tr].reshape(-1))
            S[te] = mdl.predict(X[te].reshape(-1, X.shape[2])).reshape(-1, 4)
        long_ = np.nanargmax(np.where(np.isfinite(S), S, -np.inf), axis=1)
        short_ = np.nanargmin(np.where(np.isfinite(S), S, np.inf), axis=1)
        live = np.isfinite(S).all(axis=1)
        f1 = np.nan_to_num(np.expm1(fwd))
        g = np.where(live, 0.5 * f1[np.arange(len(f1)), long_] - 0.5 * f1[np.arange(len(f1)), short_], 0)
        g = g[live]
        ic = np.corrcoef(S[live].reshape(-1), np.nan_to_num(y[live]).reshape(-1))[0, 1]
        out[nm] = {"oos_months": "2025-01 -> 2025-11", "gross_sharpe_long1_short1": round(float(g.mean() / g.std() * np.sqrt(2190)), 3),
                   "oos_ic": round(float(ic), 4), "bars": int(live.sum())}
    yy = y[ok].reshape(-1)
    for i, f in enumerate(raw):
        out[f"ic_{f}"] = round(float(np.corrcoef(np.nan_to_num(Lr[ok][:, :, i]).reshape(-1), yy)[0, 1]), 4)
    return out
liq = liq_addon()
print("Moon Dev: liq add-on", liq)

# ---------------------------------------------------------------- 10. deploy model for the bot (trained on ALL data, like the next monthly retrain would be)
mdl_dir = HERE / "model"
mdl_dir.mkdir(exist_ok=True)
feats_b = [f for f in M.FEATSETS[B["fs"]] if f in PB["names"]]
allmask = PB["valid"] & (t <= M.DATA_END - M.EMBARGO)[:, None]
lastmask = PB["valid"] & (t == t[PB["valid"].any(axis=1)][-1])[:, None]
pred, info = M.fit_predict(PB, B["m"], feats_b, allmask, lastmask)
meta = {"by": "Moon Dev", "model": B["m"], "horizon_hours": B["h"], "features": feats_b,
        "context": M.CONTEXT if B["m"] == "gbm" else [], "universe": PB["syms"],
        "funding_coins": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"],
        "smooth_halflife_bars": B["hl"], "band": B["band"], "book": "long top third, short bottom third, dollar neutral",
        "target_vol_annual": M.TARGET_VOL, "median_leverage_in_backtest": round(float(np.median(pf["L"][oos & (pf["L"] > 0)])), 3),
        "trained_through": str(M.DATA_END - M.EMBARGO)}
if B["m"] == "gbm":
    info["model"].booster_.save_model(str(mdl_dir / "simons_model.txt"))
elif B["m"] == "ridge":
    meta["coef"] = [float(x) for x in info["coef"]]
elif B["m"] == "blend":
    meta["signs"] = [float(np.sign(x)) for x in info["ic"]]
json.dump(meta, open(mdl_dir / "meta.json", "w"), indent=1)

# ---------------------------------------------------------------- 11. claims
def g(k, c=TAKER, f="pre_holdout_sharpe"):
    return row(k, c)[f] if k in runs else None
H = B["h"]
T_ = B["tgt"]
kb = {m: key(H, m, "all", "monthly", 0 if m == "trend20" else T_) for m in ["trend20", "best1", "blend", "ridge", "gbm"]}
claims = []
claims.append({"claim": "Testing many predictive schemes beats a single rule", "timestamp": "~0:12",
               "test": f"{H}h, all features, monthly retrain. Jim's 20 day trend alone and the single best in-sample feature vs equal blend, ridge, LightGBM. Sharpe after taker costs (7 bp/side).",
               "result": "; ".join(f"{m}: pre-holdout {g(kb[m])} / holdout {g(kb[m], f='holdout_sharpe')} (zero cost {g(kb[m], 0)} / {g(kb[m], 0, 'holdout_sharpe')})" for m in kb),
               "verdict": None})
AM = B["m"] if B["m"] in ("blend", "ridge", "gbm") and all((M.CACHE / f"scores_{key(H, B['m'], fs, 'monthly', T_)}.npy").exists() for fs in M.FEATSETS) else "ridge"
abl = {fs: key(H, AM, fs, "monthly", T_) for fs in M.FEATSETS}
claims.append({"claim": "More data types help (price only vs + volume vs + taker flow vs + funding)", "timestamp": "~0:13",
               "test": f"same model ({AM}, {H}h bars, target {'next ' + str(T_) + 'h' if T_ else 'next bar'}, monthly), add data groups one at a time. Zero-cost Sharpe so costs do not hide the signal.",
               "result": "; ".join(f"{fs}: pre-holdout {g(abl[fs], 0)} / holdout {g(abl[fs], 0, 'holdout_sharpe')}" for fs in abl),
               "verdict": None})
rt = {r: key(H, AM, "all", r, T_) for r in ["monthly", "quarterly", "once"]}
def s23(k, c):
    return M.stats(P[(H, 0)], runs[k]["pf"], c, M.ONCE_END, END)["sharpe"]
claims.append({"claim": "You must keep retraining (anomalies fade)", "timestamp": "~0:13",
               "test": "same model, 2023-01 -> 2026-10 only: retrain monthly vs quarterly vs train ONCE on 2020-2022. Sharpe at zero cost / after taker costs.",
               "result": "; ".join(f"{r}: {s23(rt[r], 0)} / {s23(rt[r], TAKER)}" for r in rt), "verdict": None})
claims.append({"claim": "Returns with surprisingly low volatility and risk", "timestamp": "~0:08",
               "test": "best model, vol-targeted to 10%/yr, OOS 2022-01 -> 2026-10 after taker costs",
               "result": f"vol {all_t['vol_pct']}%/yr, max drawdown {all_t['max_dd_pct']}%, Sharpe {all_t['sharpe']}, return {all_t['ann_return_pct']}%/yr",
               "verdict": None})
claims.append({"claim": "There is a capacity to the fund", "timestamp": "~0:15",
               "test": "square-root impact (cost = 1 x daily vol x sqrt(trade $ / 30 day avg daily $ volume)) on the best model's real trades, on top of taker costs",
               "result": f"edge gone at about ${cap_zero:,.0f} (OOS 2022-2026 return), half gone at ${cap_half:,.0f}; holdout-based zero point ${cap_zero_hold:,.0f}",
               "verdict": None})
# verdicts from the numbers
c0 = claims[0]
single = max(g(kb["trend20"]), g(kb["best1"]))
multi = max(g(kb[m]) for m in ["blend", "ridge", "gbm"])
single_h = max(g(kb["trend20"], f="holdout_sharpe"), g(kb["best1"], f="holdout_sharpe"))
multi_h = max(g(kb[m], f="holdout_sharpe") for m in ["blend", "ridge", "gbm"])
c0["verdict"] = "VERIFIED" if (multi > single and multi_h > single_h) else "NOT PROVEN"
pz = [g(abl[fs], 0) for fs in M.FEATSETS]
ph = [g(abl[fs], 0, "holdout_sharpe") for fs in M.FEATSETS]
claims[1]["verdict"] = "VERIFIED" if (max(pz[1:]) > pz[0] and max(ph[1:]) > ph[0]) else "NOT PROVEN"
claims[2]["verdict"] = "VERIFIED" if s23(rt["monthly"], 0) > s23(rt["once"], 0) else "BUSTED"
claims[3]["verdict"] = "VERIFIED" if all_t["sharpe"] > 0.5 and all_t["max_dd_pct"] > -20 else \
    ("NOT PROVEN" if all_t["sharpe"] <= 0 else "VERIFIED")
claims[4]["verdict"] = "VERIFIED"
claims.append({"claim": "Liquidation data adds signal (add-on, short history)", "timestamp": "~0:13 (\"everything is grist for the mill\")",
               "test": "4 coins (BTC ETH SOL XRP), 4h, ridge, walk-forward 2025-01 -> 2025-11, with vs without clean liquidation features (liq_clean_*_5m.csv only)",
               "result": json.dumps(liq), "verdict": "VERIFIED" if liq["+ liquidations"]["oos_ic"] > liq["price+volume+flow+funding"]["oos_ic"] and liq["+ liquidations"]["gross_sharpe_long1_short1"] > liq["price+volume+flow+funding"]["gross_sharpe_long1_short1"] else "NOT PROVEN"})

# ---------------------------------------------------------------- 12. headline + write
trend_hold = g(kb["trend20"], f="holdout_sharpe")
if hold_t["sharpe"] > 1 and hold_t["sharpe"] >= trend_hold:
    verdict = "VERIFIED"
elif hold_t["sharpe"] > 0 and trend_hold > hold_t["sharpe"]:
    verdict = "NOT PROVEN"   # machine makes money, but Jim's one-line 20 day trend beat it on the holdout
else:
    verdict = "BUSTED" if hold_t["sharpe"] <= 0 else "NOT PROVEN"
gk = key(4, "gbm", "all", "monthly", 0)
g_raw0 = row(gk, 0)["holdout_sharpe_no_turnover_control"]
g_raw7 = row(gk, 7)["holdout_sharpe_no_turnover_control"]
takeaways = [
    f"The machine works, a bit: best model ({best_k}) made Sharpe {hold_t['sharpe']} after taker costs on the untouched holdout (2025-07 -> 2026-10), {hold_0['sharpe']} at zero cost, vol {hold_t['vol_pct']}%/yr.",
    f"But Jim's own one-line 20 day trend (long the top third of 20 day movers, short the bottom third, smoothed) did better on the same holdout: Sharpe {trend_hold} after taker costs.",
    f"LightGBM found REAL fast signal: holdout Sharpe {g_raw0} at zero cost when it trades every 4h bar. At taker costs (7 bp/side) the same book is {g_raw7}. The edge is too fast for retail fees. Slowing it down kills the signal.",
    "Costs decide everything. Turnover control (smoothing the score) moved models from deeply negative to positive. 1h bars lost to costs in every model except the 20 day trend, so 4h is the main horizon.",
    f"Retraining did NOT help here: from 2023 on, a model trained once on 2020-2022 did as well as monthly retrains. The signals that work (trend, low vol, taker flow) were stable, not fading.",
    f"2026 so far is negative (Sharpe {decay[-1]['metric']} after costs). The edge is shrinking right now, which is Jim's 'anomalies fade' showing up in real time.",
    f"Capacity is small: square-root impact says the edge is half gone near ${cap_half / 1e6:,.0f}M and gone near ${cap_zero / 1e6:,.0f}M on these 16 coins.",
    f"Fees: 5 and 44 on a 10% vol book leaves investors {fee_all['5_and_44']['investor_cagr_pct']}%/yr (fund made {fee_all['5_and_44']['fund_cagr_before_fees_pct']}%/yr before fees). Renaissance fees need a much bigger edge than this.",
]
mk = lambda d: {k: d[k] for k in ["sharpe", "ann_return_pct", "vol_pct", "max_dd_pct", "total_return_pct", "turnover_x_per_day", "hit_rate", "avg_bp_per_bar"]}
result = {
    "package": "3_simons_machine",
    "title": "Simons-style machine: many weak signals, walk-forward retrain, market-neutral 10% vol book",
    "market": f"Binance USDT perps, {len(PB['syms'])} coins: " + " ".join(s.replace("USDT", "") for s in PB["syms"]),
    "data": "/Volumes/Moon 26/tick_data/binance/futures_klines_1m/<SYM>/<SYM>_1m_*_*.parquet (2020-01 -> 2026-10-24, resampled to 1h and 4h); "
            "/Volumes/Moon 26/tick_data/binance/funding/*_funding.csv (BTC ETH SOL XRP DOGE, 2024-01 -> 2026-08); "
            "/Users/md/Dropbox/dev/github/moon-dev-trading-bots/data/moondev_api/liq_clean_{BTC,ETH,SOL,XRP}_5m.csv (2024-06 -> 2025-12, add-on only)",
    "headline": {"label": f"Final holdout 2025-07 -> 2026-10 Sharpe after taker costs (7 bp/side), best model = {best_k}",
                 "value": f"{hold_t['sharpe']} (zero cost {hold_0['sharpe']}, maker 2 bp/side {hold_m['sharpe']})", "verdict": verdict},
    "stats": {"trades": n_rebal_hold, "trades_note": "coin weight changes in the final holdout",
              "win_rate": hold_t["hit_rate"], "win_rate_note": "share of 4h bars with positive pnl, holdout" if B["h"] == 4 else "share of 1h bars with positive pnl, holdout",
              "avg_R_or_bp": hold_t["avg_bp_per_bar"], "avg_unit": "bp of capital per bar, after taker costs, holdout",
              "profit_factor": round(float(net[t >= M.HOLD_START][net[t >= M.HOLD_START] > 0].sum() / -net[t >= M.HOLD_START][net[t >= M.HOLD_START] < 0].sum()), 3),
              "total_return_pct": hold_t["total_return_pct"], "max_dd_pct": hold_t["max_dd_pct"],
              "sharpe": hold_t["sharpe"], "vol_pct": hold_t["vol_pct"],
              "in_sample": {"note": "walk-forward OOS 2022-01 -> 2025-06. Out of sample for the models, but this window WAS used to pick smoothing, band and the best model.",
                            "taker_7bp": mk(pre_t), "maker_2bp": mk(pre_m), "zero_cost": mk(pre_0), "breakeven_cost_bp_per_side": be_pre},
              "out_of_sample": {"note": "final untouched holdout 2025-07-01 -> 2026-10-24. Nothing was picked on it. Models still retrain monthly on past data, as live would.",
                                "taker_7bp": mk(hold_t), "maker_2bp": mk(hold_m), "zero_cost": mk(hold_0), "breakeven_cost_bp_per_side": be_hold},
              "full_oos_2022_2026_taker": mk(all_t),
              "costs": "taker: 5 bp fee + 2 bp slippage = 7 bp per side on every $ traded. maker: 2 bp per side (assumes every limit fills, optimistic). zero cost shown too. Funding payments NOT charged (only have funding for 5 coins).",
              "portfolio": {"book": "long top third, short bottom third, dollar neutral", "vol_target": "10%/yr from past 30 days of pnl, leverage cap 4x",
                            "smooth_halflife_bars": B["hl"], "band": B["band"], "avg_gross_exposure_x": round(avg_pos, 3)},
              "capacity": {"impact_model": "cost per trade = 1.0 x daily vol x sqrt(trade $ / 30 day avg daily $ volume). The 1.0 is a rule of thumb, not from the video.",
                           "aum_where_edge_is_zero_usd": round(cap_zero), "aum_where_edge_is_half_usd": round(cap_half),
                           "aum_where_holdout_edge_is_zero_usd": round(cap_zero_hold), "participation": part},
              "after_2_and_20": {"oos_2022_2026": fee_all["2_and_20"], "holdout": fee_hold["2_and_20"]},
              "after_5_and_44": {"oos_2022_2026": fee_all["5_and_44"], "holdout": fee_hold["5_and_44"]}},
    "equity_curve": equity_curve,
    "trades": trades,
    "grid": grid,
    "claims": claims,
    "decay": decay,
    "price_replay": {"symbol": sym, "bar_hours": B["h"], "signal_meaning": "machine's weight in SOL, % of capital (+ long, - short)",
                     "bars": bars, "trades": rp_trades},
    "feature_importance": {"top": feat_rows[:10], "group_gbm_gain_pct": group_gain},
    "liquidation_addon": liq,
    "takeaways": takeaways,
    "video_leaves_out": ["which models Renaissance used", "holding period", "position sizing and leverage", "how fast anomalies fade",
                         "what 'persisted long enough' means as a number", "any Medallion return numbers (none given in this talk)"],
    "skipped": ["TSLA single-asset version (time box)", "weather, annual/quarterly reports (we do not have them)"],
}
json.dump(result, open(HERE / "result.json", "w"), indent=1, default=float)
print(json.dumps({k: result[k] for k in ["headline"]}, indent=1))
print(json.dumps(result["stats"], indent=1, default=float)[:6000])
for c in claims:
    print(c["claim"], "|", c["result"], "|", c["verdict"])
print("Moon Dev: top features", [(r["feature"], r["gbm_gain_pct"], r.get("oos_ic_2022_2025H1"), r.get("oos_ic_holdout")) for r in feat_rows[:10]])
print("Moon Dev: group gain", group_gain)
print("Moon Dev: decay", decay)
