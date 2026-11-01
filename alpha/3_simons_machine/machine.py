"""
Moon Dev's Simons machine (package 3).
Jim Simons ~0:12: "look at a lot of data and you try to simulate different predictive schemes until you get better and better at it."
Many weak features -> simple models -> walk-forward monthly retrain -> market-neutral, vol-targeted long/short book.

Lookahead rules (Moon Dev):
  * every bar is stamped at its CLOSE. A feature at t only uses bars stamped <= t.
  * features are cross-sectional ranks at the same t (no scaling fit on future data).
  * target = next bar log return minus the cross-section mean (market neutral), / cross-section std.
  * a model used in month M is trained only on rows with t < start(M) - 1 day (embargo).
  * portfolio settings (smoothing, band) and the "best model" are picked on 2022-01 -> 2025-06 only.
    2025-07-01 -> 2026-10-24 is the final holdout, never used to pick anything.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")   # first walk-forward test month
ONCE_END = pd.Timestamp("2023-01-01", tz="UTC")    # train-once model: fit on 2020-2022, never again
HOLD_START = pd.Timestamp("2025-07-01", tz="UTC")  # final untouched holdout
DATA_END = pd.Timestamp("2026-10-24 23:00", tz="UTC")
TARGET_VOL = 0.10                                   # 10% a year, from the task (video gives no number)
EMBARGO = pd.Timedelta(days=1)
N_JOBS = 5

PRICE = ["r_1", "r_4", "r_12", "r_24", "r_72", "r_168", "r_240", "r_360", "r_480",
         "vol_24", "vol_168", "range_1", "resid_24", "resid_168", "dist_high_480", "sharpe_480"]
VOLUME = ["qv_z", "qv1d_z", "trades_z", "size_z"]
FLOW = ["tbi_1", "tbi_24", "tbi_168", "tbi_dev"]
FUND = ["fund_last", "fund_3d"]
CONTEXT = ["hour", "btc_r_1", "btc_r_24", "mkt_vol_24", "mkt_disp"]  # same for every coin, GBM only
FEATSETS = {"price": PRICE, "price+volume": PRICE + VOLUME, "price+volume+flow": PRICE + VOLUME + FLOW,
            "all": PRICE + VOLUME + FLOW + FUND}


# ---------------------------------------------------------------- data + features
def load_wide(bar_hours):
    b = pd.read_parquet(CACHE / "bars_1h.parquet")
    return wide_from_bars(b[b.t <= DATA_END], bar_hours, DATA_END)


def wide_from_bars(b, bar_hours, end):
    """long hourly bars (t = close time, sym, ohlc, quote_volume, trades, taker_buy_quote_volume) -> wide frames."""
    cols = ["open", "high", "low", "close", "quote_volume", "trades", "taker_buy_quote_volume"]
    idx = pd.date_range(b.t.min(), end, freq="1h")
    W = {c: b.pivot(index="t", columns="sym", values=c).reindex(idx) for c in cols}
    if bar_hours > 1:  # 4h bar closing at 04:00 = hourly bars stamped 01,02,03,04
        n = W["close"].resample(f"{bar_hours}h", label="right", closed="right").count()
        agg = {"open": "first", "high": "max", "low": "min", "close": "last",
               "quote_volume": "sum", "trades": "sum", "taker_buy_quote_volume": "sum"}
        W = {c: getattr(W[c].resample(f"{bar_hours}h", label="right", closed="right"), agg[c])().where(n == bar_hours)
             for c in cols}
    return W


def features(W, bar_hours, fu=None):
    c, h, l = W["close"], W["high"], W["low"]
    qv, tr, tb = W["quote_volume"], W["trades"], W["taker_buy_quote_volume"]
    per_day = 24 // bar_hours
    lr = np.log(c).diff()
    F = {}
    for hrs in [1, 4, 12, 24, 72, 168, 240, 360, 480]:  # 480h = Jim's 20 day trend
        k = hrs // bar_hours
        if hrs >= bar_hours:
            F[f"r_{hrs}" if hrs != bar_hours else "r_1"] = np.log(c / c.shift(k))
    for hrs in [24, 168]:
        k = max(hrs // bar_hours, 4)
        F[f"vol_{hrs}"] = lr.rolling(k, min_periods=int(k * 0.8)).std() * np.sqrt(per_day)
    F["range_1"] = np.log(h / l)
    btc = lr["BTCUSDT"]
    k30 = 30 * per_day
    beta = lr.rolling(k30, min_periods=k30 // 2).cov(btc).div(btc.rolling(k30, min_periods=k30 // 2).var(), axis=0)
    for hrs in [24, 168]:
        k = hrs // bar_hours
        F[f"resid_{hrs}"] = np.log(c / c.shift(k)) - beta.mul(np.log(c["BTCUSDT"] / c["BTCUSDT"].shift(k)), axis=0)
    k20 = 480 // bar_hours
    F["dist_high_480"] = np.log(c / h.rolling(k20, min_periods=k20).max())
    F["sharpe_480"] = F["r_480"] / (lr.rolling(k20, min_periods=k20).std() * np.sqrt(k20))

    def tz(x, k):  # z-score vs the PAST k bars only (shift 1)
        m, s = x.shift(1).rolling(k, min_periods=k // 2).mean(), x.shift(1).rolling(k, min_periods=k // 2).std()
        return (x - m) / s
    k7 = 7 * per_day
    lq = np.log(qv.where(qv > 0))
    F["qv_z"] = tz(lq, k7)
    F["qv1d_z"] = tz(np.log(qv.rolling(per_day).sum().where(qv > 0)), 20 * per_day)
    F["trades_z"] = tz(np.log(tr.where(tr > 0)), k7)
    F["size_z"] = tz(np.log((qv / tr).where(tr > 0)), k7)
    F["tbi_1"] = 2 * tb / qv - 1
    F["tbi_24"] = 2 * tb.rolling(per_day).sum() / qv.rolling(per_day).sum() - 1
    F["tbi_168"] = 2 * tb.rolling(k7).sum() / qv.rolling(k7).sum() - 1
    F["tbi_dev"] = F["tbi_1"] - F["tbi_168"]

    fu = pd.read_parquet(CACHE / "funding.parquet") if fu is None else fu
    fw = fu.pivot_table(index="ts", columns="sym", values="rate").sort_index()
    f3 = fw.rolling(9, min_periods=9).mean()  # last 9 prints = 3 days
    grid = c.index
    F["fund_last"] = fw.reindex(fw.index.union(grid)).ffill().reindex(grid).reindex(columns=c.columns)
    F["fund_3d"] = f3.reindex(f3.index.union(grid)).ffill().reindex(grid).reindex(columns=c.columns)
    # funding only exists 2024-01 -> 2026-08 for 5 coins. After the file ends we stop using it (no stale fill).
    stale = grid > fw.index.max() + pd.Timedelta(hours=8)
    F["fund_last"].loc[stale] = np.nan
    F["fund_3d"].loc[stale] = np.nan

    valid = c.notna() & F["r_480"].notna() & (qv > 0) & F["vol_168"].notna()
    R = {k: (v.where(valid).rank(axis=1, pct=True) - 0.5) for k, v in F.items()}  # cross-sectional ranks
    # context (same for every coin at time t)
    ctx = {"hour": pd.Series(c.index.hour, index=c.index).astype(float),
           "btc_r_1": lr["BTCUSDT"], "btc_r_24": np.log(c["BTCUSDT"] / c["BTCUSDT"].shift(per_day)),
           "mkt_vol_24": F["vol_24"].where(valid).mean(axis=1), "mkt_disp": F["r_1"].where(valid).std(axis=1)}
    fwd = lr.shift(-1)  # next bar log return (TARGET only, never a feature)
    dm = fwd.where(valid).sub(fwd.where(valid).mean(axis=1), axis=0)
    yz = dm.div(dm.std(axis=1), axis=0).clip(-5, 5)
    return F, R, ctx, valid, fwd, yz


def panel(bar_hours, target_hours=0):
    W = load_wide(bar_hours)
    F, R, ctx, valid, fwd, yz = features(W, bar_hours)
    names = list(R.keys())
    t = valid.index
    syms = list(valid.columns)
    X = np.stack([R[n].reindex(columns=syms).values for n in names], axis=2).astype(np.float32)  # T,N,K
    C = np.stack([ctx[n].reindex(t).values for n in CONTEXT], axis=1).astype(np.float32)
    P = dict(t=t, syms=syms, names=names, X=X, C=C, valid=valid.values & (valid.sum(axis=1).values >= 6)[:, None],
             fwd=fwd.reindex(columns=syms).values, yz=yz.reindex(columns=syms).values, close=W["close"][syms],
             qv=W["quote_volume"][syms], W=W, F=F, bar_hours=bar_hours, per_year=8760 / bar_hours)
    if target_hours:  # slower target: next target_hours return (still traded every bar). embargo of 1 day covers it.
        assert target_hours <= 24
        c = W["close"][syms]
        f = np.log(c.shift(-(target_hours // bar_hours)) / c).where(valid[syms])
        dm = f.sub(f.mean(axis=1), axis=0)
        P["yz"] = dm.div(dm.std(axis=1), axis=0).clip(-5, 5).values
    return P


# ---------------------------------------------------------------- models
def fit_predict(P, model, feats, train_mask, test_mask):
    """train on rows in train_mask (T,N bool), predict scores for test_mask rows. returns (pred, info)."""
    j = [P["names"].index(f) for f in feats]
    X = np.nan_to_num(P["X"][:, :, j])
    y = P["yz"]
    tr = train_mask & np.isfinite(y)
    Xtr, ytr = X[tr], y[tr]
    Xte = X[test_mask]
    if model == "trend20":   # Jim's 20 day trend, one rule, no fitting
        return Xte[:, feats.index("r_480")], {}
    sd = Xtr.std(axis=0)
    ic = np.array([np.corrcoef(Xtr[:, i], ytr)[0, 1] if sd[i] > 0 else 0.0 for i in range(len(j))])  # feature with no data yet = 0
    if model == "best1":     # the single best feature in-sample
        i = int(np.nanargmax(np.abs(ic)))
        return np.sign(ic[i]) * Xte[:, i], {"ic": ic, "pick": feats[i]}
    if model == "blend":     # equal weight, sign from in-sample IC
        return Xte @ np.sign(ic) / len(j), {"ic": ic}
    if model == "ridge":
        m = Ridge(alpha=1000.0, fit_intercept=False).fit(Xtr, ytr)
        return m.predict(Xte), {"ic": ic, "coef": m.coef_}
    if model == "gbm":
        Ctr = P["C"][np.where(tr)[0]]
        Cte = P["C"][np.where(test_mask)[0]]
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=2000,
                              subsample=0.5, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                              verbose=-1, n_jobs=N_JOBS, random_state=7)
        m.fit(np.hstack([Xtr, Ctr]), ytr)
        return m.predict(np.hstack([Xte, Cte])), {"ic": ic, "gain": m.booster_.feature_importance("gain"), "model": m}
    raise ValueError(model)


def walk_forward(P, model, feats, retrain="monthly", keep_last=False):
    """returns wide score array (T,N) with NaN outside OOS, plus per-fit info."""
    t = P["t"]
    feats = [f for f in feats if f in P["names"]]  # at 4h bars there is no separate r_4 (it IS r_1)
    S = np.full(P["valid"].shape, np.nan)
    infos = []
    if retrain == "once":
        tr = P["valid"] & (t < ONCE_END - EMBARGO)[:, None]
        te = P["valid"] & (t >= ONCE_END)[:, None]
        pred, info = fit_predict(P, model, feats, tr, te)
        S[te] = pred
        return S, [info]
    step = {"monthly": 1, "quarterly": 3}[retrain]
    starts = pd.date_range(OOS_START, DATA_END, freq=f"{step}MS")
    for i, m0 in enumerate(starts):
        m1 = starts[i + 1] if i + 1 < len(starts) else DATA_END + pd.Timedelta(hours=1)
        tr = P["valid"] & (t < m0 - EMBARGO)[:, None]
        te = P["valid"] & ((t >= m0) & (t < m1))[:, None]
        pred, info = fit_predict(P, model, feats, tr, te)
        S[te] = pred
        if not (keep_last and i == len(starts) - 1):
            info.pop("model", None)
        infos.append(info)
    return S, infos


# ---------------------------------------------------------------- portfolio
def portfolio(P, S, halflife=0, band=0.0, min_n=6):
    """long top third / short bottom third, dollar neutral, vol-targeted to 10%/yr, EMA smoothing + no-trade band.
    returns dict of per-bar arrays: W (weights held from t to t+1), gross pnl, turnover."""
    Sdf = pd.DataFrame(S)
    if halflife > 0:
        Sdf = Sdf.ewm(halflife=halflife, ignore_na=True).mean().where(Sdf.notna())
    Sv = Sdf.values
    T, N = Sv.shape
    r = Sdf.rank(axis=1).values
    n = np.isfinite(Sv).sum(axis=1)
    k = np.maximum(n // 3, 1)
    w = np.zeros((T, N))
    long_ = r > (n - k)[:, None]
    short_ = r <= k[:, None]
    w[long_] = (0.5 / np.repeat(k[:, None], N, 1))[long_]
    w[short_] = -(0.5 / np.repeat(k[:, None], N, 1))[short_]
    w[n < min_n] = 0
    fwd = np.nan_to_num(np.expm1(P["fwd"]))
    g = (w * fwd).sum(axis=1)                          # unit-gross pnl, realized at t+1
    gs = pd.Series(g).where(n >= min_n)
    lb = int(30 * 24 / P["bar_hours"])
    sd = gs.shift(1).rolling(lb, min_periods=lb // 3).std().values  # only pnl already realized at t
    L = np.clip((TARGET_VOL / np.sqrt(P["per_year"])) / sd, 0, 4.0)
    L = np.nan_to_num(L)
    tgt = w * L[:, None]
    Wt = np.zeros_like(tgt)
    prev = np.zeros(N)
    slot = 0.5 / k * L
    for i in range(T):
        d = tgt[i] - prev
        move = (np.abs(d) > band * slot[i]) | (tgt[i] == 0)
        prev = np.where(move, tgt[i], prev)
        Wt[i] = prev
    turn = np.abs(np.diff(np.vstack([np.zeros(N), Wt]), axis=0)).sum(axis=1)
    pnl = (Wt * fwd).sum(axis=1)
    return {"W": Wt, "gross": pnl, "turn": turn, "L": L}


def stats(P, pf, cost_bp, start, end):
    t = P["t"]
    m = (t >= start) & (t < end)
    net = pf["gross"][m] - cost_bp / 1e4 * pf["turn"][m]
    py = P["per_year"]
    sd = net.std()
    eq = np.cumprod(1 + net)
    dd = eq / np.maximum.accumulate(eq) - 1
    act = net[np.abs(pf["W"][m]).sum(axis=1) > 0]
    return {"sharpe": round(float(net.mean() / sd * np.sqrt(py)) if sd > 0 else 0.0, 3),
            "ann_return_pct": round(float(net.mean() * py * 100), 2),
            "vol_pct": round(float(sd * np.sqrt(py) * 100), 2),
            "max_dd_pct": round(float(dd.min() * 100), 2),
            "total_return_pct": round(float((eq[-1] - 1) * 100), 2) if len(eq) else 0,
            "turnover_x_per_day": round(float(pf["turn"][m].mean() * 24 / P["bar_hours"]), 3),
            "hit_rate": round(float((act > 0).mean()), 4) if len(act) else 0,
            "avg_bp_per_bar": round(float(net.mean() * 1e4), 3)}


def per_year(P, pf, cost_bp, start):
    out = []
    for y in range(start.year, 2027):
        a, b = max(pd.Timestamp(f"{y}-01-01", tz="UTC"), start), pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
        out.append({"period": str(y), "metric": stats(P, pf, cost_bp, a, b)["sharpe"]})
    return out
