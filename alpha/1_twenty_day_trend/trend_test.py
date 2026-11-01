"""
Moon Dev: Jim Simons' 20 day trend, tested on real data.

Jim (TED, 0:10): "if you decided, okay, I'm going to predict today by the average move in the past 20 days...
years ago such a system would work, not beautifully but it would work." Then he tried 10 day, 15 day,
"see what worked best", and (0:11) it faded "because everyone could see that."

The video leaves out the exact rule, so every version here is labeled:
  sign_ls   : position = sign(mean of last N daily returns), long or short, 1x notional    (Moon Dev's literal read)
  long_only : position = 1 if mean > 0 else flat                                            (Moon Dev variant)
  sized     : position = t-stat of the N day mean / 2, capped at +-1                        (Moon Dev variant: "size by the signal",
              t-stat = mean / (60 day std / sqrt(N)), so it is vol-scaled)
Decide at the close of day t, hold to the close of day t+1, rebalance every day.

Run build_bars.py first (it builds cache/).
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
SOURCES = json.load(open(f"{CACHE}/sources.json"))

N_ALL = list(range(2, 251))
N_GRID = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250]
JIM_N = [10, 15, 20]
VARIANTS = ["sign_ls", "long_only", "sized"]
TGT_VOL = 0.10      # Moon Dev choice: 10% a year risk per market in the portfolio (video gives no sizing)
MAX_LEV = 4.0       # Moon Dev choice: cap on the vol-target weight
VOL_LOOK = 60       # Moon Dev choice: 60 bar lookback for vol

CRYPTO_COST = 0.0005 + 0.0002          # Binance taker 0.05% + 0.02% slippage, per side, per unit of position change
FUT_COST = 0.0001                      # Moon Dev assumption for Yahoo futures / FX: 1 bp per side (video gives none)
STOCK_BP, STOCK_CENTS = 0.0001, 0.005  # TSLA / SPY: 1 bp + half a cent per share, per side

# Moon Dev: in-sample / out-of-sample splits (picked so each side has years of data)
SPLITS = {"crypto_perps": "2024-01-01", "crypto_spot_long": "2024-01-01", "commodities": "2013-01-01",
          "fx": "2013-01-01", "index_futures": "2013-01-01", "stocks_SPY": "2018-01-01", "stocks_TSLA": "2025-07-01",
          "all_markets": "2018-01-01"}


# ----------------------------------------------------------------------------- data
def load_markets():
    mk = {}
    for name, s in SOURCES.items():
        g = s["group"]
        if g == "crypto_perps":
            d = pd.read_parquet(f"{CACHE}/perp_{name.split('-')[0]}_1d.parquet")
        elif g == "crypto_spot_long":
            d = pd.read_parquet(f"{CACHE}/spot_{name.split('-')[0]}_1d.parquet")
        else:
            d = pd.read_parquet(f"{CACHE}/{s['file']}")
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        if g == "stocks":
            g = f"stocks_{name}"
        cost = "crypto" if g.startswith("crypto") else ("stock" if g.startswith("stocks") else "fut")
        mk[name] = dict(df=d, group=g, cost=cost, path=s["path"], start=s["start"], end=s["end"])
    print(f"🌙 Moon Dev: loaded {len(mk)} markets")
    return mk


def ppy_of(index):
    yrs = (index[-1] - index[0]).days / 365.25
    return len(index) / yrs


# ----------------------------------------------------------------------------- core rule
def positions(r, n, variant):
    """Moon Dev: position at bar t, from returns up to and including bar t."""
    m = r.rolling(n).mean()
    if variant == "sign_ls":
        p = np.sign(m)
    elif variant == "long_only":
        p = (m > 0).astype(float).where(m.notna())
    else:
        z = m / (r.rolling(VOL_LOOK).std() / np.sqrt(n))
        p = (z / 2).clip(-1, 1)
    return p


def cost_per_unit(mkt, close):
    if mkt["cost"] == "crypto":
        return pd.Series(CRYPTO_COST, index=close.index)
    if mkt["cost"] == "fut":
        return pd.Series(FUT_COST, index=close.index)
    return STOCK_BP + STOCK_CENTS / close


def run(mkt, n, variant, close=None, vol_target=False):
    """Moon Dev: returns a DataFrame with pos (held over the NEXT bar), gross and net pnl, indexed by the bar the
    pnl is earned on."""
    c = mkt["df"]["close"] if close is None else close
    r = c.pct_change()
    p = positions(r, n, variant)
    if vol_target:
        ppy = ppy_of(c.index)
        w = (TGT_VOL / (r.rolling(VOL_LOOK).std() * np.sqrt(ppy))).clip(upper=MAX_LEV)
        p = p * w
    p = p.fillna(0.0)
    held = p.shift(1)                              # position decided at t-1 close, earns r_t
    gross = held * r
    turn = (p - p.shift(1)).abs()                  # trade at the close of t-1 -> charged on bar t
    cpu = cost_per_unit(mkt, c)
    net = gross - (turn * cpu).shift(1)
    out = pd.DataFrame({"pos": held, "r": r, "gross": gross, "net": net}).iloc[1:]
    first = p.ne(0).idxmax() if p.ne(0).any() else p.index[-1]
    return out.loc[out.index > first]


def sharpe(x, ppy):
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(ppy))


def max_dd(pnl):
    eq = (1 + pnl.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    return float(dd.min())


def longest_dd_days(pnl):
    eq = (1 + pnl.fillna(0)).cumprod()
    peak = eq.cummax()
    under = eq < peak
    best, start = 0, None
    for t, u in under.items():
        if u and start is None:
            start = t
        if not u and start is not None:
            best = max(best, (t - start).days)
            start = None
    if start is not None:
        best = max(best, (eq.index[-1] - start).days)
    return int(best)


def summ(df, col, ppy, split):
    x = df[col]
    i, o = x[x.index < split], x[x.index >= split]
    return dict(sharpe=sharpe(x, ppy), is_sharpe=sharpe(i, ppy), oos_sharpe=sharpe(o, ppy),
                total_return_pct=float(((1 + x).prod() - 1) * 100), max_dd_pct=float(max_dd(x) * 100),
                vol_pct=float(x.std() * np.sqrt(ppy) * 100))


# ----------------------------------------------------------------------------- portfolios
def portfolio(mk, names, n, variant, col="net"):
    """Moon Dev: equal-risk portfolio. Each market vol-targeted to 10%/yr, then summed / number of markets that
    are live on that date (a market is live from its first signal to its last bar, weekends included)."""
    pnls = {nm: run(mk[nm], n, variant, vol_target=True)[["gross", "net"]] for nm in names}
    idx = pd.DatetimeIndex(sorted(set().union(*[s.index for s in pnls.values()])))
    tot = pd.DataFrame(0.0, index=idx, columns=["gross", "net"])
    live = pd.Series(0.0, index=idx)
    for s in pnls.values():
        tot = tot + s.reindex(idx).fillna(0.0)
        live += ((idx >= s.index[0]) & (idx <= s.index[-1])).astype(float)
    res = tot.div(live.replace(0, np.nan), axis=0).dropna()
    return res if col == "both" else res[col]


def yearly_sharpe(x):
    out = {}
    for y, g in x.groupby(x.index.year):
        if len(g) > 100:
            out[int(y)] = sharpe(g, ppy_of(g.index) if len(g) > 1 else 252)
    return out


def slope(yd):
    ys = np.array(sorted(yd)); v = np.array([yd[y] for y in ys])
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return dict(slope_per_year=np.nan, p_value=np.nan, years=int(ok.sum()))
    lr = sps.linregress(ys[ok], v[ok])
    return dict(slope_per_year=float(lr.slope), p_value=float(lr.pvalue), years=int(ok.sum()),
                first_year=int(ys[ok][0]), last_year=int(ys[ok][-1]))


def apply_fees(pnl, mgmt, perf):
    """Moon Dev: investor NAV after an annual management fee and an incentive fee over a high-water mark."""
    nav, hwm, out = 1.0, 1.0, []
    for y, g in pnl.groupby(pnl.index.year):
        start = nav
        nav = start * float((1 + g).prod())
        frac = min(1.0, (g.index[-1] - g.index[0]).days / 365.0 + 1 / 365)
        nav -= mgmt * start * frac
        if nav > hwm:
            nav -= perf * (nav - hwm)
        hwm = max(hwm, nav)
        out.append((y, nav))
    yrs = (pnl.index[-1] - pnl.index[0]).days / 365.25
    return dict(final_nav=float(nav), total_return_pct=float((nav - 1) * 100),
                cagr_pct=float((nav ** (1 / yrs) - 1) * 100), years=round(yrs, 2))


# ----------------------------------------------------------------------------- main
def main():
    mk = load_markets()
    groups = {}
    for nm, m in mk.items():
        groups.setdefault(m["group"], []).append(nm)
    groups["all_markets"] = groups["crypto_perps"] + groups["commodities"] + groups["fx"] + groups["index_futures"] + groups["stocks_SPY"]
    print("🌙 Moon Dev groups:", {g: len(v) for g, v in groups.items()})

    # ---------- 1. per-market sweep (N grid, all variants, 1x notional)
    grid, per_market_sel = [], []
    for nm, m in mk.items():
        ppy = ppy_of(m["df"].index)
        split = SPLITS[m["group"]]
        best = None
        for n in N_ALL:
            for v in VARIANTS:
                if v != "sign_ls" and n not in N_GRID:
                    continue
                df = run(m, n, v)
                g, ne = summ(df, "gross", ppy, split), summ(df, "net", ppy, split)
                if v == "sign_ls" and (best is None or (ne["is_sharpe"] or -9) > (best[1] or -9)):
                    best = (n, ne["is_sharpe"], ne["oos_sharpe"])
                if n in N_GRID:
                    held = df["pos"]
                    act = df[held != 0]
                    grid.append(dict(params=dict(market=nm, group=m["group"], N=n, variant=v, timeframe="1d"),
                                     sharpe_gross=g["sharpe"], sharpe_net=ne["sharpe"],
                                     is_sharpe_net=ne["is_sharpe"], oos_sharpe_net=ne["oos_sharpe"],
                                     is_sharpe_gross=g["is_sharpe"], oos_sharpe_gross=g["oos_sharpe"],
                                     total_return_pct_net=ne["total_return_pct"], max_dd_pct_net=ne["max_dd_pct"],
                                     win_rate_days=float((act["net"] > 0).mean() * 100) if len(act) else np.nan,
                                     position_changes=int((held.diff().abs() > 0).sum())))
        n20 = run(m, 20, "sign_ls")
        per_market_sel.append(dict(market=nm, group=m["group"], best_is_N=best[0], best_is_sharpe=best[1],
                                   its_oos_sharpe=best[2], n20_oos_sharpe=summ(n20, "net", ppy, split)["oos_sharpe"],
                                   buyhold_sharpe=sharpe(n20["r"], ppy)))
        print(f"🌙 Moon Dev {nm}: best IS N={best[0]} IS {best[1]:.2f} -> OOS {best[2]:.2f} | N=20 OOS {per_market_sel[-1]['n20_oos_sharpe']:.2f}")

    # ---------- 2. group portfolio sweep, N 2..250
    n_sweep, group_sel = [], {}
    for gname in ["crypto_perps", "crypto_spot_long", "commodities", "fx", "index_futures", "stocks_SPY", "stocks_TSLA", "all_markets"]:
        names = groups[gname]
        split = SPLITS[gname]
        rows = []
        for v in VARIANTS:
            for n in N_ALL:
                if v != "sign_ls" and n not in N_GRID:
                    continue
                pb = portfolio(mk, names, n, v, "both")
                pg, pn = pb["gross"], pb["net"]
                ppy = ppy_of(pn.index)
                row = dict(group=gname, variant=v, N=n,
                           is_sharpe_gross=sharpe(pg[pg.index < split], ppy), oos_sharpe_gross=sharpe(pg[pg.index >= split], ppy),
                           is_sharpe_net=sharpe(pn[pn.index < split], ppy), oos_sharpe_net=sharpe(pn[pn.index >= split], ppy),
                           sharpe_net=sharpe(pn, ppy), vol_pct=float(pn.std() * np.sqrt(ppy) * 100),
                           max_dd_pct=float(max_dd(pn) * 100))
                rows.append(row)
        rd = pd.DataFrame(rows)
        n_sweep += rows
        sel = {}
        for v in VARIANTS:
            s = rd[rd.variant == v].dropna(subset=["is_sharpe_net"])
            b = s.loc[s.is_sharpe_net.idxmax()]
            r20 = s[s.N == 20].iloc[0]
            ob = s.loc[s.oos_sharpe_net.idxmax()]
            sel[v] = dict(best_is_N=int(b.N), best_is_sharpe_net=float(b.is_sharpe_net), its_oos_sharpe_net=float(b.oos_sharpe_net),
                          n20_is_sharpe_net=float(r20.is_sharpe_net), n20_oos_sharpe_net=float(r20.oos_sharpe_net),
                          n20_is_sharpe_gross=float(r20.is_sharpe_gross), n20_oos_sharpe_gross=float(r20.oos_sharpe_gross),
                          hindsight_best_oos_N=int(ob.N), hindsight_best_oos_sharpe_net=float(ob.oos_sharpe_net),
                          overstatement_sharpe=float(b.is_sharpe_net - b.oos_sharpe_net))
        group_sel[gname] = dict(split=split, markets=names, **sel)
        print(f"🌙 Moon Dev group {gname}: {json.dumps(sel['sign_ls'])}")

    # ---------- 3. shorter bars (4h / 1h) on crypto perps, sign rule, N in bars
    tf_rows = []
    for tf, bars_per_day in [("4h", 6), ("1h", 24)]:
        for n in [2, 3, 5, 10, 15, 20, 30, 60, 120, 240, 480]:
            tot = {}
            for nm in groups["crypto_perps"]:
                d = pd.read_parquet(f"{CACHE}/perp_{nm.split('-')[0]}_{tf}.parquet")
                d.index = d.index.tz_localize(None)
                m = dict(mk[nm]); m["df"] = d
                df = run(m, n, "sign_ls", vol_target=True)
                tot[nm] = df
            idx = pd.DatetimeIndex(sorted(set().union(*[x.index for x in tot.values()])))
            res = {}
            for col in ["gross", "net"]:
                s = sum(x[col].reindex(idx).fillna(0) for x in tot.values())
                live = sum(((idx >= x.index[0]) & (idx <= x.index[-1])).astype(float) for x in tot.values())
                res[col] = (s / pd.Series(live, index=idx).replace(0, np.nan)).dropna()
            ppy = 365 * bars_per_day
            sp = SPLITS["crypto_perps"]
            row = dict(params=dict(market="crypto_perps portfolio", group="crypto_perps", N=n, variant="sign_ls", timeframe=tf),
                       N_in_days=round(n / bars_per_day, 2),
                       is_sharpe_gross=sharpe(res["gross"][res["gross"].index < sp], ppy), oos_sharpe_gross=sharpe(res["gross"][res["gross"].index >= sp], ppy),
                       is_sharpe_net=sharpe(res["net"][res["net"].index < sp], ppy), oos_sharpe_net=sharpe(res["net"][res["net"].index >= sp], ppy),
                       sharpe_net=sharpe(res["net"], ppy), sharpe_gross=sharpe(res["gross"], ppy))
            tf_rows.append(row)
            print(f"🌙 Moon Dev {tf} N={n}: gross IS {row['is_sharpe_gross']:.2f} OOS {row['oos_sharpe_gross']:.2f} | net IS {row['is_sharpe_net']:.2f} OOS {row['oos_sharpe_net']:.2f}")

    # ---------- 4. decay: yearly Sharpe of the 20 day sign rule
    decay, decay_fit = [], {}
    for nm, m in mk.items():
        ys = yearly_sharpe(run(m, 20, "sign_ls")["gross"])
        for y, v in ys.items():
            decay.append(dict(period=str(y), metric=v, level="market", name=nm, group=m["group"],
                              label="20d sign_ls Sharpe, zero cost"))
        decay_fit[nm] = slope(ys)
    for gname in ["crypto_perps", "crypto_spot_long", "commodities", "fx", "index_futures", "stocks_SPY", "all_markets"]:
        p = portfolio(mk, groups[gname], 20, "sign_ls", "gross")
        ys = yearly_sharpe(p)
        for y, v in ys.items():
            decay.append(dict(period=str(y), metric=v, level="group", name=gname, group=gname,
                              label="20d sign_ls equal-risk portfolio Sharpe, zero cost"))
        decay_fit[gname] = slope(ys)
        print(f"🌙 Moon Dev decay {gname}: {decay_fit[gname]}")

    # ---------- 5. headline portfolio, stats, fees
    out = {}
    for gname in ["all_markets", "crypto_perps", "commodities", "fx", "index_futures"]:
        for n in sorted(set([20, group_sel[gname]["sign_ls"]["best_is_N"]])):
            for v in VARIANTS:
                pn = portfolio(mk, groups[gname], n, v, "net")
                out[(gname, n, v)] = pn
    # Moon Dev: "best OOS version" among the honest candidates (N=20 or the IS-picked N)
    cand = []
    for (gname, n, v), pn in out.items():
        sp = SPLITS[gname]
        cand.append((sharpe(pn[pn.index >= sp], ppy_of(pn.index)), gname, n, v))
    cand.sort(key=lambda x: -np.nan_to_num(x[0], nan=-9))
    best_oos = cand[0]
    print(f"🌙 Moon Dev best OOS candidate: {best_oos}")
    pn = out[(best_oos[1], best_oos[2], best_oos[3])]
    sp = SPLITS[best_oos[1]]
    is_part, oos_part = pn[pn.index < sp], pn[pn.index >= sp]
    lev = TGT_VOL / (is_part.std() * np.sqrt(ppy_of(pn.index)))      # scale to 10%/yr using IS vol only
    fee_base = oos_part * lev
    fees = dict(version=f"{best_oos[1]} N={best_oos[2]} {best_oos[3]}, net of trading costs, OOS {sp} on, scaled to 10% vol with IS vol (x{lev:.2f})",
                before_fees=apply_fees(fee_base, 0, 0), after_2_and_20=apply_fees(fee_base, 0.02, 0.20),
                after_5_and_44=apply_fees(fee_base, 0.05, 0.44))
    print(f"🌙 Moon Dev fees: {json.dumps(fees, indent=1)}")

    # ---------- 6. claim (a): does the 20 day trend predict the next day? (pooled, gross, per group)
    pred = {}
    for gname in ["crypto_perps", "crypto_spot_long", "commodities", "fx", "index_futures", "stocks_SPY", "stocks_TSLA"]:
        ics, pnl = [], []
        for nm in groups[gname]:
            c = mk[nm]["df"]["close"]; r = c.pct_change()
            m20 = r.rolling(20).mean()
            z = pd.DataFrame({"s": m20, "f": r.shift(-1)}).dropna()
            ics.append(sps.spearmanr(z.s, z.f)[0])
            pnl.append(run(mk[nm], 20, "sign_ls")["gross"] / r.std())
        allp = pd.concat(pnl)
        t = allp.mean() / allp.std() * np.sqrt(len(allp))
        pred[gname] = dict(mean_rank_ic=float(np.mean(ics)), pooled_t_stat_gross=float(t), market_days=int(len(allp)),
                           hit_rate_pct=float((allp[allp != 0] > 0).mean() * 100))
        print(f"🌙 Moon Dev predict {gname}: {pred[gname]}")

    pd.to_pickle(dict(grid=grid, per_market_sel=per_market_sel, n_sweep=n_sweep, group_sel=group_sel, tf_rows=tf_rows,
                      decay=decay, decay_fit=decay_fit, fees=fees, best_oos=best_oos, pred=pred,
                      cand=cand, groups=groups), f"{CACHE}/results.pkl")
    pd.to_pickle(out, f"{CACHE}/portfolios.pkl")
    print("🌙 Moon Dev: results saved to cache/results.pkl, run make_result.py next")


if __name__ == "__main__":
    main()
