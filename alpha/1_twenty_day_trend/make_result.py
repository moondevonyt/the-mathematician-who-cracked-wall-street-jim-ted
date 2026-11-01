"""
Moon Dev: turn cache/results.pkl (from trend_test.py) into result.json in the shared plan.md format.
"""
import json, os
import numpy as np
import pandas as pd
from trend_test import (load_markets, run, sharpe, max_dd, longest_dd_days, ppy_of, cost_per_unit, SPLITS,
                        CRYPTO_COST, FUT_COST, TGT_VOL)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def clean(o):
    """Moon Dev: NaN -> None and numpy -> python so json is valid."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def trades_of(mkt, name, n=20, variant="sign_ls"):
    """Moon Dev: a trade = one run of the same side. Entry and exit at the daily close."""
    df = run(mkt, n, variant)
    c = mkt["df"]["close"]
    cpu = cost_per_unit(mkt, c)
    side = df["pos"]
    runs = (side != side.shift()).cumsum()
    out = []
    for _, g in df.groupby(runs):
        s = g["pos"].iloc[0]
        if s == 0:
            continue
        i_entry = c.index.get_loc(g.index[0]) - 1
        entry_t, exit_t = c.index[i_entry], g.index[-1]
        gross = float((1 + g["gross"]).prod() - 1)
        net = gross - float(cpu.iloc[i_entry]) - float(cpu.loc[exit_t])
        out.append(dict(market=name, side="long" if s > 0 else "short", entry_time=str(entry_t.date()),
                        entry_price=float(c.iloc[i_entry]), exit_time=str(exit_t.date()), exit_price=float(c.loc[exit_t]),
                        days=len(g), gross_bp=gross * 1e4, net_bp=net * 1e4))
    return out


def block(pn, ppy):
    return dict(sharpe=sharpe(pn, ppy), total_return_pct=float(((1 + pn).prod() - 1) * 100), max_dd_pct=float(max_dd(pn) * 100),
                vol_pct=float(pn.std() * np.sqrt(ppy) * 100), start=str(pn.index[0].date()), end=str(pn.index[-1].date()),
                win_rate_days_pct=float((pn[pn != 0] > 0).mean() * 100), longest_drawdown_days=longest_dd_days(pn))


def main():
    R = pd.read_pickle(f"{CACHE}/results.pkl")
    P = pd.read_pickle(f"{CACHE}/portfolios.pkl")
    mk = load_markets()
    G, gs = R["groups"], R["group_sel"]

    # ---------------- headline portfolio: Jim's literal rule, all markets, N = 20, net of costs
    pn = P[("all_markets", 20, "sign_ls")]
    ppy = ppy_of(pn.index)
    sp = SPLITS["all_markets"]
    full, is_b, oos_b = block(pn, ppy), block(pn[pn.index < sp], ppy), block(pn[pn.index >= sp], ppy)
    gross_all = __import__("trend_test").portfolio(mk, G["all_markets"], 20, "sign_ls", "gross")
    full["sharpe_zero_cost"] = sharpe(gross_all, ppy)

    # trades across every market in the headline portfolio
    all_tr = []
    for nm in G["all_markets"]:
        all_tr += trades_of(mk[nm], nm)
    tr = pd.DataFrame(all_tr)
    wins, losses = tr.net_bp[tr.net_bp > 0].sum(), -tr.net_bp[tr.net_bp < 0].sum()

    # ---------------- equity curve (max 500 points)
    eq = (1 + pn).cumprod()
    pick = np.unique(np.linspace(0, len(eq) - 1, 500).round().astype(int))
    eq_pts = [[str(eq.index[i].date()), float(eq.iloc[i])] for i in pick]

    # ---------------- trades list: BTC perp 20 day rule, last 300
    btc_tr = trades_of(mk["BTCUSDT-perp"], "BTCUSDT-perp")
    trades = btc_tr[-300:]

    # ---------------- price replay: BTC perp last 400 days with the 20 day signal
    d = mk["BTCUSDT-perp"]["df"].iloc[-400:]
    full_r = mk["BTCUSDT-perp"]["df"]["close"].pct_change()
    sig = np.sign(full_r.rolling(20).mean()).reindex(d.index)
    bars = [[str(t.date()), float(r.open), float(r.high), float(r.low), float(r.close), float(sig.loc[t])] for t, r in d.iterrows()]
    t0 = str(d.index[0].date())
    replay_tr = [t for t in btc_tr if t["exit_time"] >= t0]

    # ---------------- grid for the dashboard: per market + group portfolios + 4h / 1h
    grid = list(R["grid"])
    for row in R["n_sweep"]:
        grid.append(dict(params=dict(market=f"{row['group']} portfolio", group=row["group"], N=row["N"], variant=row["variant"], timeframe="1d"),
                         **{k: row[k] for k in ["is_sharpe_gross", "oos_sharpe_gross", "is_sharpe_net", "oos_sharpe_net", "sharpe_net", "vol_pct", "max_dd_pct"]}))
    grid += R["tf_rows"]

    # ---------------- buy and hold benchmark, same equal-risk weights (always long), net of costs
    import trend_test as T
    keep = T.positions
    T.positions = lambda r, n, v: pd.Series(1.0, index=r.index).where(r.notna())
    bh = {}
    for g in ["all_markets", "crypto_perps", "commodities", "fx", "index_futures"]:
        b = T.portfolio(mk, G[g], 20, "always_long", "net")
        bh[g] = dict(is_sharpe=sharpe(b[b.index < SPLITS[g]], ppy_of(b.index)), oos_sharpe=sharpe(b[b.index >= SPLITS[g]], ppy_of(b.index)))
    T.positions = keep
    print(f"🌙 Moon Dev buy and hold benchmark: {bh}")

    # ---------------- claims
    pred, fit = R["pred"], R["decay_fit"]
    pms = pd.DataFrame(R["per_market_sel"])
    trad = pms[pms.group.isin(["commodities", "fx", "index_futures"])]
    cry = pms[pms.group == "crypto_perps"]

    def gsel(g, k):
        return gs[g]["sign_ls"][k]

    claim_a = (f"Pooled t-stat of next-day P&L of the 20 day sign rule (zero cost): crypto perps {pred['crypto_perps']['pooled_t_stat_gross']:.2f}, "
               f"commodities {pred['commodities']['pooled_t_stat_gross']:.2f}, FX {pred['fx']['pooled_t_stat_gross']:.2f}, "
               f"index futures {pred['index_futures']['pooled_t_stat_gross']:.2f}. All-markets portfolio Sharpe {full['sharpe_zero_cost']:.2f} zero cost, {full['sharpe']:.2f} net.")
    claim_b = "; ".join(f"{g}: IS-picked N={gsel(g,'best_is_N')} OOS {gsel(g,'its_oos_sharpe_net'):.2f} vs N=20 OOS {gsel(g,'n20_oos_sharpe_net'):.2f}"
                        for g in ["crypto_perps", "commodities", "fx", "index_futures", "all_markets"])
    claim_c = "; ".join(f"{g}: {fit[g]['slope_per_year']:+.3f} Sharpe/yr (p={fit[g]['p_value']:.2f}, {fit[g]['first_year']}-{fit[g]['last_year']})"
                        for g in ["crypto_spot_long", "crypto_perps", "commodities", "fx", "index_futures", "stocks_SPY"])
    claim_d = (f"Trade win rate {(tr.net_bp > 0).mean()*100:.1f}% over {len(tr)} trades, day win rate {full['win_rate_days_pct']:.1f}%, "
               f"longest drawdown {full['longest_drawdown_days']} days, max DD {full['max_dd_pct']:.1f}%")

    J = json.load(open(f"{HERE}/verdicts.json"))  # Moon Dev: the verdict words are written by hand after reading the numbers
    claims = [
        dict(claim="The 20 day average move predicts the next day (trend following works)", timestamp="0:10",
             test="sign(mean of last 20 daily returns), long/short, hold 1 day; pooled t-stat per asset class + all-markets portfolio Sharpe",
             result=claim_a, verdict=J["a"]),
        dict(claim="Try 10 day, 15 day... 'see what worked best': some N beats 20 out of sample", timestamp="0:10",
             test="pick N in 2..250 on in-sample net Sharpe only, report out-of-sample", result=claim_b, verdict=J["b"]),
        dict(claim="Trend following faded 'because everyone could see that'", timestamp="0:11",
             test="calendar-year Sharpe of the 20 day rule (zero cost), straight-line fit vs year", result=claim_c, verdict=J["c"]),
        dict(claim="It works 'not beautifully': near 50% wins, long losing stretches", timestamp="0:10",
             test="all-markets 20 day portfolio, net", result=claim_d, verdict=J["d"]),
    ]

    srcs = json.load(open(f"{CACHE}/sources.json"))
    data_str = " | ".join(f"{k}: {v['path']} ({v['start']} -> {v['end']})" for k, v in srcs.items())

    result = dict(
        package="1_twenty_day_trend",
        title="Jim Simons' 20 day trend: does it work, and does it fade?",
        market=("16 Binance USDT perps + 7 Binance spot (long history) + 4 commodity futures (CL, GC, SI, NG) + 4 FX "
                "(EURUSD, GBPUSD, USDJPY, AUDUSD) + 3 index futures (ES, NQ, YM) + SPY + TSLA, daily bars"),
        data=data_str,
        headline=dict(label=J["headline_label"], value=J["headline_value"], verdict=J["headline_verdict"]),
        stats=dict(
            version="all_markets equal-risk portfolio, N=20, sign_ls (Jim's literal rule), each market vol-targeted to 10%/yr, net of costs",
            trades=int(len(tr)), win_rate=float((tr.net_bp > 0).mean() * 100), avg_R_or_bp=float(tr.net_bp.mean()),
            avg_unit="bp per trade, net, 1x notional, averaged over all markets",
            profit_factor=float(wins / losses), total_return_pct=full["total_return_pct"], max_dd_pct=full["max_dd_pct"],
            sharpe=full["sharpe"], sharpe_zero_cost=full["sharpe_zero_cost"], vol_pct=full["vol_pct"],
            longest_drawdown_days=full["longest_drawdown_days"], win_rate_days_pct=full["win_rate_days_pct"],
            in_sample=dict(period=f"{is_b['start']} -> {is_b['end']}", **is_b),
            out_of_sample=dict(period=f"{oos_b['start']} -> {oos_b['end']}", **oos_b),
            costs=(f"crypto: Binance taker 0.05% + 0.02% slippage = {CRYPTO_COST*1e4:.0f} bp per side per unit of position change (a flip = 2 sides); "
                   f"futures/FX: {FUT_COST*1e4:.0f} bp per side (Moon Dev assumption, video gives none); TSLA/SPY: 1 bp + half a cent per share per side. "
                   "Zero-cost numbers shown next to every net number. No funding charged on perps (not in the task)."),
            selection_by_group=gs,
            per_market_selection=R["per_market_sel"],
            predict_next_day=pred,
            timeframes_4h_1h=R["tf_rows"],
            after_fees=R["fees"],
            after_5_and_44=R["fees"]["after_5_and_44"],
            after_2_and_20=R["fees"]["after_2_and_20"],
            decay_fit=fit,
            honest_candidates_ranked_by_oos=[dict(oos_sharpe_net=c[0], group=c[1], N=c[2], variant=c[3]) for c in R["cand"]],
            splits=SPLITS,
            buy_and_hold_equal_risk=bh,
        ),
        equity_curve=eq_pts,
        trades=trades,
        grid=grid,
        claims=claims,
        decay=R["decay"],
        price_replay=dict(symbol="BTCUSDT-perp (Binance), daily, signal = sign of 20 day mean return (+1 long, -1 short)",
                          bars=bars, trades=replay_tr),
        takeaways=J["takeaways"],
    )
    json.dump(clean(result), open(f"{HERE}/result.json", "w"))
    print(f"🌙 Moon Dev: wrote result.json ({os.path.getsize(f'{HERE}/result.json')/1e6:.2f} MB), "
          f"{len(eq_pts)} equity points, {len(trades)} trades, {len(grid)} grid rows")


if __name__ == "__main__":
    main()
