"""
Moon Dev's Edge Finder dashboard builder (Jim Simons TED video).
Reads the 3 real result.json files in ../alpha, trims them, and bakes them into index.html.
Every number on the dashboard comes from these files. Nothing is made up here.
Run: python3 build_dashboard.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALPHA = HERE.parent / "alpha"
P1 = json.load(open(ALPHA / "1_twenty_day_trend" / "result.json"))
P2 = json.load(open(ALPHA / "2_anomaly_hunt" / "result.json"))
P3 = json.load(open(ALPHA / "3_simons_machine" / "result.json"))
BOT = (ALPHA / "3_simons_machine" / "bot_simons_machine.py").read_text()
print("🌙 Moon Dev: loaded 3 result.json files")


def r(v, d=4):
    return round(v, d) if isinstance(v, float) else v


# ======================= package 1: the 20 day trend =======================
s1 = P1["stats"]
GROUPS = ["crypto_perps", "crypto_spot_long", "commodities", "fx", "index_futures", "stocks_SPY", "stocks_TSLA",
          "all_markets"]
# portfolio grid per group and variant: N -> [is_net, oos_net, is_gross, oos_gross]
pgrid = {}
for x in P1["grid"]:
    p = x["params"]
    if p["timeframe"] != "1d" or not p["market"].endswith("portfolio"):
        continue
    pgrid.setdefault(p["group"], {}).setdefault(p["variant"], {})[p["N"]] = [
        r(x["is_sharpe_net"], 3), r(x["oos_sharpe_net"], 3), r(x["is_sharpe_gross"], 3), r(x["oos_sharpe_gross"], 3)]
assert set(pgrid) == set(GROUPS), "Moon Dev: missing a group portfolio in package 1 grid"

# every single market at N=20, sign_ls, 1d
mkt20 = {}
for x in P1["grid"]:
    p = x["params"]
    if p["timeframe"] == "1d" and p["variant"] == "sign_ls" and p["N"] == 20 and not p["market"].endswith("portfolio"):
        mkt20[p["market"]] = {"group": p["group"], "is": r(x["is_sharpe_net"], 3), "oos": r(x["oos_sharpe_net"], 3)}
for m in s1["per_market_selection"]:
    mkt20[m["market"]].update({"best_N": m["best_is_N"], "best_is": r(m["best_is_sharpe"], 3),
                               "best_oos": r(m["its_oos_sharpe"], 3), "bh": r(m["buyhold_sharpe"], 3)})

decay1 = {}
for x in P1["decay"]:
    if x["level"] == "group":
        decay1.setdefault(x["group"], []).append([x["period"], r(x["metric"], 3)])

# same 20 days of lookback on faster bars (crypto perps portfolio, sign_ls)
tf = [{"tf": "1d", "N": 20, **{k: r(v, 3) for k, v in zip(["is_net", "oos_net", "is_gross", "oos_gross"],
                                                              pgrid["crypto_perps"]["sign_ls"][20])}}]
for x in s1["timeframes_4h_1h"]:
    if x["N_in_days"] == 20.0:
        tf.append({"tf": x["params"]["timeframe"], "N": x["params"]["N"], "is_net": r(x["is_sharpe_net"], 3),
                   "oos_net": r(x["oos_sharpe_net"], 3), "is_gross": r(x["is_sharpe_gross"], 3),
                   "oos_gross": r(x["oos_sharpe_gross"], 3)})
assert len(tf) == 3, "Moon Dev: expected 1d, 4h and 1h rows at 20 days"

rp1 = P1["price_replay"]
p1 = {
    "headline": P1["headline"],
    "pgrid": pgrid,
    "sel": {g: {v: {k: r(val, 3) if isinstance(val, float) else val for k, val in s1["selection_by_group"][g][v].items()}
                for v in ["sign_ls", "long_only", "sized"]} for g in GROUPS},
    "splits": s1["splits"],
    "bh": s1["buy_and_hold_equal_risk"],
    "decay_fit": {g: s1["decay_fit"][g] for g in GROUPS if g in s1["decay_fit"]},
    "decay": decay1,
    "pred": s1["predict_next_day"],
    "mkt20": mkt20,
    "tf": tf,
    "fees": s1["after_fees"],
    "port": {k: s1[k] for k in ["trades", "win_rate", "sharpe", "sharpe_zero_cost", "vol_pct", "max_dd_pct",
                                "longest_drawdown_days", "win_rate_days_pct", "in_sample", "out_of_sample", "costs"]},
    "eq": [[t, r(v, 4)] for t, v in P1["equity_curve"]],
    "replay": {"symbol": rp1["symbol"], "bars": rp1["bars"], "trades": rp1["trades"]},
    "trades": P1["trades"],
    "claims": P1["claims"],
}

# ======================= package 2: the anomaly hunt =======================
s2 = P2["stats"]
pts2 = [[x["is_t"], x["oos_t"], 2 if x["survived"] else (1 if x["not_random"] else 0), x["params"]["id"],
         x["params"]["scheme"], x["direction"]] for x in P2["grid"]]
surv = [{"id": x["params"]["id"], "scheme": x["params"]["scheme"], "dir": x["direction"], "is_t": x["is_t"],
         "oos_t": x["oos_t"], "is_bp": x["is_bp"], "oos_bp": x["oos_bp"], "taker": x["oos_net_taker_bp"],
         "maker": x["oos_net_maker_bp"], "hl": x["half_life_years"], "tick": x["tick_bp_oos"],
         "per_day": x["trades_per_day_oos"]} for x in P2["grid"] if x["survived"]]
surv.sort(key=lambda z: -z["oos_bp"])
assert len(surv) == s2["filters"]["held_oos_t>=2 (survived)"], "Moon Dev: survivor count mismatch"
rp2 = P2["price_replay"]
p2 = {
    "headline": P2["headline"],
    "filters": s2["filters"],
    "hit": s2["hit_rates"],
    "hit_t": s2["hit_rate_by_is_t"],
    "persist": s2["persistence_rule"],
    "half_life": s2["median_half_life_years_survivors"],
    "no_fade": s2["survivors_with_no_fade"],
    "kept": s2["median_edge_kept_oos_fdr_set"],
    "liq": s2["liquidity_by_year_median_11_coins"],
    "quoted": s2["real_quoted_spreads"],
    "pts": pts2,
    "surv": surv,
    "decay": P2["decay"],
    "oos": s2["out_of_sample"],
    "costs": s2["costs"],
    "replay": {"symbol": rp2["symbol"], "bars": rp2["bars"], "trades": rp2["trades"][-120:]},
    "claims": P2["claims"],
}

# ======================= package 3: the Simons machine =======================
s3 = P3["stats"]
g3 = [x for x in P3["grid"] if "holdout_sharpe" in x]


def g3row(model, target, features="all", horizon="4h", retrain="monthly", sm=None, band=None):
    out = {}
    for x in g3:
        p = x["params"]
        if (p["model"], p["target"], p["features"], p["horizon"], p["retrain"]) == (model, target, features, horizon,
                                                                                   retrain):
            if sm is not None and (p["smooth_halflife_bars"], p["band"]) != (sm, band):
                continue
            out[p["cost_bp_per_side"]] = [x["pre_holdout_sharpe"], x["holdout_sharpe"]]
    assert len(out) == 8, f"Moon Dev: grid rows missing for {model} {target} {features}"
    return out


# the model race: the exact configs in package 3 claim (a), plus the model we actually picked before the holdout
race = [
    {"key": "trend20", "name": "Jim's 20 day trend (1 line)", "sub": "long top third of 20 day movers, short bottom third",
     "c": g3row("trend20", "next bar")},
    {"key": "blend_all", "name": "Blend, all 25 features", "sub": "not picked: lower Sharpe before the holdout",
     "c": g3row("blend", "next 24h")},
    {"key": "blend_pv", "name": "THE MACHINE we picked", "sub": "blend of 19 price + volume features, picked before the holdout",
     "c": g3row("blend", "next 24h", "price+volume")},
    {"key": "ridge", "name": "Ridge regression", "sub": "all features", "c": g3row("ridge", "next 24h")},
    {"key": "gbm", "name": "LightGBM", "sub": "200 trees, all features", "c": g3row("gbm", "next 24h")},
    {"key": "best1", "name": "Best single feature", "sub": "strongest past link", "c": g3row("best1", "next 24h")},
]
assert race[2]["c"][7][1] == s3["out_of_sample"]["taker_7bp"]["sharpe"], "Moon Dev: picked model != headline"
COSTS = sorted(race[0]["c"].keys())

datatypes = [{"name": n, "c": g3row("blend", "next 24h", f)} for n, f in
             [("price only", "price"), ("+ volume", "price+volume"), ("+ taker flow", "price+volume+flow"),
              ("+ funding (all)", "all")]]

# retrain test only lives in the claim text: "monthly: 1.443 / 1.248; quarterly: ...; once: ..."
import re
retrain = [{"name": m[0], "zero": float(m[1]), "taker": float(m[2])}
           for m in re.findall(r"(monthly|quarterly|once): ([\d.-]+) / ([\d.-]+)", P3["claims"][2]["result"])]
assert len(retrain) == 3, "Moon Dev: could not read the retrain test"

# LightGBM trading every 4h bar with NO turnover control (raw speed vs cost)
gbm_raw = {x["params"]["cost_bp_per_side"]: x["holdout_sharpe_no_turnover_control"] for x in g3
           if (x["params"]["model"], x["params"]["target"], x["params"]["features"], x["params"]["horizon"],
               x["params"]["retrain"]) == ("gbm", "next bar", "all", "4h", "monthly")}
assert len(gbm_raw) == 8, "Moon Dev: gbm raw rows missing"

rp3 = P3["price_replay"]
p3 = {
    "headline": P3["headline"],
    "is": s3["in_sample"], "oos": s3["out_of_sample"], "full": s3["full_oos_2022_2026_taker"],
    "portfolio": s3["portfolio"], "capacity": s3["capacity"],
    "fees20": s3["after_2_and_20"], "fees44": s3["after_5_and_44"],
    "costs_txt": s3["costs"],
    "race": race, "costs": COSTS, "datatypes": datatypes, "retrain": retrain, "gbm_raw": gbm_raw,
    "eq": [[t[:19], r(v, 4)] for t, v in P3["equity_curve"]],
    "decay": P3["decay"],
    "fi": P3["feature_importance"],
    "liq": P3["liquidation_addon"],
    "replay": {"symbol": rp3["symbol"], "bars": [[b[0][:19]] + b[1:] for b in rp3["bars"]],
               "trades": [dict(t, time=t["time"][:19]) for t in rp3["trades"]]},
    "trades": [dict(t, time=t["time"][:19]) for t in P3["trades"]],
    "claims": P3["claims"],
    "leaves_out": P3["video_leaves_out"],
}

DATA = {"p1": p1, "p2": p2, "p3": p3, "bot": BOT}
js = json.dumps(DATA, separators=(",", ":"), ensure_ascii=False)
tpl = (HERE / "template.html").read_text()
out = tpl.replace("/*__DATA__*/null", js)
(HERE / "index.html").write_text(out)
print(f"🌙 Moon Dev: wrote index.html ({len(out) / 1e6:.2f} MB, data {len(js) / 1e6:.2f} MB)")
