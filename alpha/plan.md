# Moon Dev Edge Finder plan: "The mathematician who cracked Wall Street | Jim Simons" (TED)

Three test packages, run in parallel by three Opus agents. Real data only. If data is missing, fail loudly.

## What Jim actually says (time stamps from transcript.md)
- Renaissance: big returns "with surprisingly low volatility and risk compared with other hedge funds." [~0:08]
- Method: "You design an algorithm, you test it out on a computer, does it work, doesn't it work." [~0:09]
- The 20 day trend: "commodities or currencies had a tendency to trend... in periods. If you decided, I'm going to predict
  today by the average move in the past 20 days... years ago such a system would work, not beautifully but it would work."
  Test 10 day, 15 day, "try all those things and see what worked best." [0:10]
- Decay: "trend following would have been great in the 60s, sort of okay in the 70s. By the 80s it wasn't... because
  everyone could see that." [0:10-0:11]
- How they stayed ahead: "other approaches and shorter-term approaches", plus "a tremendous amount of data" and very
  smart people. [0:11]
- Machine learning: "You look at a lot of data and you try to simulate different predictive schemes until you get better
  and better at it." [~0:12]
- Data: "everything is grist for the mill... weather, annual reports, monthly, quarterly reports, the historic data itself,
  volumes, you name it." Terabytes a day. [~0:13]
- Anomalies: "any one anomaly might be a random thing. However, if you have enough data, you can tell that it's not...
  an anomaly that's persisted for a sufficiently long time, so the probability of it being random is not high. But these
  things fade after a while. Anomalies can get washed out. So you have to keep on top of the business." [~0:13]
- Fees: 2 and 20 is normal; Renaissance charged "5 and 44". Capacity: they bought out investors "because there's a
  capacity to the fund." [~0:15]
- Science in markets "reduced volatility, increased liquidity, spreads are narrower." [~0:16]

## What Jim LEAVES OUT (never invent; sweep and label "video leaves this out")
The exact trend rule (sign of the 20 day mean? position size by it?), the holding period, which "shorter-term
approaches", which models, what "persisted long enough" means as a number, how fast anomalies fade, position sizing,
leverage. He gives NO Medallion return numbers in this video, so never quote any as if he did.

## Data we do NOT have (say so, do not fake)
Weather, annual / quarterly reports, Fed interest rate history, 1960s-80s commodities and currencies.

## Shared output format (every package)
Folder `alpha/<n>_<name>/` with the code, `notes.md` (plain 5th grade words), and `result.json`:
```
{
  "package": "1_twenty_day_trend", "title": "...", "market": "...", "data": "exact paths + date range",
  "headline": {"label": "...", "value": "...", "verdict": "VERIFIED | BUSTED | BETTER VERSION FOUND | NOT PROVEN"},
  "stats": {"trades": 0, "win_rate": 0, "avg_R_or_bp": 0, "profit_factor": 0, "total_return_pct": 0, "max_dd_pct": 0,
            "sharpe": 0, "vol_pct": 0, "in_sample": {...}, "out_of_sample": {...}, "costs": "what was charged",
            "after_5_and_44": {...}},
  "equity_curve": [[iso_time, equity], ...],          // max 500 points
  "trades": [ ... max 300 real trades ... ],
  "grid": [{"params": {...}, "...": 0}],              // for dashboard sliders, include "market" as a param
  "claims": [{"claim": "...", "timestamp": "0:10", "test": "...", "result": "number", "verdict": "..."}],
  "decay": [{"period": "2021", "metric": 0}],         // how the edge changes over time (Jim's "anomalies fade")
  "price_replay": {"symbol": "...", "bars": [[t,o,h,l,c,signal], ...], "trades": [...]},  // max 400 bars
  "takeaways": ["...", "..."]
}
```
Units: say "bp" (price move, 1 bp = 0.01%) or "R" (multiple of risk to the stop) explicitly in every label.

## Package 1: Jim's 20 day trend, and does it fade? (alpha/1_twenty_day_trend/)
## Package 2: Shorter-term anomaly hunt with Jim's "is it random?" test (alpha/2_anomaly_hunt/)
## Package 3: Simons-style machine: many weak signals, walk-forward, low volatility (alpha/3_simons_machine/)
