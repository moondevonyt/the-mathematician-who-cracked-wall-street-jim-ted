# Moon Dev anomaly hunt (package 2)

Jim Simons, TED [~0:11]: they stayed ahead with "shorter-term approaches" and "a tremendous amount of data".
Jim [~0:13]: "any one anomaly might be a random thing. However, if you have enough data, you can tell that it's not...
an anomaly that's persisted for a sufficiently long time... But these things fade after a while."

This package does that process on Moon Dev's real data: hunt many short-term anomalies, test if each one is random,
then measure how fast each one fades.

## PART 1. The pre-declared test list (written BEFORE any result was looked at)

Units: every effect is in bp (1 bp = 0.01% price move), per trade, in the direction picked in-sample.

Each test is "signal at bar close, then the next bar's return". Direction (follow or fade) is picked from the in-sample
sign only, then frozen for out-of-sample. So each row is ONE two-sided test.

Coins in the crypto battery (11 perps with data from 2020, so they have 4 in-sample years):
BTC, ETH, BNB, XRP, DOGE, SOL, LINK, AVAX, ZEC, CRV, FIL (all USDT perps, Binance 1m klines).
Left out on purpose: ARB, SUI, TIA (start 2023, under 1 year in-sample), STRK, INIT (start 2024/2025, no in-sample).

| # | Family | Markets | Rule | Tests |
|---|---|---|---|---|
| A | Last bar return, 1m | 11 coins | sign of last 1m return -> next 1m return | 11 |
| B | Last bar return, 5m | 11 coins | same on 5m bars | 11 |
| C | Last bar return, 15m | 11 coins | same on 15m bars | 11 |
| D | Last bar return, 1h | 11 coins | same on 1h bars | 11 |
| E | Hour of day (UTC) | BTC, ETH | long the 1h bar that starts at hour h, h = 0..23 | 48 |
| F | Day of week (UTC) | BTC, ETH | long the UTC day, Mon..Sun | 14 |
| G | Session | BTC, ETH | US cash session 9:30-16:00 New York time, outside US session (16:00 -> next 9:30), weekend (Sat+Sun) | 6 |
| H | Volume spike, 5m | 11 coins | 5m volume > 4x the median of the past 288 bars (1 day): sign of that bar's return -> next 5m | 11 |
| I | Volume spike, 5m -> 1h | 11 coins | same signal -> next 12 bars (1h) | 11 |
| J | Taker buy imbalance, 5m | 11 coins | imb = 2*taker_buy_vol/vol - 1, z vs past 288 bars, abs(z) > 2: sign(z) -> next 5m | 11 |
| K | Taker buy imbalance, 1h | 11 coins | same on 1h bars, z vs past 168 bars | 11 |
| L | Cross coin lead-lag, 1m | 10 alts | sign of BTC last 1m return -> alt next 1m return | 10 |
| M | Cross coin lead-lag, 5m | 10 alts | same on 5m | 10 |
| N | Funding extreme | BTC ETH SOL XRP DOGE | rate above its past 90 readings' 90th pct or below the 10th pct: -> next 8h return | 5 |
| O | Funding change | same 5 | sign of (rate minus last rate) -> next 8h return | 5 |
| P | Liquidation spike -> next 5m | BTC ETH SOL XRP | liq_usd above past 2016 bars' (1 week) 99th pct: sign of liq_imbalance -> next 5m | 4 |
| Q | Liquidation spike -> next 1h | same | same -> next 12 bars | 4 |
| R | TSLA last bar return | TSLA | 1m, 5m, 15m (inside the regular session only) | 3 |
| S | TSLA half hour of day | TSLA | long each 30 min block, 9:30..15:30 (13 blocks) | 13 |
| T | TSLA day of week | TSLA | open -> close, Mon..Fri | 5 |
| U | TSLA open drift | TSLA | sign of 9:30-10:00 return -> 10:00-16:00 return | 1 |
| V | TSLA close drift | TSLA | sign of 9:30-15:30 return -> 15:30-16:00 return | 1 |
| W | TSLA overnight vs intraday | TSLA | long overnight (close -> open); long intraday (open -> close); sign of overnight -> intraday | 3 |
| X | TSLA block prints | TSLA | print direction (sign of price vs last lit) -> lit price 5 min later. Off-exchange (TRF) >= 10k shares, TRF >= 50k shares, lit >= 10k shares | 3 |
| Y | TSLA trade count spike | TSLA | 1m trade count > 4x past 390 bar median: sign of bar return -> next 5 min | 1 |

Total: 224 tests. (First draft said 234. That was an adding mistake: the rows above sum to 224. No test was added or dropped.)

## Pre-declared rules for "is it random?"

- In-sample (IS) and out-of-sample (OOS): crypto IS 2020-01 -> 2023-12, OOS 2024-01 -> 2026-10.
  Funding (data starts 2024-01), liquidations (data 2024-06 -> 2025-12) and TSLA: first 60% of days IS, last 40% OOS.
- t-stat: on DAILY P&L (sum of that day's trades, in bp), so trades inside one day do not count as free extra proof.
- Filters, in order: naive |t| >= 2; Bonferroni p < 0.05 / 224; Benjamini-Hochberg FDR q = 0.05;
  deflated Sharpe probability > 0.95 (Bailey and Lopez de Prado, N = 224 trials);
  circular shift permutation test (300 random shifts of the signal vs the returns, p <= 0.01).
- "Held OOS" = same sign as IS and OOS t >= 2.
- "Persisted" (Jim's rule as a number): split IS into 4 equal time chunks (crypto: about one per year).
  Strict = t >= 2 in the IS direction in all 4 chunks. Loose = same sign in all 4 chunks.
- Costs per round trip: crypto taker 2 x (5 bp fee + 2 bp slippage) = 14 bp; crypto maker 2 x 2 bp = 4 bp;
  TSLA 2 x (1 bp + half a cent). Every signal is charged as its own round trip (in, then out one bar later).
- Fade: per survivor, effect in bp per calendar quarter, fit effect = a * exp(-k * years), half-life = ln(2) / k.

## PART 2. What we found (written AFTER the run)

How to rebuild: `python hunt.py` (about 6 min, makes `cache/`, 1.3 GB, gitignored), then `python real_spreads.py`, then
`python report.py` (writes `result.json`). Python: /opt/anaconda3/bin/python.

### The funnel (224 tests)
| Filter | Passed |
|---|---|
| Looked real in-sample, abs(t) >= 2 | 82 |
| Bonferroni (p < 0.05 / 224) | 38 |
| Benjamini-Hochberg FDR 5% | 57 |
| FDR + deflated Sharpe > 0.95 | 31 |
| FDR + deflated Sharpe + shuffle test p <= 0.01 ("not random") | 31 |
| Not random AND held out-of-sample (OOS t >= 2, same sign) | 13 |
| ...and still positive after taker costs (14 bp round trip) | 0 |
| ...and still positive after maker costs (4 bp round trip) | 2 |

### Jim's claims, in numbers
1. "With enough data you can tell it's not random." Yes. In-sample t of 10 or more held OOS 6 of 9 times (67%).
   t of 2 to 3 held only 9 of 35 (26%). The 3 randomness checks raised the hit rate from 31% to 42%.
2. "Persisted for a sufficiently long time." This is the best filter we found. t >= 2 in EVERY one of 4 in-sample
   chunks: 9 of 16 held OOS (56%). Pooled only: 16 of 66 (24%). Just "same sign in every chunk" did not help (29% vs 35%).
3. "These things fade." Yes. The typical FDR passer kept only 12% of its in-sample bp out-of-sample.
   Survivors' median half-life: 1.6 years (fit on quarterly bp). 3 of 13 survivors showed no fade.
   Did the strongest fade fastest? Not clearly. Spearman(IS t, fade speed) = +0.19, p 0.17. Weak, not proven.
4. "Shorter-term approaches." The edge is real but tiny. Every survivor is a 1m or 5m bar effect.
   None pays a taker. See the table below.
5. "Spreads are narrower, more liquidity." Yes on our data. Median coin spread proxy (Abdi-Ranaldo, from 1m high/low/close):
   3.79 bp in 2020, 0.47 bp in 2025. Median trades per minute: 72 -> 540. Real BTC perp quoted spread (bookTicker):
   0.035 bp (Aug 2023) -> 0.015 bp (Mar 2024). The proxy is noisy on BTC (volatility leaks into it).

### Top 3 survivors (OOS = 2024-01 -> 2026-10, all bp per trade)
| Anomaly | OOS gross | After taker (14 bp) | After maker (4 bp) | OOS t | One tick |
|---|---|---|---|---|---|
| CRV 1m: fade the last bar | +6.83 | -7.17 | +2.83 | 42.0 | 23.8 bp |
| CRV 5m: fade a taker-imbalance spike | +1.06 | -12.94 | -2.94 | 4.0 | 23.8 bp |
| CRV 5m: fade the last bar | +4.50 | -9.50 | +0.50 | 27.5 | 23.8 bp |

Why these are not money: CRV trades near $0.42, and its price step (tick) is $0.001. That is 24 bp. A 6.8 bp "edge" is a
quarter of one tick. It is bid-ask bounce: the last trade hit the bid or the ask, the next one hits the other side.
You cannot take that as a taker. The edge GREW out-of-sample only because CRV's price fell, so each tick got bigger in bp.
The maker number assumes every maker order fills at the close price. In real life the fills you get are the bad ones.

The other 10 survivors are "BTC moved last minute, the alt follows next minute" on 8 alts (plus FIL and ZEC 5m bounce).
Real, steady, and 0.05 to 0.35 bp per trade. A taker round trip is 14 bp. That is 40x to 270x too small.

### What failed
Funding extremes (5 coins), liquidation spikes (4 coins), hour of day, day of week, US session, weekends, and all 30
TSLA tests (open/close drift, overnight vs intraday, block prints, half-hour blocks) failed the randomness checks.
SOL and XRP liquidation spikes passed FDR in-sample (+4 to +19 bp per trade), but failed the shuffle test and did not
reach OOS t >= 2. They are the closest thing to a tradeable lead here and need more data (only 18 months exist).

### Notes and limits
- Equity curve in result.json: best survivor (CRV 1m fade), growth of 1.0 taking ONE average trade per day, net of
  taker costs: -87%. Gross and maker curves are in `equity_curve_gross` and `equity_curve_maker`.
- Funding (2024-01 -> 2026-08) and liquidations (2024-06 -> 2025-12) start after the crypto in-sample period ends, so they
  use their own 60/40 time split. TSLA bars are built from lit_rth_hist (tsla_1m_flow only covers 3 months).
- Deflated Sharpe uses N = 224 trials and a null Sharpe spread of 1/sqrt(days) per test, so fast and slow tests compare fairly.
- The video leaves out: which shorter-term approaches, what "persisted long enough" means, and how fast anomalies fade.
  Every number here is Moon Dev's own test, not Jim's. Jim gives no Medallion return numbers in this talk.
