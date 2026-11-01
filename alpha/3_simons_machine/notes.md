# Package 3: Moon Dev's Simons machine

Jim (~0:12): "what we did was machine learning. You look at a lot of data and you try to simulate different
predictive schemes until you get better and better at it." We built a small, honest version on Moon Dev's data.

## Verdict: NOT PROVEN

The machine makes money after costs, with low risk. But Jim's own one-line 20 day trend beat it on the final test.

| Final holdout, 2025-07-01 -> 2026-10-24 (never used to pick anything) | Sharpe | Return per year | Max drawdown |
|---|---|---|---|
| Best machine (blend of 19 weak features, 4h bars), taker 7 bp/side | 0.84 | 9.0% | -18.8% |
| Same, maker 2 bp/side | 0.97 | 10.5% | -18.2% |
| Same, zero cost | 1.03 | 11.0% | -17.9% |
| Jim's 20 day trend alone (cross-section, smoothed), taker | 1.39 | | |
| LightGBM, all features, taker | -0.62 | | |

Full walk-forward test 2022-01 -> 2026-10, best machine, taker: Sharpe 1.36, 14.7% a year, vol 10.8% a year,
max drawdown -18.8%. Note: 2022-01 -> 2025-06 is out of sample for the models, but we DID use it to pick the
smoothing, the band and the best model. Only the holdout is fully clean.

## The data (real only)
- Binance USDT perp 1 minute bars, 16 coins (BTC ETH BNB XRP LINK ZEC DOGE CRV SOL AVAX FIL ARB SUI TIA STRK INIT),
  2020-01 -> 2026-10-24, from `/Volumes/Moon 26/tick_data/binance/futures_klines_1m/`. Made into 1h and 4h bars.
  Each coin joins when it has 20 days of history. We trade only when 6+ coins are live.
- Funding: `/Volumes/Moon 26/tick_data/binance/funding/`, only BTC ETH SOL XRP DOGE, only 2024-01 -> 2026-08.
- Liquidations: only `liq_clean_{BTC,ETH,SOL,XRP}_5m.csv` (data_sources.md bans the `liq_data_*_ohlcv` files).
  2024-06 -> 2025-12. Add-on test only.
- We do NOT have weather, annual reports, or old commodity data. Not faked.
- Skipped: the TSLA version (time box).

## Features (all known at the bar close, no peeking)
Every bar is stamped at its close. Each feature is turned into a rank across coins at that same moment, so no
scaling uses future data. 19 to 25 features:
- Price: return over 1 bar, 12h, 1, 3, 7, 10, 15 and 20 days (20 days = Jim's trend), 1 and 7 day volatility,
  last bar high-low range, 1 and 7 day return after removing BTC's move (30 day beta), distance from the 20 day
  high, 20 day return divided by 20 day vol.
- Volume: dollar volume z-score (vs past 7 days), 24h dollar volume z-score (vs past 20 days), trade count z-score,
  average trade size z-score.
- Taker flow: taker buy share of volume (as -1 to +1) over 1 bar, 1 day, 7 days, and last bar minus 7 day.
- Funding: last funding rate, 3 day average (5 coins only, missing = neutral).
- Context, LightGBM only (same for every coin): hour of day, BTC's last bar and 1 day move, market vol, spread of moves.

Target: the next bar's return minus the average of all coins (market neutral), divided by that bar's spread.
We also tested a slower target: the next 24 hours (still trading every 4h bar).

## Models (simplest first)
1. `trend20`: Jim's 20 day trend, no fitting.
2. `best1`: the single feature with the strongest past link.
3. `blend`: every feature, equal weight, sign set from the past.
4. `ridge`: ridge regression.
5. `gbm`: small LightGBM (200 trees, 15 leaves). Settings fixed up front, never tuned.

Walk-forward: train on all past data, retrain every month, predict the next month only, 1 day gap so no target
leaks. First test month is 2022-01. Also: retrain every quarter, and train ONCE on 2020-2022 and never again.

## Portfolio
Each bar: long the top third of coins, short the bottom third, same dollars each side. Sized to 10% vol a year
from the past 30 days of pnl (max 4x). Turnover control: smooth the score (EMA, 0 to 48 bars) and skip small
trades (band). Costs on every dollar traded: taker 5 bp fee + 2 bp slippage = 7 bp per side. Maker 2 bp per side
(assumes every limit fills, too kind). Zero cost too. Funding payments are NOT charged (no data for 11 coins).
Main horizon is 4h: at 1h, costs ate every model except the 20 day trend.

## The 5 claims
| Claim | Result | Verdict |
|---|---|---|
| (a) Many schemes beat one rule (~0:12) | Holdout Sharpe after taker: 20 day trend 1.39, blend 1.36, ridge 0.23, LightGBM -0.70, best single feature -1.06 (all 4h, 24h target except trend) | NOT PROVEN |
| (b) More data types help (~0:13) | Blend at zero cost, holdout: price only 0.53, + volume 1.03, + taker flow 1.60, + funding 1.54. Before the holdout: 1.56, 1.75, 1.50, 1.48 | VERIFIED (mixed) |
| (c) Keep retraining (~0:13) | 2023 -> 2026, Sharpe zero cost / taker: monthly 1.44 / 1.25, quarterly 1.49 / 1.30, train once 1.47 / 1.35 | BUSTED |
| (d) Low volatility (~0:08) | vol 10.8% a year, max drawdown -18.8%, Sharpe 1.36 over 2022-2026 | VERIFIED |
| (e) Capacity is limited (~0:15) | edge half gone near $10M, gone near $38M (holdout version: gone near $14M) | VERIFIED |
| Add-on: liquidations help | 4 coins, 2025: IC 0.017 -> 0.025, but Sharpe 0.21 -> 0.03 | NOT PROVEN |

## What carried weight (LightGBM gain, all features)
- Price 58%, context 18%, taker flow 15%, volume 8%, funding 0.6%.
- Top coin features: 7 day volatility (calm coins beat wild ones, OOS IC -0.041 before holdout, -0.044 holdout),
  7 day taker buy share (buyers in control keep winning, IC +0.033 / +0.030), distance from the 20 day high
  (near the high keeps winning, IC +0.036 / +0.047). Jim's 20 day return: 5% of gain.
- IC = how well a feature's rank lines up with the next return rank (-1 to +1). 0.03 is small but real across 16 coins.

## The big lesson: speed vs cost
LightGBM found real fast signal: trading every 4h bar with no smoothing, its holdout Sharpe was 1.50 at zero cost
(2.13 on 1h bars). It trades about 2.7x its capital a day, so taker costs made it -4.35. Smoothing cut the costs but
also killed the signal. Jim's fast edge needs Jim's costs. That raw book breaks even near 1.8 bp per side (ours is 7).

## Decay (Sharpe after taker, best model, per year)
2022: 1.77, 2023: 0.37, 2024: 2.93, 2025: 2.98, 2026 (to Oct): -1.92. Train-once version: 2023 0.84, 2024 2.47,
2025 2.72, 2026 -1.05. 2026 is losing for both, so the edge is fading right now.

## Capacity
Square-root impact: cost per trade = 1.0 x daily vol x sqrt(trade $ / 30 day avg daily $ volume). The 1.0 is a
rule of thumb, NOT from the video. At $1M the median trade is 0.11% of a 4h bar's volume and impact costs 2.4% a
year. At $10M: 1.1% of the bar, 7.5% a year. At $100M: 11%, 24% a year.

## Jim's fees (best model, 2022-01 -> 2026-10, 10% vol)
- Fund before fees: 15.1% a year.
- After 2 and 20: investors get 9.7% a year.
- After 5 and 44 (Renaissance): investors get 3.6% a year. On the holdout alone: -5.9% total.

## What the video leaves out
Which models, holding period, position sizing, leverage, how fast anomalies fade, what "persisted long enough"
means as a number. No Medallion return numbers are given in this talk. Top third / bottom third, 10% vol, the
cost numbers, the smoothing grid and the impact constant are our choices, not Jim's.

## Files
- `prep.py`: 1m -> 1h bars, funding, clean liqs into `cache/` (gitignored, about 150 MB).
- `machine.py`: features, models, walk-forward, portfolio, stats.
- `run_models.py A|B|C|D`: runs all configs, caches scores.
- `report.py`: picks settings before the holdout, writes `result.json` and `model/meta.json`.
- `bot_simons_machine.py`: dry run. Pulls Binance public 1h bars, builds the features, prints the long and short
  book. The order call goes in `place_order()`.
