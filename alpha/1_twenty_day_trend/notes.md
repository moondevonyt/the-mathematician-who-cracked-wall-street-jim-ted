# Package 1: Jim Simons' 20 day trend. Does it work? Does it fade?

Built by Moon Dev. Real data only.

## What Jim said
- 0:10: "predict today by the average move in the past 20 days... years ago such a system would work, not beautifully but it would work."
- 0:10: try 10 day, 15 day, "see what worked best."
- 0:11: great in the 60s, okay in the 70s, gone by the 80s, "because everyone could see that."

## What the video leaves out (so we label each guess)
The exact rule, how long to hold, how big to bet. We tested three versions:
- **sign_ls**: go long if the N day average move is up, short if down. Hold 1 day. (Our literal read.)
- **long_only**: long if up, flat if down.
- **sized**: bet size = how strong the trend is (a t-stat of the N day average, divided by 2, capped at 1x).

## Data we used
| Group | Markets | Dates |
|---|---|---|
| Crypto perps | 16 Binance USDT perps (BTC, ETH, SOL, XRP, DOGE, BNB, LINK, AVAX, CRV, FIL, ZEC, ARB, SUI, TIA, STRK, INIT), 1m bars made into 1d / 4h / 1h | 2020-01 -> 2026-10 |
| Crypto spot | BTC, ETH, BNB, XRP, LINK, DOGE, ZEC spot | 2017-08 (BTC, ETH) -> 2026-10 |
| Commodities | CL oil, GC gold, SI silver, NG gas (Yahoo daily, front month, not back-adjusted) | 2000-08 -> 2024-09 |
| FX | EURUSD, GBPUSD, USDJPY, AUDUSD (Yahoo daily) | 2000 to 2006 start -> 2024-09 |
| Index futures | ES, NQ, YM (Yahoo daily) | 2000 -> 2024-09 |
| Stocks | SPY (Yahoo daily) and TSLA (Nasdaq lit prints, 9:30 to 4) | SPY 2010 -> 2026-09, TSLA 2024-07 -> 2026-06 |

Clean up: we dropped 5 bad Yahoo FX prints (EURUSD and USDJPY on 2008-10-08 and 2008-12-08, AUDUSD 2006-12-25: one day jumps of 17% to 480% that snap back) and the day oil closed below zero (2020-04-20). Nothing else.

We do NOT have 1960s to 1980s data, so we can't see the years Jim says it was great.

In-sample (IS) and out-of-sample (OOS):
crypto IS to 2023-12, OOS 2024-01 -> 2026-10. Commodities, FX, index futures IS 2000 -> 2012, OOS 2013 -> 2024-09. SPY OOS from 2018. TSLA OOS from 2025-07. All-markets portfolio OOS from 2018.

Costs: crypto 7 bp a side (Binance taker 0.05% + 0.02% slip). Futures and FX 1 bp a side (our guess, the video gives none). TSLA and SPY 1 bp + half a cent a share. No perp funding charged.

## What we found
1. **It still works in crypto.** Jim's literal 20 day rule on the 16 perps (equal risk): Sharpe 1.01 IS, **0.72 OOS** after costs. Holding the same coins at equal risk: 1.23 IS, 0.47 OOS. So in the new data trend beat holding. The edge is "not beautiful": only about 49% of days win. The money comes from riding a few big moves.
2. **It is dead in Jim's markets.** Commodities 20 day: 0.24 IS, **-0.18 OOS**. FX: 0.20 IS, 0.08 OOS. A t-test on the next-day P&L: crypto 4.4, commodities 0.15, FX 0.56. That fits Jim: by our 2000s data it had already faded.
3. **"See what worked best" fools you.** Crypto: best IS N was 193 days (1.76 Sharpe). OOS it made 0.24. Plain 20 made 0.72. Coin by coin, the best IS N averaged 1.98 in sample and -0.09 out. Picking N overstated Sharpe by 0.4 to 1.5. In the old markets the IS winner was a slow trend (N about 100 to 190), and that did beat 20 OOS there (commodities 0.20, FX 0.46, index futures 0.67).
4. **Does it fade?** We fit a straight line to the yearly Sharpe. Crypto perps: **-0.17 Sharpe a year** (2020: 2.15, 2022: 0.03, 2026: 1.12). Spot crypto: -0.13 a year. Commodities -0.025, FX -0.03. All slope down, but none is significant (every p > 0.15). So: maybe fading, not proven.
5. **Shorter bars die on costs.** Same rule on 1h crypto bars is positive before costs (1.50 OOS at N = 20 bars) and negative at every N after costs. 4h is about break-even to 0.5.
6. **Portfolio (Jim's low volatility idea).** All markets, each set to 10% a year risk, 20 day rule: Sharpe 0.36 net (0.45 with no cost), vol 5%, max drawdown -11.8%. Its worst drawdown lasted **3,237 days** (May 2011 -> Mar 2020). Winning trades: 33%. Winning days: 51%.
7. **Jim's fees.** Best honest OOS version (all markets, N = 20, long-only, scaled to 10% vol using IS vol only), 2018 -> 2026: 11.2% a year before fees, **7.4% after 2 and 20, 3.5% after 5 and 44.** Note: holding everything at equal risk had a 0.70 OOS Sharpe, and this long-only version had 0.97, so part of it is just being long.

## Verdict
VERIFIED, with a twist. Jim was right on all three points: a 20 day trend can work, it's ugly, and it fades once everyone sees it. In commodities and FX it was already gone by 2000. In crypto it still pays (0.72 OOS Sharpe) but the yearly trend points down.

## Files
- `build_bars.py`: makes daily / 4h / 1h bars from the real 1m files into `cache/`.
- `trend_test.py`: all the tests (sweeps, IS/OOS, decay, fees). Writes `cache/results.pkl`.
- `make_result.py`: writes `result.json`. Verdict words live in `verdicts.json` (written by hand after reading the numbers).
Run: `python build_bars.py && python trend_test.py && python make_result.py` (about 5 minutes).
