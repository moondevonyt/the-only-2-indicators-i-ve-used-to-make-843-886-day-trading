# Package 2: does it work on crypto? (Moon Dev)

He says at 0:30: "it works with gold, silver, crypto, doesn't matter what you trade."
We tested that on 16 Binance perps, 2020 to 2026-10, real 1m klines only.

## Verdict: BUSTED

- We picked the best settings on old data (2020 to 2024-06), then ran them on new data (2024-07 to 2026-10).
- Every setup fell apart on the new data, after costs (avg R per trade, in-sample -> out-of-sample):
  - pullback into 9/20 zone (4h): +0.39R -> +0.01R
  - base breakout (4h): +0.58R -> -0.05R
  - 0:30 daily swing, close reclaims the 20 SMA (1d): +1.10R -> -0.13R
  - his day-trade version (US session, 1m/2m/5m by time of day): -0.66R -> -0.58R
- Random entries with the SAME exits did as well or better on the new data (swing: random +0.18R vs his -0.13R).
- Only 4% of the 384 pullback configs made money on the new data. Base: 11%. Swing: 12%. US session: 0%.
- Spot (BTC, ETH, SOL) gives the same sign as perps, so this is not a perp quirk.

Why the old data looked great: 2020-21 was a huge crypto bull run. Trailing stops rode it. Random entries made money then too.

## The claims, as numbers (1h, all 16 coins, longs plus the short mirror)

| Claim | Time | Number | Verdict |
|---|---|---|---|
| Flat 20 SMA = chop, don't trade it | 0:10 | flat-slope entries +0.07R vs sloped ~0.00R; trend strength does not rise with slope | BUSTED |
| "45 degree" slope is best | 0:12 | middle slope band 0.00R, very steep +0.09R | BUSTED |
| More touches of the 20 = more reliable | 0:11 | 0/1/2/3+ touches: +0.02 / +0.01 / +0.04 / +0.01R, win rate 34% flat | BUSTED |
| Far from the 20 = exit zone | 0:13 | most stretched 10% of bars went UP 0.20% more over 12 bars (1h). 15m: slight dip | BUSTED |
| Skip the first pullback after a big run | 0:14 | first +0.04R vs later +0.03R | BUSTED |
| 200 SMA is support / resistance | 0:15 | 1h: rejects 50.3% vs 51.6% for fake lines 3-6% away. 1d: 47.4% vs 48.5% | BUSTED |
| 200 SMA is a magnet / Target 1 | 0:16 | reaches the 200 first 49.3% of the time, a coin flip (fake lines 49.4%) | BUSTED |
| 9/20 "entry zone" beats other entries | 0:13 | zone +0.01R, shallow +0.01R, deep +0.02R | MIXED |

Raw tables (every bucket, 15m check, counts) are in `result.json` under `claim_tables` and `ma200_events`.

## Small things that helped (not enough)
- Higher time frames lose less. Zero-cost avg R: 5m -0.02, 15m -0.01, 1h +0.03, 4h +0.08.
- Higher time frame alignment and the trailing stop helped a little on average.
- The 200 SMA "skip if in the way" rule did nothing (avg R the same on or off).

## Crypto has no 9:30 open
His time-of-day plan is built for US stocks. We tested two ways:
1. 24/7, no clock, each time frame with its higher time frame (5m+15m, 15m+1h, 1h+4h, 4h+1d).
2. US session only (9:30-16:00 New York time, DST aware, so 13:30 or 14:30 UTC start), his mapping:
   9:30-10 = 1m with 2m, 10-12 = 2m with 5m, 12-16 = 5m with 15m. Flat by 16:00.
   Before costs it made +0.06R. After costs -0.58R. Stops under a 1m bar are tiny, so fees are a big part of each R.

## What the video leaves out (all swept, labeled in `grid` and `left_out_by_video`)
- Slope number for "flat" and "45 degrees". We use 20 SMA change per bar, in ATRs, over 5 bars. Swept 0.03 / 0.08 / 0.15.
- How close the 200 SMA must be to skip a trade. Swept off / 1.5R.
- Profit target when no 200 SMA is in the way. Swept 1.5R / 2R / 3R / trail under pivots.
- What "accelerated" means. We use close more than 3 ATR past the 20 SMA.
- What a "base" is. Swept 6 / 12 / 24 bars, box 1.5 / 3 ATR wide.
- 0:30 swing: how long a pullback can last (3 / 10 / 30 bars) and what "uptrend" means (over the 200, or 20 SMA rising).
- How far under the 20 a bar may dip and still count (0.25 ATR), max hold time, position size (1% risk per trade).
- Gap / catalyst bias. Crypto has no overnight gap, so we left this out. Said here so nobody thinks we tested it.

## How we tested (so you can trust it)
- Entry over the trigger bar high (low for shorts), stop at the other end, just like his LASE trade.
- Shorts are the exact mirror: we flip the price and run the same long code.
- Higher time frame data only used once that bar has closed. Pivots only used once confirmed. No peeking.
- Inside one bar we guess the path (the nearer high or low came first). Same rule for his entries and the random ones.
- Costs: 0.05% Binance taker + 0.02% slippage, each side (0.14% round trip). Zero-cost numbers shown too.
- 672 configs x 16 coins = 10,752 backtests. Params picked on in-sample only.

## Files
- `prep.py` builds 5m/15m/1h/4h/1d bars from the raw 1m files into `cache/` (gitignored, 459 MB).
- `core.py` indicators + numba trade loop. `setups.py` the 4 setups. `engine.py` runs one config on one coin, plus random baseline.
- `sweep.py` all configs on all coins. `claims.py` every claim. `report.py` IS pick, OOS test, baselines, spot check, `result.json`.
- Run order: `python prep.py && python sweep.py && python claims.py && python report.py` (about 5 minutes on 8 cores).
