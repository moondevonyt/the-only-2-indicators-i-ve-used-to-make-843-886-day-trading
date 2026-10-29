# Package 3: Moon Dev's edge layer

**Verdict: BUSTED.** Moon Dev's data makes his MA pullback lose less. It does not make it win.

## The question
Emmanuel says the MAs are "only one part" of his plan (0:01, 0:36). The rest is price action he never puts a number on.
So we asked: if we add data most traders don't have (taker flow, liquidations, funding, TSLA block prints), does his signal turn into a real edge?

## What we built
- His pullback signal, exactly as he says it: 20 SMA sloping, 9 EMA on the trend side, a pullback into the 9/20 zone,
  buy the break of the trigger bar, stop at the other end of that bar, skip if the 200 SMA is within 1R overhead (0:10 to 0:24).
- Crypto: 16 Binance perps, 5m and 15m, 2020 to 2026-10. TSLA: 2m and 5m, regular hours, 2024-07 to 2026-09.
- Each trade gets scored at the close of the trigger bar (before we enter, so no peeking).
- Filters were picked on in-sample only (crypto 2020 to 2024-06). Then judged on 2024-07 to 2026-10.
- Costs: crypto 0.07% a side (0.05% fee + 0.02% slip). TSLA 1 bp a side + half a cent a share. Zero cost shown too.

## Results, out of sample, avg R a trade after costs

| Market | MA only | MA + best 2 data filters | MA + his bottoming tail | tail + 1 data filter |
|---|---|---|---|---|
| Crypto 5m (16 perps) | -0.77 | -0.49 | -0.63 | -0.37 |
| Crypto 15m (16 perps) | -0.42 | **-0.27** | -0.29 | -0.20 |
| TSLA 2m | -0.17 | -0.13 | -0.02 | -0.02 |
| TSLA 5m | -0.16 | -0.08 | -0.11 | -0.16 |

Before any cost, crypto MA only was about -0.05R a trade. So there is no edge to start with. Costs then make it much worse.

## Best data filter
**Taker buy imbalance on the trigger bar** (`imb_trig`), plus trigger bar volume vs its last 20 bars (`rvol`).
Plain words: on a long, skip the trigger bar if sellers clearly beat buyers on it, or if it was a quiet bar.
It got picked on in-sample for crypto 5m, crypto 15m and TSLA 5m, and it helped out of sample every time.
Q1 to Q5 of `imb_trig` went from about -0.12R to 0.00R (zero cost) on crypto 5m, in both periods.

## The catch: most of that is just the candle
Taker flow and candle shape move together (corr about 0.36). When we split by where the trigger bar closed in its range:
- Trigger bar closes in the bottom third of its range (no bottoming tail): **-0.38R** a trade before costs (crypto 5m).
- Closes in the top third (bottoming tail): **+0.03R**.
- Inside each candle group, flow adds only about 0.01 to 0.04R.

That candle rule is **his own words** (enter above a doji / bottoming tail bar, 0:24 to 0:25). It is the strongest single filter we found.
It still can't pay 0.14% round trip on a stop that is 0.29% wide (5m crypto median).

## Liquidations and funding (short samples, say so)
- Liquidations: `liq_clean_{BTC,ETH,SOL,XRP}_5m.csv`, 2024-06 to 2025-12, 18 months. First 60% in-sample, last 40% out.
  "Longs flushing into a long pullback = fuel" helped a bit on crypto 15m out of sample (-0.54R to -0.39R). Still losing. NOT PROVEN.
- We did NOT use `liq_data_*_ohlcv.csv`. data_sources.md bans them: their prices are liquidation prints, not market prices.
- Funding (2024-01 to 2026-08, 60/40 split): the in-sample pick made out of sample worse. BUSTED.
- TSLA block prints (>= 1k and >= 5k shares, direction vs last lit price): noisy, no clean pattern. 2m picked it as a second filter, OOS still lost.

## What the video leaves out (we swept these, did not invent them)
- Slope number for "flat" and "45 degrees": swept 0.1 / 0.3 / 0.6 ATR of 20 SMA move over 5 bars. Steeper was not better.
- Profit target other than the 200 SMA: swept 1R, 1.5R, 2R, 3R, and "200 SMA if 1R+ away". Base = 2R.
- Max hold: 48 bars (crypto), end of day (TSLA). Position size: equity curve uses 1% risk a trade.
- "Right above" for the 200 SMA: within 1R.

## What we did not test (timebox)
- Raw tick trades (BTC/ETH/SOL/XRP/DOGE): the kline taker volume is built from the same trades, so we used that.
- ZEC OKX liquidations (30 days) and HYPE liquidations (no HYPE price data on Moon 26): too short or no prices.
- His multi-timeframe line-up (2m 20 SMA = 5m 9 EMA) and the gap-day bias. Package 1 owns that for TSLA.

## Files
- `common.py`: MAs + the trade engine (numba).
- `build_crypto.py`, `build_tsla.py`: bars, signals, features. Cache goes to `cache/` (gitignored, 650 MB).
- `analyze.py`: buckets, IS filter pick, OOS check. `make_result.py`: writes `result.json`.
- `bot_ma_pullback_flow.py`: dry-run bot, BTC 15m, his rules + the flow filter. Last 60 days: 121 trades, -0.55R a trade. Do not run it live.

Run order: `python build_crypto.py && python build_tsla.py && python make_result.py`.
