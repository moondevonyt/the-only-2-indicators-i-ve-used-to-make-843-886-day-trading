# Package 1: Emmanuel's exact rules on TSLA (Moon Dev)

**Verdict: NOT PROVEN.** His exact rules make a little money on TSLA after costs, but not enough to tell apart from luck.
Random entries on the same days, same side, held the same time, do about as well.

## Data
- Real Databento Nasdaq Basic trades, `/Volumes/Moon 26/tick_data/databento/xnas_basic_trades/TSLA/`, Nasdaq lit prints only (publisher 81).
- 2024-07-01 to 2026-09-23. 557 days. 217,186 one-minute bars, 09:30 to 16:00 New York time.
- Skipped Databento's 3 bad days (2024-07-25, 2025-12-15, 2026-01-12). The day after each gets no gap bias.
- Bars: 1m, 2m, 5m, 15m, 1h, daily. 9 EMA, 20 SMA, 200 SMA on each.
- The daily 200 SMA needs 200 days, so it only exists from **2025-04-21**. Before that there is no daily 200 target. We did not fake it.
- In-sample (IS) 2024-07 to 2025-12. Out-of-sample (OOS) 2026-01 to 2026-09. All params picked on IS only.
- Costs: 1 bp per side + half a cent slippage per side. Size: 1x the account per trade, no leverage (the video gives no size).

## The exact system we coded
Gap of 1%+ sets the side (gap up = long only). Time of day sets the charts: 9:30-10 uses 1m + 2m, 10-12 uses 2m + 5m, 12-16 uses 5m + 15m.
Long when: 20 SMA rising (not flat, not too steep), price over it, 9 EMA over the 20, slow chart agrees, the bar dips into the 9 EMA and closes over the 20,
the bar is a doji or a bottoming tail, the 20 held 2+ earlier touches, it is not the first pullback after a big stretch, and no 200 SMA sits within 0.3% overhead.
Buy one cent over the bar high (next bar only). Stop one cent under its low. Target the 1h or daily 200 SMA if one is ahead. Raise the stop under each new pivot.
Flat at 16:00. Shorts are the mirror.

## Numbers
| Version | IS trades | IS net | OOS trades | OOS net | OOS PF |
|---|---|---|---|---|---|
| Exact rules | 88 | +2.5% (0 cost +4.6%) | 36 | +5.3% (0 cost +6.2%) | 2.0 |
| Every literal rule stacked (adds "20 on fast chart = 9 EMA on slow chart") | 2 | +0.7% | 2 | -0.6% | 0 |
| No MA filters (just touch of 9 EMA in gap direction) | 1,694 | -0.6% (0 cost +47%) | 644 | -10.7% | 0.90 |
| Best of 16,524 combos on IS (base breakout, 6 bars, 3 ATR wide, 1R target) | 481 | +129% | 152 | +11.5% | 1.26 |
| Best IS pullback, both sides every day (no gap bias) | 352 | +17.6% | 173 | +2.4% | 1.10 |

- Exact rules t-stat: 0.51 IS, 1.27 OOS. Not significant. Win rate 32% IS, 33% OOS. Max drawdown -3.2% IS.
- Random control (same day, side, hold, 1,000 runs): random **beat** the exact rules in 84% of runs IS, and lost to them in 87% of runs OOS (only 36 trades). Mixed, so not proven.
- The best IS combo fell from t-stat 3.4 to 1.1 OOS, and random entries on its days beat it in 99% of OOS runs. Its IS money came from the gap side, not the MAs.
- Grid: 7,077 of 16,524 combos made money IS, 6,774 OOS. About a coin flip.

## The claims, tested on TSLA
| Claim | Time | Result | Verdict |
|---|---|---|---|
| Flat 20 SMA = chop, do not trade | 0:10 | flat -0.03R vs sloped -0.08R. Steepest (> 0.25 ATR/bar) was best, +0.08R | Not supported |
| More touches = more reliable | 0:11 | 0 touches -0.07R, 1: -0.04R, 2: -0.00R, 3+: -0.05R | Not supported |
| Far from the 20 SMA = exit, not entry | 0:13 | Next 30 min in trend direction: in zone -0.3 bp, 1.5-2.5 ATR +3.7 bp, over 3.5 ATR +4.8 bp. Stretch kept going | Not supported (opposite) |
| Skip first pullback after a stretch | 0:14 | first one -0.105R (n=482) vs others -0.042R (n=1,148) | Weak, about 1 standard error |
| Gap up = long, gap down = short | 0:22 | Close follows a 1-2% gap 59% of days, 2-3% gap 64%, tiny gaps 46%. Trades with gap +0.19R vs against +0.05R | Yes in 2024-25, faded in 2026 |
| 200 SMA is support / resistance and the target | 0:15 | 1h 200 rejects 42% vs a fake level 1% away 39%. Daily 61% vs 59%. 200 target hit 3.8% of the time | Not supported |

Gap side alone (hold 10:00 to 16:00 on 1-3% gap days): +65 bp a day IS, +6 bp a day OOS. That was the real driver in 2024-25, and it faded.

## What worked
- The MA filters beat the raw pullback entry. Raw entries lose money after costs. Filtered entries do not.
- Following the gap side helped in 2024-25.

## What did not
- The MAs did not beat random entries taken on the same days and side.
- Flat-slope, touch count, extension, and 200 SMA claims all failed on TSLA.
- Stacking every literal rule leaves 4 trades in 2.2 years.
- Costs eat most of the raw edge: 1m and 2m stops are small, so 3 bp round trip is a big slice of 1R.

## What the video leaves out (all swept, none invented)
- **Gap size:** swept 0 (any), 1%, 2%, 3%. Base case 1%.
- **"Flat" 20 SMA:** slope in ATRs per bar (20 SMA change over 5 bars / 5 / ATR14). Swept 0, 0.05, 0.1. Base case 0.05.
- **"45 degrees" / too steep:** swept no cap, 0.25, 0.4 ATR per bar. Base case 0.25. (Median TSLA slope is about 0.1.)
- **Extension / acceleration:** close more than X ATR from the 20 SMA. Swept off, 2.5, 4. Base case 2.5.
- **Touch count and what a touch is:** touch = low within 0.1 ATR of the 20 and close holds it. Count resets when the slope turns. Swept 0, 1, 2. Base case 2 (he says "first two times bounced, the next you can trust").
- **Doji / bottoming tail:** doji = body 25% of range or less. Tail = lower wick half the range or more. Swept vs any bar.
- **Timeframe alignment:** swept none, slow chart trend agrees, and trend + touch of slow 9 EMA.
- **200 SMA "right above":** swept off and within 0.3%.
- **Exit with no 200 near:** swept 1R, 2R, 3R, trail under 3-bar pivots. Base case trail (what he did on LASE).
- **Base:** swept 6, 10, 15 bars; 2, 3, 4 ATR wide; 20 SMA inside the range.
- **Position size, risk per trade, adds:** not given. We use 1x notional, one position, no adds.
- **Order life:** the buy stop lives for one bar of the trigger chart.
- If stop and target hit in the same minute, we count the stop.

## Files
- `build_bars.py`: raw trades to `bars_1m.parquet` (5 MB, kept).
- `backtest.py`: bars, indicators, signals, numba fill sim, 16,524-combo grid (`grid_rows.pkl`, gitignored).
- `report.py`: IS pick, controls, claims. `finalize.py`: writes `result.json`.
- Run: `python build_bars.py && python backtest.py && python report.py && python finalize.py` (anaconda python, needs numba + pyarrow).
