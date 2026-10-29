# Moon Dev Edge Finder plan: "The ONLY 2 Indicators" (Emmanuel Malyarovich)

Three test packages, run in parallel by three Opus agents. Real data only. If data is missing, fail loudly.

## The exact rules from the video (time stamps from transcript.md)
- Indicators: 9 EMA (short), 20 SMA (medium), 200 SMA (long). Regular candles. Volume ok. [0:03-0:07]
- Trade WITH the trend "95% of the time". Only two setups: (1) pullback / retracement into the MAs,
  (2) base breakout (long) or base breakdown (short) after a consolidation. Same rules mirrored for shorts. [0:08-0:10]
- Rule 1: never trade when the 20 SMA is flat / sideways / wavy. [0:10]
- Long: 20 SMA rising and UNDER price. 9 EMA above the 20 SMA. "Entry zone" = between the 9 EMA and the 20 SMA. [0:11-0:13]
- More prior touches of the 20 SMA that held = more reliable. [0:11]
- Slope: "ideally 45 degrees". Steeper = overextended, flatter = no momentum. [0:12]
- Extension = distance from price to the 20 SMA. Far = exit zone, not entry zone. [0:13]
- After the trend accelerates (extended + steep slope), SKIP the first retracement / first breakout. [0:14]
- 200 SMA: flatter is better. Acts as support / resistance. Use as Target 1. Skip a long if the 200 is right above entry.
  Most useful on 1h, daily, weekly. [0:15-0:17]
- Timeframes by time of day (US Eastern): 9:30-10:00 use 1m + 2m; 10:00-12:00 use 2m + 5m; 12:00-16:00 use 5m + 15m.
  Look for alignment across the two (example: pullback to the 20 SMA on 2m = the 9 EMA on 5m). [0:20]
- Bias comes from the daily chart: overnight gap (catalyst) + daily breakout. Gap up = look long, gap down = look short. [0:22, 0:31]
- Execution (LASE trade): entry above the high of a doji / bottoming-tail bar at the MA, stop below that bar's low.
  Add on later pullbacks, raise stop under each new pivot. Exited early; the real top was the daily 200 SMA. [0:24-0:27]
- Daily swing idea: uptrend retraces, enter after price RECLAIMS the 20 SMA, stop below the retracement low. [0:30]
- "Works on small caps, mid caps, large caps, gold, silver, crypto." [0:30]

## What the video LEAVES OUT (never invent these; sweep them and label them as sweeps)
Slope number for "flat" and "45 degrees", extension threshold, how many touches, gap size, what a "base" is
(bars / range width), profit target besides the 200 SMA, exit when no 200 SMA is near, position size, risk per trade.

## Shared output format (every package)
Folder `alpha/<n>_<name>/` with the code, `notes.md` (plain words, what it proved, what the video left out),
and `result.json`:
```
{
  "package": "1_tsla_exact_rules", "title": "...", "market": "...", "data": "exact paths + date range",
  "headline": {"label": "...", "value": "...", "verdict": "VERIFIED | BUSTED | BETTER VERSION FOUND | NOT PROVEN"},
  "stats": {"trades": 0, "win_rate": 0, "avg_R": 0, "profit_factor": 0, "total_return_pct": 0, "max_dd_pct": 0,
            "sharpe": 0, "in_sample": {...}, "out_of_sample": {...}, "costs": "what was charged"},
  "equity_curve": [[iso_time, equity], ...],          // max 500 points
  "trades": [{"t_in","t_out","side","entry","exit","R","pnl_pct","setup","symbol"}],   // max 300, real trades
  "grid": [{"params": {...}, "trades": 0, "win_rate": 0, "avg_R": 0, "profit_factor": 0, "return_pct": 0}],
  "claims": [{"claim": "...", "timestamp": "0:10", "test": "...", "result": "number", "verdict": "..."}],
  "price_replay": {"symbol": "...", "bars": [[t,o,h,l,c,ema9,sma20,sma200], ...], "trades": [...]},  // one real day/window, max 400 bars
  "takeaways": ["...", "..."]
}
```

## Package 1: TSLA, his exact day-trading rules (stocks, his home market)
Data: Databento TSLA trades 2024-07 -> 2026-09 on Moon 26. Build 1m/2m/5m/15m/1h/daily bars (RTH).

## Package 2: Crypto, "it works on crypto" plus every individual claim measured
Data: Binance 1m klines, 16 perps (2020 -> 2026-10) and spot, on Moon 26.

## Package 3: Moon Dev's edge layer, data others don't have
His setups + order flow (taker buy volume, TSLA tick flow / block prints), liquidations, funding. Find the better version
and write a small example bot.
