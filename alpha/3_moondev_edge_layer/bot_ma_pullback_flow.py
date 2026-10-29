"""
🌙 Moon Dev's MA pullback + order flow bot (DRY RUN ONLY, no real orders).

His rules (Emmanuel Malyarovich, "The ONLY 2 Indicators", transcript.md):
  - 20 SMA must slope (0:10). Long: 20 SMA rising and under price, 9 EMA above it (0:11).
  - Pullback into the 9 EMA / 20 SMA zone (0:13). Skip if the 200 SMA sits within 1R overhead (0:16).
  - Buy a break of the trigger bar high, stop under its low (0:24). Shorts are the mirror.
Moon Dev's add-on (picked on 2020 -> 2024-06 data only):
  - trigger bar volume >= 0.82x its last 20 bars
  - trigger bar taker imbalance on our side >= -0.094 (buyers not getting run over on a long)

⚠️ Backtest verdict: BUSTED. OOS 2024-07 -> 2026-10 this still lost about -0.49R a trade on BTC 15m
after costs. This file is a clean example of how to wire flow into a bot, not a money printer.
"""
import sys
import numpy as np
import pandas as pd

KL = "/Volumes/Moon 26/tick_data/binance/futures_klines_1m"
SYMBOL = "BTCUSDT"
TF = "15min"
SLOPE_MIN = 0.3        # 20 SMA move over 5 bars, in ATRs (video gives no number: Moon Dev sweep)
TARGET_R = 2.0         # video gives no R target: Moon Dev sweep
MAX_HOLD = 48          # bars
MIN_RVOL = 0.8166      # picked on in-sample only
MIN_IMB = -0.0945      # picked on in-sample only
DRY_RUN_DAYS = 60


def load_bars(symbol):
    """Moon Dev: real Binance perp 1m klines -> 15m bars with taker flow."""
    import glob
    f = glob.glob(f"{KL}/{symbol}/{symbol}_1m_*.parquet")
    assert f, f"Moon Dev: no 1m klines for {symbol} in {KL}"
    d = pd.read_parquet(f[0], columns=["open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"])
    b = d.set_index("open_time").resample(TF).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                                    "volume": "sum", "taker_buy_volume": "sum"}).dropna()
    c = b.close
    b["ema9"] = c.ewm(span=9, adjust=False).mean()
    b["sma20"] = c.rolling(20).mean()
    b["sma200"] = c.rolling(200).mean()
    pc = c.shift(1)
    tr = np.maximum(b.high - b.low, np.maximum((b.high - pc).abs(), (b.low - pc).abs()))
    b["atr"] = tr.rolling(14).mean()
    b["slope"] = (b.sma20 - b.sma20.shift(5)) / b.atr
    b["imb"] = (2 * b.taker_buy_volume - b.volume) / b.volume        # Moon Dev taker imbalance, -1..+1
    b["rvol"] = b.volume / b.volume.shift(1).rolling(20).mean()
    return b


def check_signal(bar):
    """Moon Dev: returns +1 long setup, -1 short setup, 0 nothing. bar = the just-closed trigger bar."""
    if bar.slope >= SLOPE_MIN and bar.ema9 > bar.sma20 and bar.low <= bar.ema9 and bar.close > bar.sma20:
        side = 1
    elif bar.slope <= -SLOPE_MIN and bar.ema9 < bar.sma20 and bar.high >= bar.ema9 and bar.close < bar.sma20:
        side = -1
    else:
        return 0
    if bar.rvol < MIN_RVOL or bar.imb * side < MIN_IMB:      # Moon Dev flow filter
        return 0
    return side


def place_order(side, entry, stop, target):
    """🌙 Moon Dev: THIS is where the exchange call would go (stop-entry + bracket).
    Example: exchange.create_order(SYMBOL, 'STOP_MARKET', 'buy' if side == 1 else 'sell', size, stopPrice=entry)
    Dry run: we only print."""
    print(f"   🌙 Moon Dev DRY RUN order: {'BUY' if side == 1 else 'SELL'} stop @ {entry:.2f}, "
          f"stop loss {stop:.2f}, target {target:.2f}")


def run(days=DRY_RUN_DAYS):
    b = load_bars(SYMBOL)
    start = b.index[-1] - pd.Timedelta(days=days)
    rows = b[b.index >= start - pd.Timedelta(days=5)]
    print(f"🌙 Moon Dev bot dry run on {SYMBOL} {TF}, {rows.index[0]} -> {rows.index[-1]}")
    pos, results = None, []
    for k in range(1, len(rows)):
        bar, prev = rows.iloc[k], rows.iloc[k - 1]
        if rows.index[k] < start:
            continue
        if pos is not None:                       # manage the open trade on this bar
            d = pos["side"]
            hit_stop = bar.low <= pos["stop"] if d == 1 else bar.high >= pos["stop"]
            hit_tgt = bar.high >= pos["target"] if d == 1 else bar.low <= pos["target"]
            pos["bars"] += 1
            if hit_stop or hit_tgt or pos["bars"] >= MAX_HOLD:
                px = pos["stop"] if hit_stop else pos["target"] if hit_tgt else bar.close
                r = (px - pos["entry"]) * d / pos["risk"] - 0.0014 * pos["entry"] / pos["risk"]
                results.append(r)
                print(f"   🌙 Moon Dev exit {rows.index[k]} @ {px:.2f}  R after costs = {r:+.2f}")
                pos = None
            continue
        side = check_signal(prev)                 # trigger bar = the bar that just closed
        if side == 0:
            continue
        trig_hit = bar.high > prev.high if side == 1 else bar.low < prev.low
        if not trig_hit:
            continue
        entry = max(bar.open, prev.high) if side == 1 else min(bar.open, prev.low)
        stop = prev.low if side == 1 else prev.high
        risk = (entry - stop) * side
        gap200 = (prev.sma200 - entry) * side
        if risk <= 0 or (0 < gap200 < risk):      # 0:16 the 200 SMA right overhead = skip
            continue
        target = entry + side * TARGET_R * risk
        print(f"🌙 Moon Dev signal {rows.index[k]} {'LONG' if side == 1 else 'SHORT'} imb {prev.imb:+.2f} rvol {prev.rvol:.2f}")
        place_order(side, entry, stop, target)
        pos = {"side": side, "entry": entry, "stop": stop, "target": target, "risk": risk, "bars": 0}
        if (bar.low <= stop) if side == 1 else (bar.high >= stop):    # stopped on the entry bar
            r = -1 - 0.0014 * entry / risk
            results.append(r)
            print(f"   🌙 Moon Dev stopped on entry bar, R = {r:+.2f}")
            pos = None
    r = np.array(results)
    if len(r):
        print(f"🌙 Moon Dev dry run done: {len(r)} trades, win rate {(r > 0).mean() * 100:.0f}%, "
              f"avg R {r.mean():+.2f}, total R {r.sum():+.1f}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else DRY_RUN_DAYS)
