"""
Moon Dev edge layer: shared pieces.
His MA pullback signal (9 EMA / 20 SMA / 200 SMA) + a simple bar-by-bar trade sim.
Rules from the video (transcript.md):
  0:10  never trade a flat 20 SMA
  0:11  long = 20 SMA rising and under price, 9 EMA above the 20
  0:13  entry zone = between the 9 EMA and the 20 SMA
  0:16  skip the long if the 200 SMA is right above entry (resistance)
  0:24  entry above the trigger (doji / bottoming tail) bar high, stop under its low
What the video leaves out (these are SWEEPS, labeled as such): the slope number for "flat",
the profit target besides the 200, the max hold, how close "right above" is.
Built by Moon Dev.
"""
import numpy as np
import pandas as pd
from numba import njit

SLOPE_LOOKBACK = 5   # bars used to measure the 20 SMA slope (video gives no number, Moon Dev sweep)


def add_mas(df):
    """Moon Dev: 9 EMA, 20 SMA, 200 SMA, ATR14 on close. Regular candles (0:05)."""
    c = df["close"]
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["sma20"] = c.rolling(20).mean()
    df["sma200"] = c.rolling(200).mean()
    pc = c.shift(1)
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()))
    df["atr"] = tr.rolling(14).mean()
    df["slope"] = (df["sma20"] - df["sma20"].shift(SLOPE_LOOKBACK)) / df["atr"]
    return df


@njit(cache=True)
def find_and_trade(o, h, l, c, ema9, sma20, sma200, slope, day_id,
                   smin, smax, tgt_r, use200tgt, skip200, max_hold, flat_eod):
    """Moon Dev pullback engine. One position at a time per series.
    Returns arrays: trigger idx, entry idx, exit idx, dir, entry px, exit px, stop, target, exit code."""
    n = len(c)
    out = np.zeros((n // 4 + 10, 9))
    k = 0
    i = 201
    while i < n - 2:
        d = 0
        if (slope[i] >= smin and slope[i] <= smax and ema9[i] > sma20[i]
                and l[i] <= ema9[i] and c[i] > sma20[i]):
            d = 1
        elif (slope[i] <= -smin and slope[i] >= -smax and ema9[i] < sma20[i]
                and h[i] >= ema9[i] and c[i] < sma20[i]):
            d = -1
        if d == 0:
            i += 1
            continue
        j = i + 1
        if flat_eod and day_id[j] != day_id[i]:   # stocks: no entry on a new day
            i += 1
            continue
        # entry: break of trigger bar high (long) / low (short) on the next bar only
        if d == 1:
            if h[j] <= h[i]:
                i += 1
                continue
            entry = max(o[j], h[i])
            stop = l[i]
        else:
            if l[j] >= l[i]:
                i += 1
                continue
            entry = min(o[j], l[i])
            stop = h[i]
        risk = (entry - stop) * d
        if risk <= 0:
            i += 1
            continue
        # 0:16 skip if the 200 SMA sits right above (long) / below (short) entry, inside 1R
        gap200 = (sma200[i] - entry) * d
        if skip200 and gap200 > 0 and gap200 < risk:
            i += 1
            continue
        target = entry + d * tgt_r * risk
        if use200tgt and gap200 >= risk:
            target = sma200[i]          # 0:16 the 200 SMA as Target 1
        # walk forward from the entry bar
        code = 0
        exit_px = c[j]
        m = j
        while True:
            if d == 1:
                if l[m] <= stop:
                    exit_px = min(o[m], stop) if m > j else stop
                    code = 1
                    break
                if h[m] >= target:
                    exit_px = max(o[m], target) if m > j else target
                    code = 2
                    break
            else:
                if h[m] >= stop:
                    exit_px = max(o[m], stop) if m > j else stop
                    code = 1
                    break
                if l[m] <= target:
                    exit_px = min(o[m], target) if m > j else target
                    code = 2
                    break
            if m - j + 1 >= max_hold or m + 1 >= n:
                exit_px = c[m]
                code = 3
                break
            if flat_eod and day_id[m + 1] != day_id[m]:
                exit_px = c[m]
                code = 4
                break
            m += 1
        out[k, 0] = i
        out[k, 1] = j
        out[k, 2] = m
        out[k, 3] = d
        out[k, 4] = entry
        out[k, 5] = exit_px
        out[k, 6] = stop
        out[k, 7] = target
        out[k, 8] = code
        k += 1
        if k >= out.shape[0]:
            break
        i = m + 1
    return out[:k]


def run_signals(df, smin=0.3, smax=99.0, tgt_r=2.0, use200tgt=False, skip200=True,
                max_hold=48, flat_eod=False):
    """Moon Dev: run the engine on a bar frame (needs add_mas first). Returns a trades frame."""
    day = df["day_id"].values if "day_id" in df else np.zeros(len(df), dtype=np.int64)
    a = find_and_trade(df.open.values, df.high.values, df.low.values, df.close.values,
                       df.ema9.values, df.sma20.values, np.nan_to_num(df.sma200.values, nan=-1e18),
                       np.nan_to_num(df.slope.values), day.astype(np.int64),
                       smin, smax, tgt_r, use200tgt, skip200, max_hold, flat_eod)
    t = pd.DataFrame(a, columns=["i_trig", "i_in", "i_out", "dir", "entry", "exit", "stop", "target", "code"])
    for col in ["i_trig", "i_in", "i_out", "dir", "code"]:
        t[col] = t[col].astype(np.int64)
    t["t_in"] = df.index.values[t.i_in.values]
    t["t_out"] = df.index.values[t.i_out.values]
    t["risk_pct"] = (t.entry - t.stop).abs() / t.entry
    t["gross_pct"] = (t.exit - t.entry) / t.entry * t.dir
    return t


def apply_costs(t, side_pct, per_share=0.0):
    """Moon Dev costs: side_pct per side (fee + slip, as a fraction), plus a per-share cost (stocks)."""
    cost = 2 * side_pct + (per_share / t.entry + per_share / t.exit if per_share else 0.0)
    t["net_pct"] = t.gross_pct - cost
    t["R"] = t.net_pct / t.risk_pct
    t["R0"] = t.gross_pct / t.risk_pct
    return t


def stats(r, pct=None):
    """Moon Dev scorecard on an R series (and the matching pct series for returns)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n == 0:
        return {"trades": 0, "win_rate": 0, "avg_R": 0, "profit_factor": 0}
    win = r[r > 0].sum()
    loss = -r[r < 0].sum()
    s = {"trades": int(n), "win_rate": round(float((r > 0).mean() * 100), 1),
         "avg_R": round(float(r.mean()), 3),
         "profit_factor": round(float(win / loss), 2) if loss > 0 else 99.0}
    if pct is not None:
        s["return_pct"] = round(float(np.sum(pct) * 100), 1)
    return s
