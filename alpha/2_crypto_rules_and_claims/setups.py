"""Moon Dev: the 4 setups from the video as signal arrays (long space, shorts are the mirror).
Every number not said in the video is a parameter that gets swept."""
import itertools
import numpy as np
import pandas as pd
import core

EXITS = {"R1.5": (0, 1.5), "R2": (0, 2.0), "R3": (0, 3.0), "trail": (1, 0.0)}   # video leaves this out (besides the 200 SMA)
ACCEL_ATR = 3.0   # "accelerated / extended" = close > 3 ATR beyond the 20 SMA (video leaves the number out, ~97th pct)


def zone_bars(a, smin):
    """Moon Dev: pullback trigger bar. 20 SMA sloping up (> smin ATR/bar) and under price, 9 EMA above the 20,
    bar dips into the 9/20 entry zone (low <= 9 EMA, not more than 0.25 ATR under the 20)."""
    trend = (a["slope"] > smin) & (a["ema9"] > a["sma20"]) & (a["c"] > a["sma20"])
    return trend & (a["l"] <= a["ema9"]) & (a["l"] >= a["sma20"] - core.ZONE_TOL_ATR * a["atr"])


def leg(a):
    z0 = zone_bars(a, 0.0)
    return core.leg_features(a["c"], a["l"], a["sma20"], a["ema9"], a["atr"], np.nan_to_num(a["slope"]),
                             np.nan_to_num(a["ext"]), z0, ACCEL_ATR)


def pullback_signals(a, smin, align, skip_first, htf_ok, pb_idx):
    sig = zone_bars(a, smin)
    if align:
        sig &= htf_ok
    if skip_first:
        sig &= ~(pb_idx == 1)      # Moon Dev: skip the FIRST pullback after the trend accelerated (0:14)
    return sig, a["h"], a["l"]     # buy stop over the trigger bar high, stop at its low


def base_signals(a, nbars, width, align, htf_ok):
    """Moon Dev: base = last N bars fit in a box <= width ATR, sitting on / above the 20 SMA, 20 SMA still rising.
    Buy stop over the box high, stop under the box low. (Box size and length: video leaves this out.)"""
    hh = pd.Series(a["h"]).rolling(nbars).max().values
    ll = pd.Series(a["l"]).rolling(nbars).min().values
    sig = ((hh - ll) <= width * a["atr"]) & (a["slope"] > 0) & (a["c"] > a["sma20"]) & \
          (ll >= a["sma20"] - core.ZONE_TOL_ATR * a["atr"]) & (a["ema9"] > a["sma20"])
    if align:
        sig &= htf_ok
    return sig, hh, ll


def swing_signals(a, trend, maxbars):
    """Moon Dev: 0:30 idea. Uptrend retraces under the 20 SMA, then a CLOSE back over the 20 = enter next open,
    stop under the lowest low of the retracement. trend: 'sma200' = close over the 200, 'slope20' = 20 SMA was
    rising when the retrace began. maxbars = longest retrace that still counts (video leaves this out)."""
    c, s20, l = a["c"], a["sma20"], a["l"]
    below = c <= s20
    n = len(c)
    grp = np.cumsum(~below)                                   # same id for a run of below bars + the bar before
    run = pd.Series(below.astype(int)).groupby(grp).cumsum().values
    prev_run = np.r_[0, run[:-1]]
    reclaim = (~below) & (prev_run >= 1) & (prev_run <= maxbars)
    low_run = pd.Series(np.where(below, l, np.inf)).groupby(grp).cummin().values
    stop = np.minimum(np.r_[np.inf, low_run[:-1]], l)        # lowest low of the retrace incl. the reclaim bar
    start = np.arange(n) - prev_run - 1                       # last bar before the retrace began
    start = np.clip(start, 0, n - 1)
    if trend == "sma200":
        tr = c > a["sma200"]
    else:
        tr = a["slope"][start] > 0
    sig = reclaim & tr & np.isfinite(stop)
    return sig, np.full(n, np.nan), stop


def grid_specs():
    """Moon Dev: every combo we sweep. One dict per config."""
    out = []
    for tf, smin, block, ex, align, skip in itertools.product(["5m", "15m", "1h", "4h"], [0.03, 0.08, 0.15], [0.0, 1.5],
                                                              EXITS, [0, 1], [0, 1]):
        out.append(dict(setup="pullback", tf=tf, slope_min=smin, block200_R=block, exit=ex, htf_align=align, skip_first=skip))
    for tf, nb, w, ex, align in itertools.product(["5m", "15m", "1h", "4h"], [6, 12, 24], [1.5, 3.0], EXITS, [0, 1]):
        out.append(dict(setup="base", tf=tf, base_bars=nb, base_width_atr=w, block200_R=1.5, exit=ex, htf_align=align))
    for tf, trend, mb, ex in itertools.product(["1h", "4h", "1d"], ["sma200", "slope20"], [3, 10, 30], EXITS):
        out.append(dict(setup="swing_reclaim20", tf=tf, trend=trend, max_retrace_bars=mb, block200_R=0.0, exit=ex))
    for smin, block, ex in itertools.product([0.03, 0.08, 0.15], [0.0, 1.5], EXITS):
        out.append(dict(setup="pullback_us_session", tf="1m/2m/5m", slope_min=smin, block200_R=block, exit=ex, htf_align=1))
    return out


def key(p):
    return tuple(sorted(p.items()))
