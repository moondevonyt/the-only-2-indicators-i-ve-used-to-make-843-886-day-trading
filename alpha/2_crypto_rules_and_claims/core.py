"""Moon Dev: core engine for the "ONLY 2 Indicators" crypto test.
9 EMA, 20 SMA, 200 SMA. Signals are vectorized, the trade loop is numba.
Shorts are the exact mirror: we flip the prices (x -> -x, high <-> low) and run the SAME long code."""
import os
import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
SYMS = ["ARB", "AVAX", "BNB", "BTC", "CRV", "DOGE", "ETH", "FIL", "INIT", "LINK", "SOL", "STRK", "SUI", "TIA", "XRP", "ZEC"]
OOS_START = pd.Timestamp("2024-07-01", tz="UTC")        # Moon Dev: IS = 2020 -> 2024-06, OOS = 2024-07 -> 2026-10
COST_SIDE = 0.0005 + 0.0002                              # Moon Dev: Binance taker 0.05% + 0.02% slippage, per side
TF_MIN = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
HTF = {"1m": "2m", "2m": "5m", "5m": "15m", "15m": "1h", "1h": "4h", "4h": "1d"}

# Moon Dev: numbers the video never gives. Fixed here, labeled "video leaves this out" everywhere.
SLOPE_LOOKBACK = 5        # bars used to measure the 20 SMA slope
ZONE_TOL_ATR = 0.25       # how far under the 20 SMA a pullback bar may poke and still count (he says touches are not perfect)
TOUCH_TOL_ATR = 0.10      # a "touch" of the 20 SMA = low within 0.1 ATR of it
MAX_HOLD = {"1m": 120, "2m": 120, "5m": 96, "15m": 96, "1h": 120, "4h": 60, "1d": 60}


def load(sym, tf):
    f = f"{CACHE}/{sym}_{tf}.parquet"
    assert os.path.exists(f), f"Moon Dev: missing cache {f}, run prep.py first"
    return pd.read_parquet(f)


def indicators(d):
    """Moon Dev: his 3 lines + ATR (only used to make slope / distance comparable across coins)."""
    c = d["close"]
    out = pd.DataFrame(index=d.index)
    out["ema9"] = c.ewm(span=9, adjust=False).mean()
    out["sma20"] = c.rolling(20).mean()
    out["sma200"] = c.rolling(200).mean()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(), (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean().replace(0, np.nan)   # Moon Dev: dead flat outage bars give ATR 0, treat as no data
    return out


def arrays(d, side):
    """Moon Dev: numpy arrays in 'long space'. side=-1 flips prices so a short is a long on -price."""
    ind = indicators(d)
    s = float(side)
    if side == 1:
        o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    else:
        o, h, l, c = -d["open"].values, -d["low"].values, -d["high"].values, -d["close"].values
    a = dict(o=o, h=h, l=l, c=c, ema9=s * ind["ema9"].values, sma20=s * ind["sma20"].values,
             sma200=s * ind["sma200"].values, atr=ind["atr"].values)
    k = SLOPE_LOOKBACK
    sl = np.full(len(c), np.nan)
    sl[k:] = (a["sma20"][k:] - a["sma20"][:-k]) / k
    a["slope"] = sl / a["atr"]                                  # 20 SMA slope, ATR per bar
    a["ext"] = (c - a["sma20"]) / a["atr"]                       # extension from the 20 SMA in ATRs
    a["piv"] = pivot_lows(l)
    return a


def pivot_lows(l):
    """Moon Dev: pivot low at bar j-2, CONFIRMED at bar j (2 bars each side). Value lands on bar j, no peeking."""
    n = len(l)
    p = np.full(n, np.nan)
    m = (l[2:-2] < l[:-4]) & (l[2:-2] < l[1:-3]) & (l[2:-2] <= l[3:-1]) & (l[2:-2] <= l[4:])
    idx = np.where(m)[0] + 2
    p[idx + 2] = l[idx]
    return p


def htf_trend(d, tf, side):
    """Moon Dev: higher time frame agrees = last CLOSED HTF bar has close beyond the 20, 9 beyond 20, 20 sloping our way.
    Mapped onto the entry TF bar close times (no look-ahead)."""
    htf = HTF[tf]
    hd = load_any(d.attrs["sym"], htf)
    a = arrays(hd, side)
    ok = (a["c"] > a["sma20"]) & (a["ema9"] > a["sma20"]) & (a["slope"] > 0)
    close_t = hd.index + pd.Timedelta(minutes=TF_MIN[htf])
    s = pd.Series(ok, index=close_t)
    ltf_close = d.index + pd.Timedelta(minutes=TF_MIN[tf])
    return s.reindex(ltf_close, method="ffill").fillna(False).values.astype(bool)


_RAW = {}


def load_any(sym, tf):
    """Moon Dev: cached 5m+ bars from disk, 1m / 2m built from the raw 1m file on the fly (too big to cache)."""
    if tf in ("1m", "2m"):
        key = (sym, tf)
        if key not in _RAW:
            import prep
            m1 = prep.load_1m(sym + "")
            _RAW[(sym, "1m")] = m1
            _RAW[(sym, "2m")] = prep.resample(m1, "2min")
        d = _RAW[key]
    else:
        d = load(sym, tf)
    d.attrs["sym"] = sym
    return d


@njit(cache=True)
def leg_features(c, l, sma20, ema9, atr, slope, ext, zone, accel_thr):
    """Moon Dev: per bar, inside the current trend leg (20 SMA sloping our way):
    touches = held touches of the 20 SMA BEFORE this bar, pb_idx = which pullback this is since the last acceleration
    (ext > accel_thr ATR), -1 if no acceleration yet in this leg."""
    n = len(c)
    touches = np.zeros(n, np.int64)
    pb_idx = np.full(n, -1, np.int64)
    t_cnt = 0
    in_touch = False
    accel = False
    pb = 0
    in_zone = False
    for i in range(1, n):
        if not (slope[i] > 0):
            t_cnt = 0; in_touch = False; accel = False; pb = 0; in_zone = False
            continue
        touches[i] = t_cnt
        held = (l[i] <= sma20[i] + 0.10 * atr[i]) and (c[i] > sma20[i])
        if held and not in_touch:
            t_cnt += 1
        in_touch = held
        if ext[i] > accel_thr:
            accel = True; pb = 0
        if zone[i] and not in_zone:
            pb += 1
        in_zone = zone[i]
        if accel:
            pb_idx[i] = pb
    return touches, pb_idx


@njit(cache=True)
def sim(o, h, l, c, sma200, piv, sig, ent_lvl, stop_lvl, exit_mode, rmult, use200, block, max_hold, cf, overlap, eod):
    """Moon Dev trade loop (long space).
    sig[t] = setup bar t is done. ent_lvl[t] = buy-stop level (nan = buy next bar open). stop_lvl[t] = stop.
    exit_mode 0 = fixed R target, 1 = trail stop under each new confirmed pivot low.
    use200: if the 200 SMA is above entry it is the target (Target 1 in the video). block>0: skip if 200 is closer than block*R.
    Fills: gaps fill at the open; inside a bar we guess the OHLC path (nearer extreme first), same rule for every test."""
    n = len(c)
    out = np.full((n, 8), np.nan)
    k = 0
    busy = -1
    for t in range(n - 2):
        if not sig[t]:
            continue
        if (not overlap) and t < busy:
            continue
        e = t + 1
        if np.isnan(ent_lvl[t]):
            entry = o[e]
        else:
            if h[e] < ent_lvl[t]:
                continue                       # Moon Dev: trigger bar high never broken next bar, no trade
            entry = max(o[e], ent_lvl[t])
        stop = stop_lvl[t]
        risk = entry - stop
        if not (risk > 0):
            continue
        s2 = sma200[t]
        in_path = use200 and (s2 > entry)
        if block > 0 and s2 > entry and (s2 - entry) < block * risk:
            continue                           # Moon Dev: 200 SMA right in the way, skip (his rule)
        tgt = np.inf
        if exit_mode == 0:
            tgt = entry + rmult * risk
        if in_path:
            tgt = s2
        ex = np.nan
        j = e
        last = min(n - 1, e + max_hold)
        at_open = entry == o[e]
        while True:
            if j > e and exit_mode == 1 and piv[j - 1] > stop and piv[j - 1] < c[j - 1]:
                stop = piv[j - 1]
            low_first = (o[j] - l[j]) < (h[j] - o[j])   # Moon Dev: OHLC path guess, the nearer extreme came first
            if j > e and o[j] <= stop:
                ex = o[j]
                break
            if j > e and o[j] >= tgt:
                ex = o[j]
                break
            if j == e and low_first and not at_open:
                # Moon Dev: the dip came BEFORE our buy stop filled, then O-L-H-C: target at H, stop only if the close falls through
                if h[j] >= tgt:
                    ex = tgt
                    break
                if c[j] <= stop:
                    ex = stop
                    break
            elif low_first:
                if l[j] <= stop:
                    ex = stop
                    break
                if h[j] >= tgt:
                    ex = tgt
                    break
            else:
                if h[j] >= tgt:
                    ex = tgt
                    break
                if l[j] <= stop:
                    ex = stop
                    break
            if j >= last or eod[j]:
                ex = c[j]
                break
            j += 1
        busy = j + 1
        r = (ex - entry) / risk
        cost = cf * (abs(entry) + abs(ex))
        out[k, 0] = t; out[k, 1] = e; out[k, 2] = j
        out[k, 3] = entry; out[k, 4] = ex; out[k, 5] = r
        out[k, 6] = (ex - entry - cost) / risk
        out[k, 7] = (ex - entry) / abs(entry)
        k += 1
    return out[:k]


def to_frame(res, idx, side, sym, tf, setup):
    """Moon Dev: numba rows -> trade table with real prices back in normal space."""
    if len(res) == 0:
        return pd.DataFrame()
    t = pd.DataFrame(res, columns=["t_sig", "t_in", "t_out", "entry", "exit", "R", "R_net", "ret"])
    for col in ["t_sig", "t_in", "t_out"]:
        t[col] = t[col].astype(int)
    t["time_in"] = idx[t["t_in"].values]
    t["time_out"] = idx[t["t_out"].values]
    t["entry"] = t["entry"].abs(); t["exit"] = t["exit"].abs()
    t["ret_net"] = t["ret"] - 2 * COST_SIDE
    t["side"] = "long" if side == 1 else "short"
    t["symbol"] = sym; t["tf"] = tf; t["setup"] = setup
    return t


def summarize(t, col="R_net"):
    """Moon Dev: the standard stat block. Equity = 1% risk per trade, trades in exit order (sizing: video leaves this out)."""
    if t is None or len(t) == 0:
        return {"trades": 0, "win_rate": 0, "avg_R": 0, "profit_factor": 0, "total_return_pct": 0, "max_dd_pct": 0, "sharpe": 0}
    r = t[col].values
    t = t.sort_values("time_out")
    rs = t[col].values
    eq = np.cumprod(1 + 0.01 * rs)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    daily = pd.Series(0.01 * rs, index=t["time_out"].values).groupby(pd.Grouper(freq="1D")).sum()
    sh = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    gw, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": int(len(r)), "win_rate": round(float((r > 0).mean() * 100), 2), "avg_R": round(float(r.mean()), 4),
            "profit_factor": round(float(gw / gl), 3) if gl > 0 else 99.0,
            "total_return_pct": round(float((eq[-1] - 1) * 100), 2), "max_dd_pct": round(float(dd * 100), 2),
            "sharpe": round(float(sh), 3)}
