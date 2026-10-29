"""
Moon Dev: backtest Emmanuel Malyarovich's EXACT day-trading rules on TSLA (real Databento Nasdaq lit prints).
Video: "The ONLY 2 Indicators I've Used to Make $843,886 Day Trading" (transcript.md in the repo root).

Run build_bars.py first (makes bars_1m.parquet). Then: python backtest.py
Writes result.json, grid.parquet, trades_chosen.parquet in this folder.

Every number the video does NOT give is a SWEEP and is labeled "video leaves this out".
"""
import itertools, json, os, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
TFS = [1, 2, 5, 15, 60]                 # minutes. daily is built separately
EXEC_PAIRS = {1: 2, 2: 5, 5: 15}        # [0:20] 9:30-10 -> 1m+2m, 10-12 -> 2m+5m, 12-16 -> 5m+15m
WINDOWS = {1: (0, 29), 2: (30, 149), 5: (150, 389)}  # bar END minute of session (0 = 09:30 bar)
IS_END = pd.Timestamp("2026-01-01")     # in-sample 2024-07 -> 2025-12, out-of-sample 2026-01 -> 2026-09
FEE = 0.0001                            # 1 bp per side
SLIP = 0.005                            # half a cent per side (TSLA)
TICK = 0.01

# ---------------------------------------------------------------- data (Moon Dev)
def load():
    p = os.path.join(HERE, "bars_1m.parquet")
    if not os.path.exists(p):
        raise SystemExit("Moon Dev: bars_1m.parquet missing. Run build_bars.py first (needs Moon 26 mounted).")
    m = pd.read_parquet(p)
    t = m.index
    sess = (t.hour * 60 + t.minute - 570).values.astype(np.int64)
    dates = t.normalize().tz_localize(None)
    days = pd.DatetimeIndex(dates.unique())
    day_id = days.get_indexer(dates).astype(np.int64)
    D = dict(t=t, o=m.open.values, h=m.high.values, l=m.low.values, c=m.close.values, v=m.volume.values,
             sess=sess, day=day_id, days=days)
    n = len(m)
    first = np.r_[0, np.flatnonzero(np.diff(day_id)) + 1]
    last = np.r_[first[1:] - 1, n - 1]
    D["dfirst"], D["dlast"] = first, last
    D["dend"] = last[day_id]                       # last minute index of each minute's day
    return D


def indicators(o, h, l, c):
    s = pd.Series(c)
    ema9 = s.ewm(span=9, adjust=False).mean().values
    sma20 = s.rolling(20).mean().values
    sma200 = s.rolling(200).mean().values
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).rolling(14).mean().values
    slope = (sma20 - np.r_[np.full(5, np.nan), sma20[:-5]]) / 5 / atr   # 20 SMA change per bar, in ATRs
    ext = (c - sma20) / atr                                             # extension from the 20 SMA, in ATRs
    return ema9, sma20, sma200, atr, slope, ext


@njit(cache=True)
def touch_counts(l, h, c, sma20, atr, slope):
    """Moon Dev: prior touches of the 20 SMA that HELD, inside the current run of a rising (falling) 20 SMA."""
    n = len(c)
    outL = np.zeros(n, np.int64); outS = np.zeros(n, np.int64)
    cntL = 0; cntS = 0; prevL = False; prevS = False
    for k in range(n):
        if not (slope[k] > 0): cntL = 0; prevL = False
        if not (slope[k] < 0): cntS = 0; prevS = False
        tL = (l[k] <= sma20[k] + 0.1 * atr[k]) and (c[k] >= sma20[k])
        tS = (h[k] >= sma20[k] - 0.1 * atr[k]) and (c[k] <= sma20[k])
        outL[k] = cntL - (1 if (tL and prevL) else 0)
        outS[k] = cntS - (1 if (tS and prevS) else 0)
        if tL and not prevL: cntL += 1
        if tS and not prevS: cntS += 1
        prevL = tL; prevS = tS
    return outL, outS


@njit(cache=True)
def first_after_accel(sig, accel, day):
    """Moon Dev: mark the FIRST signal after an acceleration bar (same day). [0:14] says skip it."""
    n = len(sig); out = np.zeros(n, np.bool_); armed = False
    for k in range(n):
        if k > 0 and day[k] != day[k - 1]: armed = False
        if accel[k]: armed = True
        if sig[k] and armed:
            out[k] = True; armed = False
    return out


def build_tf(D):
    """Moon Dev: RTH bars for 1m/2m/5m/15m/1h anchored at 09:30 ET, plus daily bars. Indicators run across days."""
    B = {}
    for tf in TFS:
        key = D["day"] * 1000 + D["sess"] // tf
        st = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
        en = np.r_[st[1:] - 1, len(key) - 1]
        o = D["o"][st]; c = D["c"][en]
        h = np.maximum.reduceat(D["h"], st); l = np.minimum.reduceat(D["l"], st)
        ema9, sma20, sma200, atr, slope, ext = indicators(o, h, l, c)
        day = D["day"][st]
        tL, tS = touch_counts(l, h, c, sma20, atr, slope)
        rng = h - l
        body = np.abs(c - o)
        doji = (rng > 0) & (body <= 0.25 * rng)
        btail = (rng > 0) & ((np.minimum(o, c) - l) >= 0.5 * rng)
        ttail = (rng > 0) & ((h - np.maximum(o, c)) >= 0.5 * rng)
        # confirmed 3-bar pivot low/high (bar k-1), known at the close of bar k
        pl = np.full(len(c), np.nan); ph = np.full(len(c), np.nan)
        okl = (l[1:-1] < l[:-2]) & (l[1:-1] < l[2:]) & (day[1:-1] == day[2:])
        okh = (h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]) & (day[1:-1] == day[2:])
        pl[2:][okl] = l[1:-1][okl]; ph[2:][okh] = h[1:-1][okh]
        B[tf] = dict(o=o, h=h, l=l, c=c, end=en, endmin=D["sess"][en], day=day, ema9=ema9, sma20=sma20,
                     sma200=sma200, atr=atr, slope=slope, ext=ext, tL=tL, tS=tS, doji=doji, btail=btail,
                     ttail=ttail, pl=pl, ph=ph)
    # daily bars from the 1m bars
    f, la = D["dfirst"], D["dlast"]
    dc = D["c"][la]; do = D["o"][f]
    dh = np.maximum.reduceat(D["h"], f); dl = np.minimum.reduceat(D["l"], f)
    dsma200 = pd.Series(dc).rolling(200).mean().values
    prev_c = np.r_[np.nan, dc[:-1]]
    gap = do / prev_c - 1
    # the day after a skipped degraded day has no clean prior close: no bias that day (Moon Dev)
    days = D["days"]
    bd = np.r_[99, np.diff(days.values).astype("timedelta64[D]").astype(int)]
    wk = days.dayofweek.values
    for bad in ["2024-07-25", "2025-12-15", "2026-01-12"]:
        nxt = np.searchsorted(days.values, np.datetime64(bad))
        if nxt < len(days):
            gap[nxt] = np.nan
    B["D"] = dict(o=do, h=dh, l=dl, c=dc, sma200_prev=np.r_[np.nan, dsma200[:-1]], gap=gap,
                  sma20=pd.Series(dc).rolling(20).mean().values, ema9=pd.Series(dc).ewm(span=9, adjust=False).mean().values)
    # per-minute "last completed" values used at signal time (no look-ahead)
    idx = np.arange(len(D["c"]))
    hb = B[60]
    hi = np.searchsorted(hb["end"], idx, "right") - 1
    B["h200_at"] = np.where(hi >= 0, hb["sma200"][np.clip(hi, 0, None)], np.nan)
    B["d200_at"] = B["D"]["sma200_prev"][D["day"]]
    # per-minute pivot arrays for the trailing stop, one row per exec TF
    PL = np.full((3, len(idx)), np.nan); PH = np.full((3, len(idx)), np.nan)
    for r, tf in enumerate([1, 2, 5]):
        PL[r, B[tf]["end"]] = B[tf]["pl"]; PH[r, B[tf]["end"]] = B[tf]["ph"]
    B["PL"], B["PH"] = PL, PH
    return B


# ---------------------------------------------------------------- signals (Moon Dev)
def signals(D, B, P):
    """Return a time-sorted signal table for one parameter set P."""
    out = []
    gap = B["D"]["gap"]
    for tf, slow in EXEC_PAIRS.items():
        b, s = B[tf], B[slow]
        a, z = WINDOWS[tf]
        win = (b["endmin"] >= a) & (b["endmin"] <= z)
        si = np.searchsorted(s["end"], b["end"], "right") - 1
        si = np.clip(si, 0, None)
        g = gap[b["day"]]
        for side in (1, -1):
            if P["bias"] == "gap":
                bias = (g > P["gap"]) if side == 1 else (g < -P["gap"])
            else:  # no bias control: both sides every day
                bias = np.isfinite(g) | True
            if side == 1:
                trend = (b["slope"] > P["flat"]) & (b["slope"] < P["steep"]) & (b["c"] > b["sma20"]) & (b["ema9"] > b["sma20"])
                zone = (b["l"] <= b["ema9"]) & (b["c"] >= b["sma20"])
                shape = b["doji"] | b["btail"]
                touches = b["tL"]
                s_tr = (s["slope"][si] > P["flat"]) & (s["c"][si] > s["sma20"][si]) & (s["ema9"][si] > s["sma20"][si])
                s_touch = (b["l"] <= s["ema9"][si]) & (b["c"] >= s["sma20"][si])
                accel = (b["ext"] > P["skip"]) if P["skip"] > 0 else np.zeros(len(b["c"]), bool)
            else:
                trend = (b["slope"] < -P["flat"]) & (b["slope"] > -P["steep"]) & (b["c"] < b["sma20"]) & (b["ema9"] < b["sma20"])
                zone = (b["h"] >= b["ema9"]) & (b["c"] <= b["sma20"])
                shape = b["doji"] | b["ttail"]
                touches = b["tS"]
                s_tr = (s["slope"][si] < -P["flat"]) & (s["c"][si] < s["sma20"][si]) & (s["ema9"][si] < s["sma20"][si])
                s_touch = (b["h"] >= s["ema9"][si]) & (b["c"] <= s["sma20"][si])
                accel = (b["ext"] < -P["skip"]) if P["skip"] > 0 else np.zeros(len(b["c"]), bool)
            if P["setup"] == "pullback":
                sig = win & bias & zone
                if P["filters"]:
                    sig &= trend & (touches >= P["tmin"])
                    if P["trig"] == "doji_tail": sig &= shape
                    if P["align"] >= 1: sig &= s_tr
                    if P["align"] >= 2: sig &= s_touch
                L = b["h"] + TICK if side == 1 else b["l"] - TICK
                S = b["l"] - TICK if side == 1 else b["h"] + TICK
            else:  # base breakout / breakdown hugging the 20 SMA
                N = P["base_n"]
                hi = pd.Series(b["h"]).rolling(N).max().values; lo = pd.Series(b["l"]).rolling(N).min().values
                tight = (hi - lo) <= P["base_w"] * b["atr"]
                hug = (lo <= b["sma20"]) & (b["sma20"] <= hi)
                slope_ok = (b["slope"] > P["flat"]) if side == 1 else (b["slope"] < -P["flat"])
                sameday = pd.Series(b["day"]).rolling(N).min().values == b["day"]
                sig = win & bias & tight & hug & slope_ok & sameday
                if P["align"] >= 1: sig &= s_tr
                L = hi + TICK if side == 1 else lo - TICK
                S = lo - TICK if side == 1 else hi + TICK
            acc25 = (b["ext"] > 2.5) if side == 1 else (b["ext"] < -2.5)
            fa25 = first_after_accel(sig, acc25, b["day"])   # for the claims test (Moon Dev)
            if P["skip"] > 0:
                sig &= ~first_after_accel(sig, accel, b["day"])
            sig &= np.isfinite(b["atr"]) & np.isfinite(b["sma20"])
            k = np.flatnonzero(sig)
            if len(k) == 0:
                continue
            e = b["end"][k]
            h200 = B["h200_at"][e]; d200 = B["d200_at"][e]; x200 = b["sma200"][k]
            lev = np.vstack([h200, d200, x200])
            if side == 1:
                ahead = np.where(lev > L[k], lev, np.nan)
                near = np.nanmin(np.where(np.isfinite(ahead), ahead, np.inf), axis=0)
                tgt_lev = np.vstack([h200, d200]); tgt = np.where(tgt_lev > L[k], tgt_lev, np.inf).min(axis=0)
                block = (near - L[k]) / L[k] < P["block"]
            else:
                ahead = np.where(lev < L[k], lev, np.nan)
                near = np.nanmax(np.where(np.isfinite(ahead), ahead, -np.inf), axis=0)
                tgt_lev = np.vstack([h200, d200]); tgt = np.where(tgt_lev < L[k], tgt_lev, -np.inf).max(axis=0)
                block = (L[k] - near) / L[k] < P["block"]
            tgt = np.where(np.isfinite(tgt), tgt, np.nan) if P["use200"] else np.full(len(k), np.nan)
            keep = ~block if (P["block"] > 0 and P["filters"]) else np.ones(len(k), bool)
            out.append(pd.DataFrame({"e": e[keep], "side": side, "L": L[k][keep], "S": S[k][keep], "T200": tgt[keep],
                                     "tf": tf, "slope": b["slope"][k][keep], "touches": touches[k][keep],
                                     "ext": b["ext"][k][keep], "gap": g[k][keep], "first_acc": fa25[k][keep],
                                     "dist200": (side * (near - L[k]) / L[k])[keep]}))
    if not out:
        return pd.DataFrame(columns=["e", "side", "L", "S", "T200", "tf"])
    return pd.concat(out).sort_values("e", kind="stable").reset_index(drop=True)


@njit(cache=True)
def simulate(e, side, L, S, T200, tfrow, rmult, trail, o, h, l, c, dend, PL, PH):
    """Moon Dev: one position at a time. Buy-stop above the trigger bar, valid for one exec bar.
    Stop below the trigger low. Target = 200 SMA (1h/daily) if there, else R multiple or trail under pivots.
    Flat at the day's last minute. If stop and target hit in the same minute, the stop wins (conservative)."""
    n = len(e)
    fi = np.full(n, -1, np.int64); xi = np.full(n, -1, np.int64)
    fp = np.full(n, np.nan); xp = np.full(n, np.nan); why = np.zeros(n, np.int64)
    free = -1
    tfm = np.array([1, 2, 5])
    for q in range(n):
        if e[q] < free: continue
        last = dend[e[q]]
        stop_scan = min(e[q] + tfm[tfrow[q]], last)
        f = -1; px = 0.0
        for j in range(e[q] + 1, stop_scan + 1):
            if side[q] == 1 and h[j] >= L[q]:
                f = j; px = max(o[j], L[q]); break
            if side[q] == -1 and l[j] <= L[q]:
                f = j; px = min(o[j], L[q]); break
        if f < 0: continue
        stp = S[q]; risk = abs(px - stp)
        if risk <= 0: continue
        tg = T200[q]
        if np.isnan(tg) and rmult > 0:
            tg = px + side[q] * rmult * risk
        ex = -1; ep = np.nan; w = 0
        for j in range(f, last + 1):
            if side[q] == 1:
                if j > f and o[j] <= stp: ex = j; ep = o[j]; w = 1; break
                if l[j] <= stp: ex = j; ep = stp; w = 1; break
                if j > f and not np.isnan(tg) and h[j] >= tg: ex = j; ep = max(o[j], tg); w = 2; break
                if trail and not np.isnan(PL[tfrow[q], j]) and PL[tfrow[q], j] - 0.01 > stp: stp = PL[tfrow[q], j] - 0.01
            else:
                if j > f and o[j] >= stp: ex = j; ep = o[j]; w = 1; break
                if h[j] >= stp: ex = j; ep = stp; w = 1; break
                if j > f and not np.isnan(tg) and l[j] <= tg: ex = j; ep = min(o[j], tg); w = 2; break
                if trail and not np.isnan(PH[tfrow[q], j]) and PH[tfrow[q], j] + 0.01 < stp: stp = PH[tfrow[q], j] + 0.01
        if ex < 0: ex = last; ep = c[last]; w = 3
        fi[q] = f; xi[q] = ex; fp[q] = px; xp[q] = ep; why[q] = w
        free = ex + 1
    return fi, xi, fp, xp, why


def run(D, B, P, sig=None):
    """Moon Dev: signals -> fills -> trades table with net and zero-cost returns."""
    if sig is None:
        sig = signals(D, B, P)
    if len(sig) == 0:
        return pd.DataFrame()
    tfrow = sig["tf"].map({1: 0, 2: 1, 5: 2}).values.astype(np.int64)
    rm = {"R1": 1.0, "R2": 2.0, "R3": 3.0, "trail": 0.0, "eod": 0.0}[P["exit"]]
    fi, xi, fp, xp, why = simulate(sig["e"].values.astype(np.int64), sig["side"].values.astype(np.int64),
                                   sig["L"].values, sig["S"].values, sig["T200"].values.astype(float), tfrow, rm,
                                   P["exit"] == "trail", D["o"], D["h"], D["l"], D["c"], D["dend"], B["PL"], B["PH"])
    ok = fi >= 0
    t = sig[ok].copy()
    t["fi"], t["xi"], t["entry"], t["exit"], t["why"] = fi[ok], xi[ok], fp[ok], xp[ok], why[ok]
    sd = t["side"].values
    gross = sd * (t["exit"].values - t["entry"].values)
    risk = np.abs(t["entry"].values - t["S"].values)
    cost = 2 * SLIP + FEE * (t["entry"].values + t["exit"].values)
    t["pnl0"] = gross / t["entry"].values                  # zero cost
    t["pnl"] = (gross - cost) / t["entry"].values          # net of costs
    t["R"] = (gross - cost) / risk
    t["R0"] = gross / risk
    t["t_in"] = D["t"][t["fi"].values]; t["t_out"] = D["t"][t["xi"].values]
    t["date"] = t["t_in"].dt.tz_localize(None).dt.normalize()
    t["hold"] = t["xi"].values - t["fi"].values
    return t


def stats(t, col="pnl", rcol="R"):
    if t is None or len(t) == 0:
        return dict(trades=0, win_rate=0, avg_R=0, profit_factor=0, total_return_pct=0, max_dd_pct=0, sharpe=0, avg_pnl_bp=0, tstat=0)
    p = t[col].values
    eq = np.cumprod(1 + p)
    dd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
    daily = t.groupby("date")[col].sum()
    alld = pd.bdate_range(t["date"].min(), t["date"].max())
    daily = daily.reindex(alld, fill_value=0.0)
    sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    pos, neg = p[p > 0].sum(), -p[p < 0].sum()
    return dict(trades=int(len(p)), win_rate=round(float((p > 0).mean()), 4), avg_R=round(float(t[rcol].mean()), 4),
                profit_factor=round(float(pos / neg), 3) if neg > 0 else 99.0,
                total_return_pct=round(float((eq[-1] - 1) * 100), 2), max_dd_pct=round(float(dd * 100), 2),
                sharpe=round(float(sh), 2), avg_pnl_bp=round(float(p.mean() * 1e4), 2),
                tstat=round(float(p.mean() / p.std() * np.sqrt(len(p))), 2) if len(p) > 2 and p.std() > 0 else 0)


def split(t):
    if len(t) == 0:
        return t, t
    return t[t["date"] < IS_END], t[t["date"] >= IS_END]


# ---------------------------------------------------------------- grid (Moon Dev)
BASE = dict(setup="pullback", bias="gap", filters=True, gap=0.01, flat=0.05, steep=0.25, align=1, trig="doji_tail",
            skip=2.5, tmin=2, block=0.003, use200=True, exit="trail", base_n=10, base_w=3.0)
GRID_PB = dict(gap=[0.0, 0.01, 0.02, 0.03], flat=[0.0, 0.05, 0.1], steep=[np.inf, 0.25, 0.4], align=[0, 1, 2],
               trig=["plain", "doji_tail"], skip=[0.0, 2.5, 4.0], tmin=[0, 1, 2], block=[0.0, 0.003],
               exit=["R1", "R2", "R3", "trail"])
GRID_BASE = dict(gap=[0.0, 0.01, 0.02], flat=[0.0, 0.05, 0.1], base_n=[6, 10, 15], base_w=[2.0, 3.0, 4.0], align=[0, 1],
                 skip=[0.0, 2.5], exit=["R1", "R2", "trail"])

_D = _B = None


def _init():
    global _D, _B
    _D = load(); _B = build_tf(_D)


def _one(P):
    t = run(_D, _B, P)
    i, o = split(t)
    row = {"params": {k: (None if (isinstance(v, float) and np.isinf(v)) else v) for k, v in P.items()}}
    for name, tt in (("is", i), ("oos", o)):
        s = stats(tt); s0 = stats(tt, "pnl0", "R0")
        row[name] = s; row[name + "_0cost"] = {"avg_R": s0["avg_R"], "total_return_pct": s0["total_return_pct"],
                                               "profit_factor": s0["profit_factor"], "avg_pnl_bp": s0.get("avg_pnl_bp", 0)}
    return row


def grid_configs():
    cfgs = []
    keys = list(GRID_PB)
    for vals in itertools.product(*[GRID_PB[k] for k in keys]):
        P = dict(BASE); P.update(dict(zip(keys, vals))); cfgs.append(P)
    keys = list(GRID_BASE)
    for vals in itertools.product(*[GRID_BASE[k] for k in keys]):
        P = dict(BASE); P.update(dict(zip(keys, vals))); P["setup"] = "base"; cfgs.append(P)
    return cfgs


if __name__ == "__main__":
    t0 = time.time()
    cfgs = grid_configs()
    print(f"Moon Dev grid: {len(cfgs)} configs")
    with Pool(12, initializer=_init) as pool:
        rows = pool.map(_one, cfgs, chunksize=20)
    pd.to_pickle(rows, os.path.join(HERE, "grid_rows.pkl"))
    print(f"Moon Dev grid done in {time.time() - t0:.0f}s")
