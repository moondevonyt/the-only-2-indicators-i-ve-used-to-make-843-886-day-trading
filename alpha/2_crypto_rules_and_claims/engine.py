"""Moon Dev: turn one config + one coin into real trades. Used by the sweep, the baselines and the final report."""
import numpy as np
import pandas as pd
import core
import setups

_A = {}


def get(sym, tf, side):
    """Moon Dev: arrays + higher-TF trend + leg features, cached per worker."""
    k = (sym, tf, side)
    if k not in _A:
        d = core.load_any(sym, tf)
        a = core.arrays(d, side)
        a["idx"] = d.index
        a["htf_ok"] = core.htf_trend(d, tf, side) if tf in core.HTF else np.ones(len(d), bool)
        a["touches"], a["pb_idx"] = setups.leg(a)
        a["eod"] = np.zeros(len(d), bool)
        _A[k] = a
    return _A[k]


def clear():
    _A.clear()
    core._RAW.clear()


def et_minutes(idx, tf):
    """Moon Dev: bar CLOSE time in US Eastern (DST aware) as minutes after midnight, and weekday."""
    t = (idx + pd.Timedelta(minutes=core.TF_MIN[tf])).tz_convert("America/New_York")
    return (t.hour * 60 + t.minute).values, t.weekday.values


SESSION = {"1m": (570, 600), "2m": (600, 720), "5m": (720, 960)}   # 9:30-10, 10-12, 12-16 ET (0:20 in the video)


def run_sim(a, sig, ent, stop, spec, tf, sym, side, overlap=False, eod=None):
    mode, rm = setups.EXITS[spec["exit"]]
    res = core.sim(a["o"], a["h"], a["l"], a["c"], a["sma200"], a["piv"], sig.astype(np.bool_),
                   ent.astype(np.float64), stop.astype(np.float64), mode, rm, True, float(spec.get("block200_R", 0.0)),
                   core.MAX_HOLD[tf], core.COST_SIDE, overlap, a["eod"] if eod is None else eod)
    t = core.to_frame(res, a["idx"], side, sym, tf, spec["setup"])
    if len(t):
        t["risk_atr"] = (t["entry"] - np.abs(stop[t["t_sig"].values])).abs() / a["atr"][t["t_sig"].values]
    return t


def signals(a, spec):
    s = spec["setup"]
    if s.startswith("pullback"):
        return setups.pullback_signals(a, spec["slope_min"], spec["htf_align"], spec.get("skip_first", 0), a["htf_ok"], a["pb_idx"])
    if s == "base":
        return setups.base_signals(a, spec["base_bars"], spec["base_width_atr"], spec["htf_align"], a["htf_ok"])
    return setups.swing_signals(a, spec["trend"], spec["max_retrace_bars"])


def trades_for(sym, spec):
    """Moon Dev: all real trades (long + short mirror) for one coin and one config."""
    out = []
    for side in (1, -1):
        if spec["setup"] == "pullback_us_session":
            for tf, (m0, m1) in SESSION.items():
                a = get(sym, tf, side)
                mins, wd = et_minutes(a["idx"], tf)
                sig, ent, stop = signals(a, spec)
                sig = sig & (mins > m0) & (mins <= m1) & (wd < 5)
                eod = (mins >= 960) & (mins - core.TF_MIN[tf] < 960)       # Moon Dev: day trader, flat by 16:00 ET
                out.append(run_sim(a, sig, ent, stop, spec, tf, sym, side, eod=eod))
        else:
            a = get(sym, spec["tf"], side)
            sig, ent, stop = signals(a, spec)
            out.append(run_sim(a, sig, ent, stop, spec, spec["tf"], sym, side))
    out = [t for t in out if len(t)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def random_baseline(sym, spec, real, seed):
    """Moon Dev: same coin, same TF, same count, same sides, same stop size (in ATR), same exits, RANDOM entry bars."""
    rng = np.random.default_rng(seed)
    out = []
    for side_name, side in (("long", 1), ("short", -1)):
        rs = real[real["side"] == side_name]
        for tf, g in rs.groupby("tf"):
            a = get(sym, tf, side)
            n = len(a["c"])
            ok = np.where(np.isfinite(a["atr"]) & np.isfinite(a["sma20"]))[0]
            ok = ok[ok < n - 2]
            # Moon Dev: random bars from the same period (IS or OOS) as the real trades
            is_real = g["time_in"] < core.OOS_START
            pos = a["idx"][ok] < core.OOS_START
            pick = np.r_[rng.choice(ok[pos], int(is_real.sum())) if pos.any() and is_real.any() else [],
                         rng.choice(ok[~pos], int((~is_real).sum())) if (~pos).any() and (~is_real).any() else []].astype(int)
            sig = np.zeros(n, bool); sig[pick] = True
            ratr = np.full(n, np.nan)
            ratr[pick] = rng.choice(g["risk_atr"].replace([np.inf, -np.inf], np.nan).dropna().values, len(pick))
            stop = a["c"] - ratr * a["atr"]
            res = core.sim(a["o"], a["h"], a["l"], a["c"], a["sma200"], a["piv"], sig, np.full(n, np.nan), stop,
                           setups.EXITS[spec["exit"]][0], setups.EXITS[spec["exit"]][1], True,
                           float(spec.get("block200_R", 0.0)), core.MAX_HOLD[tf], core.COST_SIDE, True, a["eod"])
            out.append(core.to_frame(res, a["idx"], side, sym, tf, "random"))
    out = [t for t in out if len(t)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sums(t):
    """Moon Dev: additive stats so we can pool coins without keeping every trade."""
    row = {}
    for per, m in (("all", None), ("is", True), ("oos", False)):
        x = t if m is None else t[(t["time_in"] < core.OOS_START) == m]
        for col, tag in (("R_net", "n"), ("R", "g")):
            r = x[col].values if len(x) else np.array([])
            row[f"{per}_{tag}_cnt"] = len(r)
            row[f"{per}_{tag}_win"] = int((r > 0).sum())
            row[f"{per}_{tag}_sum"] = float(r.sum())
            row[f"{per}_{tag}_gw"] = float(r[r > 0].sum())
            row[f"{per}_{tag}_gl"] = float(-r[r < 0].sum())
            row[f"{per}_{tag}_log"] = float(np.log1p(0.01 * np.maximum(r, -99)).sum())
    return row
