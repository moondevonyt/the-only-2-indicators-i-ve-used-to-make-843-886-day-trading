"""Moon Dev: every single claim from the video about the 9 EMA / 20 SMA / 200 SMA, measured as a hard number
across all 16 Binance perps, longs and the short mirror, full 2020 -> 2026-10 history. Writes cache/claims.json."""
import json
import numpy as np
import pandas as pd
from numba import njit
from multiprocessing import Pool
import core, setups

TFS = ["1h", "15m"]          # Moon Dev: 1h is the main read, 15m is the check
FWD = 12                     # forward bars for extension / magnet tests (video leaves this out)


@njit(cache=True)
def first_hit(h, l, starts, up, dn, nmax):
    """Moon Dev: +1 if high reaches up first, -1 if low reaches dn first, 0 if neither or both in one bar."""
    out = np.zeros(len(starts), np.int64)
    n = len(h)
    for k in range(len(starts)):
        for j in range(starts[k] + 1, min(n, starts[k] + 1 + nmax)):
            u = h[j] >= up[k]
            d = l[j] <= dn[k]
            if u and d:
                break
            if u:
                out[k] = 1
                break
            if d:
                out[k] = -1
                break
    return out


def ma_trades(a, tf, side, sym):
    """Moon Dev: every trigger bar in trend direction (ANY slope), plus shallow and deep bars, entered over the bar high,
    stop at bar low, 2R target, no 200 rule. Each trigger judged on its own (overlap allowed)."""
    dirn = (a["ema9"] > a["sma20"]) & (a["c"] > a["sma20"]) & np.isfinite(a["slope"])
    zone = dirn & (a["l"] <= a["ema9"]) & (a["l"] >= a["sma20"] - core.ZONE_TOL_ATR * a["atr"])
    shallow = dirn & (a["l"] > a["ema9"]) & (a["slope"] > 0.03)
    deep = dirn & (a["l"] < a["sma20"] - core.ZONE_TOL_ATR * a["atr"]) & (a["slope"] > 0.03)
    out = []
    for name, sig in (("zone", zone), ("shallow", shallow), ("deep", deep)):
        res = core.sim(a["o"], a["h"], a["l"], a["c"], a["sma200"], a["piv"], sig, a["h"].astype(float), a["l"].astype(float),
                       0, 2.0, False, 0.0, core.MAX_HOLD[tf], core.COST_SIDE, True, np.zeros(len(a["c"]), np.bool_))
        t = pd.DataFrame(res[:, [0, 5, 6]], columns=["t_sig", "R", "R_net"])
        ts = t["t_sig"].astype(int).values
        t["kind"] = name; t["slope"] = a["slope"][ts]; t["touches"] = a["touches"][ts]; t["pb_idx"] = a["pb_idx"][ts]
        t["ext"] = a["ext"][ts]; t["tf"] = tf; t["side"] = side; t["sym"] = sym
        out.append(t)
    return pd.concat(out, ignore_index=True)


def bar_level(a, tf, side, sym):
    """Moon Dev: every 3rd bar. Slope vs forward trend efficiency, extension vs forward return (in trend direction)."""
    c = a["c"]; n = len(c)
    i = np.arange(250, n - 21, 3)
    absdiff = np.abs(np.diff(c, prepend=c[0]))
    cs = np.cumsum(absdiff)
    path = cs[i + 20] - cs[i]
    er = np.abs(c[i + 20] - c[i]) / np.where(path > 0, path, np.nan)
    fwd = (c[i + FWD] - c[i]) / np.abs(c[i]) * 100              # long space: up = good for this side
    up = a["slope"][i] > 0                                        # Moon Dev: keep only bars where this side IS the trend
    return pd.DataFrame({"abs_slope": np.abs(a["slope"][i]), "er": er, "ext": a["ext"][i], "fwd": fwd, "up": up,
                         "tf": tf, "side": side, "sym": sym})


def ma200_events(a, tf, side, sym):
    """Moon Dev: price under a line for 10 closes then tags it. Rejected = falls 1.5 ATR before rising 1.5 ATR past the line.
    Same test on placebo lines (200 SMA shifted +-3%, +-6%). Magnet: from 1-3 ATR under the line, does price reach the line
    before falling the same distance?"""
    rows = []
    c, h, l, atr = a["c"], a["h"], a["l"], a["atr"]
    base = a["sma200"]
    nmax = 24 if tf != "1d" else 20
    for name, mult in (("sma200", 1.0), ("placebo-6%", 0.94), ("placebo-3%", 0.97), ("placebo+3%", 1.03), ("placebo+6%", 1.06)):
        L = base * (mult if side == 1 else 2 - mult)          # Moon Dev: in short space the line is -price, shift mirrored
        below = (c < L).astype(int)
        run10 = pd.Series(below).rolling(10).sum().shift(1).values == 10
        tag = run10 & (h >= L) & np.isfinite(atr)
        st = np.where(tag)[0]
        st = st[st < len(c) - 2]
        r = first_hit(h, l, st, L[st] + 1.5 * atr[st], L[st] - 1.5 * atr[st], nmax)
        dist = L - c
        mag = np.where((dist > 1 * atr) & (dist < 3 * atr))[0][::5]
        mag = mag[mag < len(c) - 2]
        m = first_hit(h, l, mag, L[mag], c[mag] - dist[mag], nmax * 3)
        rows.append({"line": name, "tf": tf, "side": side, "sym": sym, "tags": len(st), "rejected": int((r == -1).sum()),
                     "broke": int((r == 1).sum()), "mag_n": int((m != 0).sum()), "mag_hit": int((m == 1).sum())})
    return rows


def one(sym):
    trades, bars, ev = [], [], []
    for tf in TFS + ["1d"]:
        d = core.load(sym, tf)
        for side in (1, -1):
            a = core.arrays(d, side)
            a["touches"], a["pb_idx"] = setups.leg(a)
            ev += ma200_events(a, tf, side, sym)
            if tf in TFS:
                trades.append(ma_trades(a, tf, side, sym))
                bars.append(bar_level(a, tf, side, sym))
    print(f"Moon Dev claims {sym} done", flush=True)
    return pd.concat(trades, ignore_index=True), pd.concat(bars, ignore_index=True), pd.DataFrame(ev)


def wr(x, col="R"):
    return {"n": int(len(x)), "win_rate": round(float((x[col] > 0).mean() * 100), 1) if len(x) else None,
            "avg_R": round(float(x[col].mean()), 3) if len(x) else None,
            "avg_R_net": round(float(x["R_net"].mean()), 3) if len(x) else None}


def build(T, B, E):
    cl = []
    out_tables = {}
    for tf in TFS:
        t = T[T["tf"] == tf]; b = B[B["tf"] == tf]
        z = t[t["kind"] == "zone"]
        # (a) + (b) slope buckets
        bins = [-np.inf, 0, 0.03, 0.08, 0.15, 0.25, np.inf]
        labs = ["against (<0)", "flat (0-0.03)", "low (0.03-0.08)", "mid (0.08-0.15)", "steep (0.15-0.25)", "very steep (>0.25)"]
        z = z.assign(sb=pd.cut(z["slope"], bins, labels=labs))
        sb = {k: wr(g) for k, g in z.groupby("sb", observed=True)}
        bq = b.assign(q=pd.qcut(b["abs_slope"], 5, labels=False))
        er = bq.groupby("q")["er"].mean().round(4).tolist()
        # (c) touches
        tc = z[z["slope"] > 0].assign(tb=np.minimum(z["touches"], 3))
        tch = {str(k) + ("+" if k == 3 else ""): wr(g) for k, g in tc.groupby("tb")}
        # (d) extension deciles (only bars where the side is the trend)
        bu = b[b["up"]].dropna(subset=["ext", "fwd"])
        bu = bu.assign(dq=pd.qcut(bu["ext"], 10, labels=False))
        ext = bu.groupby("dq").agg(ext_mid=("ext", "median"), fwd_pct=("fwd", "mean"), up_pct=("fwd", lambda x: (x > 0).mean() * 100)).round(4)
        # (e) first pullback after acceleration
        acc = z[z["pb_idx"] >= 1]
        first, later = wr(acc[acc["pb_idx"] == 1]), wr(acc[acc["pb_idx"] >= 2])
        # (g) zone vs elsewhere (same slope floor 0.03)
        zz = t[(t["kind"] == "zone") & (t["slope"] > 0.03)]
        kinds = {"zone": wr(zz), "shallow (above 9)": wr(t[t["kind"] == "shallow"]), "deep (under 20)": wr(t[t["kind"] == "deep"])}
        out_tables[tf] = dict(slope_buckets=sb, er_by_abs_slope_quintile=er, touches=tch,
                              extension_deciles=ext.reset_index().to_dict("records"), first_pb=first, later_pb=later, entry_kind=kinds)

    m = out_tables["1h"]; q = out_tables["15m"]
    flat = m["slope_buckets"]["flat (0-0.03)"]; sloped = [m["slope_buckets"][k] for k in ["low (0.03-0.08)", "mid (0.08-0.15)", "steep (0.15-0.25)"]]
    flat15 = q["slope_buckets"]["flat (0-0.03)"]
    er = m["er_by_abs_slope_quintile"]
    a_ok = flat["avg_R"] < min(s["avg_R"] for s in sloped) and er[0] < er[-1]
    cl.append(dict(claim="Never trade a flat / sideways 20 SMA, it is chop", timestamp="0:10",
                   test="1h, 16 coins, both sides: pullback trigger trades (over bar high, stop bar low, 2R) bucketed by 20 SMA slope (ATR/bar, 5-bar lookback). Plus forward 20-bar trend efficiency by |slope| quintile.",
                   result=f"flat-slope trades avg {flat['avg_R']}R ({flat['win_rate']}% win, n={flat['n']}) vs sloped buckets {[s['avg_R'] for s in sloped]}R. 15m flat: {flat15['avg_R']}R. Trend efficiency flattest->steepest quintile: {er[0]} -> {er[-1]}",
                   verdict="VERIFIED" if a_ok else ("MIXED" if er[0] < er[-1] else "BUSTED")))
    low, mid, hi = m["slope_buckets"]["low (0.03-0.08)"], m["slope_buckets"]["mid (0.08-0.15)"], m["slope_buckets"]["steep (0.15-0.25)"]
    vhi = m["slope_buckets"]["very steep (>0.25)"]
    b_ok = mid["avg_R"] > low["avg_R"] and mid["avg_R"] > max(hi["avg_R"], vhi["avg_R"])
    cl.append(dict(claim="Best slope is about 45 degrees; too steep = overextended, too flat = no momentum", timestamp="0:12",
                   test="Same trades, slope buckets. '45 degrees' depends on chart zoom, so slope is ATR per bar (video leaves the number out). Middle band 0.08-0.15 vs flatter and steeper.",
                   result=f"1h avg R: low {low['avg_R']}, mid {mid['avg_R']}, steep {hi['avg_R']}, very steep {vhi['avg_R']} (15m: {[q['slope_buckets'][k]['avg_R'] for k in ['low (0.03-0.08)', 'mid (0.08-0.15)', 'steep (0.15-0.25)', 'very steep (>0.25)']]})",
                   verdict="VERIFIED" if b_ok else ("MIXED" if mid["avg_R"] > low["avg_R"] or mid["avg_R"] > hi["avg_R"] else "BUSTED")))
    tch = m["touches"]
    tv = [tch[k]["avg_R"] for k in ["0", "1", "2", "3+"]]
    c_ok = tv[3] > tv[0] and tv[2] > tv[0]
    cl.append(dict(claim="More prior touches of the 20 SMA that held = more reliable bounce", timestamp="0:11",
                   test="Rising-20 pullback trades split by how many held touches of the 20 SMA came before in the same trend leg (0, 1, 2, 3+).",
                   result=f"1h avg R by touches 0/1/2/3+: {tv}; win %: {[tch[k]['win_rate'] for k in ['0', '1', '2', '3+']]}; 15m avg R: {[q['touches'][k]['avg_R'] for k in ['0', '1', '2', '3+']]}",
                   verdict="VERIFIED" if (c_ok and tv == sorted(tv)) else ("MIXED" if c_ok else "BUSTED")))
    ed = m["extension_deciles"]
    top, midd = ed[-1]["fwd_pct"], np.mean([r["fwd_pct"] for r in ed[3:7]])
    d_ok = top < midd
    cl.append(dict(claim="Far from the 20 SMA = exit zone, not entry zone", timestamp="0:13",
                   test=f"Every 3rd bar where the 20 SMA slopes this side's way: forward {FWD}-bar return (in trend direction) by extension decile ((close - 20 SMA) / ATR).",
                   result=f"1h top extension decile (median {ed[-1]['ext_mid']} ATR) fwd {top:.3f}% vs middle deciles {midd:.3f}% vs bottom decile {ed[0]['fwd_pct']:.3f}%. 15m top {q['extension_deciles'][-1]['fwd_pct']:.3f}%",
                   verdict="VERIFIED" if (d_ok and top < 0) else ("MIXED" if d_ok else "BUSTED")))
    f1, f2 = m["first_pb"], m["later_pb"]
    e_ok = f1["avg_R"] < f2["avg_R"]
    cl.append(dict(claim="Skip the first pullback after the trend accelerates; it fails more than later ones", timestamp="0:14",
                   test="Pullback trades after a 3+ ATR extension in the same leg (threshold: video leaves it out). First pullback vs 2nd and later.",
                   result=f"1h first: {f1['avg_R']}R, {f1['win_rate']}% win (n={f1['n']}); later: {f2['avg_R']}R, {f2['win_rate']}% (n={f2['n']}). 15m first {q['first_pb']['avg_R']}R vs later {q['later_pb']['avg_R']}R",
                   verdict="VERIFIED" if (e_ok and q["first_pb"]["avg_R"] < q["later_pb"]["avg_R"]) else ("MIXED" if e_ok or q["first_pb"]["avg_R"] < q["later_pb"]["avg_R"] else "BUSTED")))
    # (f) 200 SMA
    g = E.groupby(["tf", "line"])[["tags", "rejected", "broke", "mag_n", "mag_hit"]].sum()
    g["reject_pct"] = (g["rejected"] / (g["rejected"] + g["broke"]) * 100).round(1)
    g["magnet_pct"] = (g["mag_hit"] / g["mag_n"] * 100).round(1)
    f_txt, f_ok, m_ok = [], [], []
    for tf in ["1h", "1d"]:
        s = g.loc[(tf, "sma200")]
        pl = g.loc[tf].drop("sma200")
        f_txt.append(f"{tf}: 200 SMA rejects {s['reject_pct']}% of tags (n={int(s['tags'])}) vs placebo lines {pl['reject_pct'].mean():.1f}%; magnet {s['magnet_pct']}% vs placebo {pl['magnet_pct'].mean():.1f}%")
        f_ok.append(s["reject_pct"] > pl["reject_pct"].mean() + 3)
        m_ok.append(s["magnet_pct"] > pl["magnet_pct"].mean() + 3)
    cl.append(dict(claim="The 200 SMA acts as support / resistance (price bounces off it)", timestamp="0:15",
                   test="Price closes under (over) a line 10 bars, then tags it. Rejected = moves 1.5 ATR away before 1.5 ATR through. Same test on placebo lines = 200 SMA shifted +-3% / +-6%.",
                   result="; ".join(f_txt), verdict="VERIFIED" if all(f_ok) else ("MIXED" if any(f_ok) else "BUSTED")))
    cl.append(dict(claim="The 200 SMA is a magnet / natural Target 1", timestamp="0:16",
                   test="From 1-3 ATR away from the line: % of times price reaches the line before moving the same distance the other way (50% = coin flip). 200 SMA vs placebo lines.",
                   result="; ".join(f_txt), verdict="VERIFIED" if all(m_ok) else ("MIXED" if any(m_ok) else "BUSTED")))
    k = m["entry_kind"]
    g_ok = k["zone"]["avg_R"] > max(k["shallow (above 9)"]["avg_R"], k["deep (under 20)"]["avg_R"])
    cl.append(dict(claim="The 'entry zone' between the 9 EMA and 20 SMA beats entries elsewhere", timestamp="0:13",
                   test="Rising-20 trigger bars (slope > 0.03): low inside the 9/20 zone vs bar that never reached the 9 (shallow) vs bar that broke >0.25 ATR under the 20 (deep). Same entry over bar high, stop bar low, 2R.",
                   result=f"1h avg R: zone {k['zone']['avg_R']} ({k['zone']['win_rate']}%), shallow {k['shallow (above 9)']['avg_R']}, deep {k['deep (under 20)']['avg_R']}; 15m: zone {q['entry_kind']['zone']['avg_R']}, shallow {q['entry_kind']['shallow (above 9)']['avg_R']}, deep {q['entry_kind']['deep (under 20)']['avg_R']}",
                   verdict="VERIFIED" if g_ok else ("MIXED" if k["zone"]["avg_R"] > min(k["shallow (above 9)"]["avg_R"], k["deep (under 20)"]["avg_R"]) else "BUSTED")))
    ev_tab = g.reset_index().to_dict("records")
    return cl, out_tables, ev_tab


if __name__ == "__main__":
    with Pool(8) as p:
        res = p.map(one, core.SYMS)
    T = pd.concat([r[0] for r in res], ignore_index=True); B = pd.concat([r[1] for r in res], ignore_index=True); E = pd.concat([r[2] for r in res], ignore_index=True)
    cl, tables, ev = build(T, B, E)
    json.dump({"claims": cl, "tables": tables, "ma200": ev}, open(f"{core.CACHE}/claims.json", "w"), indent=1, default=float)
    for c in cl:
        print(f"Moon Dev claim [{c['verdict']}] {c['claim']}\n   {c['result']}")
