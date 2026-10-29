"""
Moon Dev: pick params on IN-SAMPLE only, report OOS honestly, run baselines + random control + the claims,
and write result.json in the shared plan.md format. Run backtest.py first (makes grid_rows.pkl).
"""
import json, os
import numpy as np
import pandas as pd
from backtest import (HERE, BASE, IS_END, FEE, SLIP, load, build_tf, run, stats, split, signals)

rng = np.random.default_rng(7)
D = load(); B = build_tf(D)
rows = pd.read_pickle(os.path.join(HERE, "grid_rows.pkl"))
print(f"Moon Dev report: {len(rows)} grid rows, {len(D['days'])} days {D['days'][0].date()} -> {D['days'][-1].date()}")

# daily 200 SMA warm-up (we never fake it)
d200 = B["D"]["sma200_prev"]
d200_first = D["days"][np.flatnonzero(np.isfinite(d200))[0]].date() if np.isfinite(d200).any() else None
print(f"Moon Dev: daily 200 SMA first usable on {d200_first}")


def pick(setup, min_trades=60):
    c = [r for r in rows if r["params"]["setup"] == setup and r["is"]["trades"] >= min_trades]
    return max(c, key=lambda r: r["is"]["tstat"])


def P_of(row):
    P = dict(row["params"])
    for k in ("steep",):
        if P[k] is None: P[k] = np.inf
    return P


def block(t):
    i, o = split(t)
    return dict(all=stats(t), is_=stats(i), oos=stats(o), all0=stats(t, "pnl0", "R0"), is0=stats(i, "pnl0", "R0"),
                oos0=stats(o, "pnl0", "R0"))


# ---------------------------------------------------------------- the exact baseline + the IS-best versions
exact_P = dict(BASE)
t_exact = run(D, B, exact_P)
best_pb = pick("pullback"); best_base = pick("base")
P_pb = P_of(best_pb); P_bs = P_of(best_base)
t_pb = run(D, B, P_pb); t_bs = run(D, B, P_bs)
# controls: same pullback entries, no MA filters (only gap bias + touch of the 9 EMA + close over the 20)
nofilt_P = dict(BASE, filters=False, skip=0.0); t_nof = run(D, B, nofilt_P)
nobias_P = dict(P_pb, bias="none"); t_nob = run(D, B, nobias_P)
res = {"exact": block(t_exact), "best_pullback": block(t_pb), "best_base": block(t_bs), "no_filters": block(t_nof),
       "no_bias": block(t_nob)}
for k, v in res.items():
    print(f"Moon Dev {k:14s} IS {v['is_']}  OOS {v['oos']}")

# chosen = the IS-best of the two setups (by IS t-stat)
chosen_name, t_ch, P_ch = ("best_pullback", t_pb, P_pb) if best_pb["is"]["tstat"] >= best_base["is"]["tstat"] else ("best_base", t_bs, P_bs)
print(f"Moon Dev chosen on IS: {chosen_name} {P_ch}")


# ---------------------------------------------------------------- random-entry same-hold control
def random_control(t, reps=500):
    c = D["c"]; first, last = D["dfirst"], D["dlast"]
    di = D["days"].get_indexer(t["date"])
    hold = t["hold"].values.clip(1); sd = t["side"].values
    real = t["pnl"].mean()
    means = []
    for _ in range(reps):
        lo = first[di]; hi = np.maximum(last[di] - hold, lo + 1)
        s = rng.integers(lo, hi)
        x = np.minimum(s + hold, last[di])
        en, ex = c[s], c[x]
        p = (sd * (ex - en) - 2 * SLIP - FEE * (en + ex)) / en
        means.append(p.mean())
    means = np.array(means)
    return dict(real_avg_bp=round(real * 1e4, 2), random_avg_bp=round(means.mean() * 1e4, 2),
                random_p95_bp=round(np.percentile(means, 95) * 1e4, 2), pct_random_beat_real=round(float((means >= real).mean()), 3))


rc_exact = random_control(t_exact) if len(t_exact) else {}
rc_ch = random_control(t_ch)
print("Moon Dev random control exact:", rc_exact, " chosen:", rc_ch)

# ---------------------------------------------------------------- claims (Moon Dev)
claims = []
raw_P = dict(BASE, filters=False, skip=0.0, gap=0.0, exit="R2", use200=False)
t_raw = run(D, B, raw_P)            # every touch of the 9 EMA in gap direction, fixed 2R / stop / 16:00 exit


def bucket_R(t, col, bins, labels):
    g = pd.cut(t[col], bins=bins, labels=labels)
    return t.groupby(g, observed=False)["R"].agg(["count", "mean"]).round(3)


# 1. flat 20 SMA = chop  [0:10]
t_raw["sslope"] = t_raw["slope"] * t_raw["side"]
b1 = bucket_R(t_raw, "sslope", [-9, 0, 0.03, 0.06, 0.15, 0.25, 9], ["against", "flat<0.03", "0.03-0.06", "0.06-0.15", "0.15-0.25", "steep>0.25"])
print("Moon Dev claim flat:\n", b1)
flatR = b1.loc["flat<0.03", "mean"]; slopedR = t_raw[(t_raw.sslope > 0.03) & (t_raw.sslope < 0.25)]["R"].mean()
claims.append(dict(claim="Never trade a flat / sideways 20 SMA (it is chop)", timestamp="0:10",
                   test="All gap-direction touches of the 9 EMA on his time-of-day timeframes, 2R target, stop under trigger bar. Avg net R by 20 SMA slope (ATR per bar; 'flat' cut-off is a sweep, video leaves it out).",
                   result=f"flat (<0.03 ATR/bar) avg R {flatR:+.3f} (n={int(b1.loc['flat<0.03','count'])}) vs sloped 0.03-0.25 avg R {slopedR:+.3f}; steep>0.25 avg R {b1.loc['steep>0.25','mean']:+.3f}",
                   verdict="SUPPORTED" if slopedR > flatR + 0.03 else ("NOT SUPPORTED" if slopedR <= flatR else "WEAK"),
                   table={str(k): [int(v["count"]), float(v["mean"])] for k, v in b1.iterrows()}))

# 2. more touches = more reliable  [0:11]
tt = t_raw[t_raw.sslope > 0].copy()
b2 = bucket_R(tt, "touches", [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"])
print("Moon Dev claim touches:\n", b2)
claims.append(dict(claim="More prior touches of the 20 SMA that held = more reliable", timestamp="0:11",
                   test="Same raw signals with a rising (falling) 20 SMA. Avg net R by how many touch episodes held earlier in the same 20 SMA run (touch = low within 0.1 ATR of the 20 and close holds it; video leaves the definition out).",
                   result="; ".join(f"{k} touches: {v['mean']:+.3f}R (n={int(v['count'])})" for k, v in b2.iterrows()),
                   verdict="SUPPORTED" if b2["mean"].iloc[-1] > b2["mean"].iloc[0] + 0.03 else "NOT SUPPORTED",
                   table={str(k): [int(v["count"]), float(v["mean"])] for k, v in b2.iterrows()}))

# 3. extension = exit zone, not entry zone  [0:13]
b5 = B[5]; fwd_n = 30
idx_end = b5["end"]; dend = D["dend"][idx_end]
x = np.minimum(idx_end + fwd_n, dend)
fwd = D["c"][x] / b5["c"] - 1
up = (b5["slope"] > 0.03) & (b5["c"] > b5["sma20"]) & (b5["ema9"] > b5["sma20"])
dn = (b5["slope"] < -0.03) & (b5["c"] < b5["sma20"]) & (b5["ema9"] < b5["sma20"])
ok = (x > idx_end) & np.isfinite(b5["ext"])
ext_dir = np.where(up, b5["ext"], -b5["ext"]); fwd_dir = np.where(up, fwd, -fwd)
m = ok & (up | dn)
e3 = pd.DataFrame({"ext": ext_dir[m], "fwd": fwd_dir[m] * 1e4})
b3 = e3.groupby(pd.cut(e3.ext, [-9, 0.5, 1.5, 2.5, 3.5, 99], labels=["in zone <0.5", "0.5-1.5", "1.5-2.5", "2.5-3.5", ">3.5 ATR"]), observed=False)["fwd"].agg(["count", "mean"]).round(2)
print("Moon Dev claim extension:\n", b3)
claims.append(dict(claim="Far from the 20 SMA = exit zone, not entry zone", timestamp="0:13",
                   test="Every 5m bar in a trend (20 SMA sloped, price and 9 EMA on the trend side). Next 30 min return in the trend direction, bucketed by distance from the 20 SMA in ATRs (threshold is a sweep, video leaves it out).",
                   result="; ".join(f"{k}: {v['mean']:+.2f} bp (n={int(v['count'])})" for k, v in b3.iterrows()),
                   verdict="SUPPORTED" if b3["mean"].iloc[-1] < b3["mean"].iloc[0] else "NOT SUPPORTED",
                   table={str(k): [int(v["count"]), float(v["mean"])] for k, v in b3.iterrows()}))

# 4. skip the first pullback after acceleration  [0:14]
skip_P = dict(BASE, skip=0.0, tmin=0, align=1, trig="plain", block=0.0, exit="R2", use200=False, gap=0.0)
t_s = run(D, B, skip_P)
fa = t_s[t_s.first_acc]["R"]; nf = t_s[~t_s.first_acc]["R"]
claims.append(dict(claim="After the trend accelerates, skip the FIRST pullback / breakout", timestamp="0:14",
                   test="Filtered pullback signals (trend + alignment), 2R exit. Compare the first pullback after a bar more than 2.5 ATR from the 20 SMA (video leaves the number out) vs all other pullbacks.",
                   result=f"first pullback after acceleration avg R {fa.mean():+.3f} (n={len(fa)}), win {(fa>0).mean():.1%}; other pullbacks avg R {nf.mean():+.3f} (n={len(nf)}), win {(nf>0).mean():.1%}",
                   verdict="SUPPORTED" if fa.mean() < nf.mean() - 0.03 else "NOT SUPPORTED"))
print("Moon Dev claim skip-first:", claims[-1]["result"])

# 5. gap days trend (bias from the gap)  [0:22, 0:31]
Dd = B["D"]
gp = Dd["gap"]; oc = Dd["c"] / Dd["o"] - 1; eff = np.abs(Dd["c"] - Dd["o"]) / (Dd["h"] - Dd["l"])
dd = pd.DataFrame({"gap": gp, "oc": oc, "eff": eff, "date": D["days"]}).dropna()
dd["cont"] = np.sign(dd.oc) == np.sign(dd.gap)
dd["ag"] = dd.gap.abs()
b5g = dd.groupby(pd.cut(dd.ag, [0, 0.005, 0.01, 0.02, 0.03, 1], labels=["<0.5%", "0.5-1%", "1-2%", "2-3%", ">3%"]), observed=False).agg(
    days=("oc", "size"), cont_rate=("cont", "mean"), avg_eff=("eff", "mean"), avg_abs_oc=("oc", lambda s: s.abs().mean())).round(3)
print("Moon Dev claim gap:\n", b5g)
nb = t_nob.copy(); nb["with_gap"] = np.sign(nb["gap"]) == nb["side"]
wg = nb[nb.with_gap]["R"]; ag = nb[~nb.with_gap]["R"]
claims.append(dict(claim="Gap up = look long, gap down = look short (gap days trend)", timestamp="0:22",
                   test="Per day: does the 09:30-16:00 move follow the gap? Plus: the IS-chosen setup traded BOTH ways every day, split by with-gap vs against-gap.",
                   result=(f"close follows gap: " + "; ".join(f"{k} gap {v['cont_rate']:.0%} of {int(v['days'])} days" for k, v in b5g.iterrows())
                           + f". Trades with gap avg R {wg.mean():+.3f} (n={len(wg)}) vs against gap {ag.mean():+.3f} (n={len(ag)})"),
                   verdict="SUPPORTED" if (wg.mean() > ag.mean() + 0.03) else "NOT SUPPORTED",
                   table={str(k): [int(v["days"]), float(v["cont_rate"]), float(v["avg_eff"])] for k, v in b5g.iterrows()}))


# 6. 200 SMA = support / resistance, magnet target  [0:15-0:17]
def sr_test(level, tag, look=60, band=0.003):
    c, h, l = D["c"], D["h"], D["l"]
    out = {"real": [], "placebo": []}
    for lab, mult in (("real", [1.0]), ("placebo", [0.99, 1.01])):
        for mu in mult:
            L = level * mu
            for d in range(len(D["days"])):
                a, z = D["dfirst"][d], D["dlast"][d]
                if not np.isfinite(L[a]) or z - a < look + 30: continue
                for j in range(a + 30, z - look):   # first approach from below or above, one per day per level
                    lv = L[j]
                    if np.all(c[j - 30:j] < lv) and h[j] >= lv:
                        out[lab].append(c[j + look] < lv); break
                    if np.all(c[j - 30:j] > lv) and l[j] <= lv:
                        out[lab].append(c[j + look] > lv); break
    r, p = np.mean(out["real"]), np.mean(out["placebo"])
    return r, p, len(out["real"]), len(out["placebo"])


r1, p1, n1, m1 = sr_test(B["h200_at"], "1h")
r2, p2, n2, m2 = sr_test(B["d200_at"], "daily")
mag = t_exact[np.isfinite(t_exact["T200"])] if len(t_exact) else t_exact
hit = (mag["why"] == 2).mean() if len(mag) else np.nan
blk = t_s.copy(); blk["near"] = blk["dist200"] < 0.003
claims.append(dict(claim="The 200 SMA acts as support / resistance and is the target", timestamp="0:15",
                   test="First touch of the 1h and daily 200 SMA each day (after 30 min on one side): does price stay rejected 60 min later? Placebo = same test on the 200 shifted 1% up and down. Plus: trades with a 200 SMA within 0.3% in the way vs not.",
                   result=(f"1h 200: rejected {r1:.1%} (n={n1}) vs placebo {p1:.1%} (n={m1}); daily 200 (from {d200_first}): rejected {r2:.1%} (n={n2}) vs placebo {p2:.1%} (n={m2}). "
                           f"Exact-system trades with a 200 target hit it {hit:.1%} of the time (n={len(mag)}). Trades with a 200 within 0.3% in the way avg R {blk[blk.near]['R'].mean():+.3f} (n={int(blk.near.sum())}) vs clear {blk[~blk.near]['R'].mean():+.3f}"),
                   verdict="SUPPORTED" if (r1 > p1 + 0.05) else ("WEAK" if r1 > p1 else "NOT SUPPORTED")))
print("Moon Dev claim 200:", claims[-1]["result"])

# ---------------------------------------------------------------- result.json (Moon Dev)
def fmt_trades(t, setup):
    out = []
    for r in t.itertuples():
        out.append(dict(t_in=r.t_in.isoformat(), t_out=r.t_out.isoformat(), side="long" if r.side == 1 else "short",
                        entry=round(float(r.entry), 2), exit=round(float(r.exit), 2), R=round(float(r.R), 3),
                        pnl_pct=round(float(r.pnl * 100), 4), setup=setup, symbol="TSLA", tf=f"{int(r.tf)}m"))
    return out


setup_name = "pullback" if P_ch["setup"] == "pullback" else "base breakout"
eq = np.cumprod(1 + t_ch["pnl"].values)
ts = t_ch["t_out"].values
sel = np.unique(np.linspace(0, len(eq) - 1, min(500, len(eq))).astype(int))
equity = [[pd.Timestamp(ts[i]).isoformat(), round(float(eq[i]), 5)] for i in sel]

# price replay: the chosen setup's best OOS day (labeled as such), 2m bars with 2m indicators
t_oos = t_ch[t_ch["date"] >= IS_END]
dayR = (t_oos if len(t_oos) else t_ch).groupby("date")["R"].agg(["sum", "count"])
best_day = dayR[dayR["count"] >= 2]["sum"].idxmax() if (dayR["count"] >= 2).any() else dayR["sum"].idxmax()
b2m = B[2]; di = D["days"].get_loc(best_day); m2 = b2m["day"] == di
bars = []
for k in np.flatnonzero(m2):
    tt = D["t"][b2m["end"][k]] - pd.Timedelta(minutes=1)
    vals = [b2m[x][k] for x in ("o", "h", "l", "c", "ema9", "sma20", "sma200")]
    bars.append([tt.isoformat()] + [None if not np.isfinite(v) else round(float(v), 3) for v in vals])
replay_trades = fmt_trades(t_ch[t_ch["date"] == best_day], setup_name)

ie, oe = res["exact"]["is_"], res["exact"]["oos"]
ic, oc_ = (res[chosen_name]["is_"], res[chosen_name]["oos"])
grid_out = []
for r in rows:
    p = {k: v for k, v in r["params"].items() if k not in ("bias", "filters", "trig") or True}
    grid_out.append(dict(params={k: p[k] for k in ("setup", "gap", "flat", "steep", "align", "trig", "skip", "tmin", "block", "exit", "base_n", "base_w")},
                         trades=r["is"]["trades"], win_rate=r["is"]["win_rate"], avg_R=r["is"]["avg_R"],
                         profit_factor=r["is"]["profit_factor"], return_pct=r["is"]["total_return_pct"],
                         oos_trades=r["oos"]["trades"], oos_avg_R=r["oos"]["avg_R"], oos_profit_factor=r["oos"]["profit_factor"],
                         oos_return_pct=r["oos"]["total_return_pct"], is_return_0cost=r["is_0cost"]["total_return_pct"]))
n_pos_is = sum(1 for g in grid_out if g["return_pct"] > 0 and g["trades"] >= 30)
n_pos_oos = sum(1 for g in grid_out if g["oos_return_pct"] > 0 and g["oos_trades"] >= 15)
print(f"Moon Dev grid: {n_pos_is}/{len(grid_out)} configs positive IS net, {n_pos_oos} positive OOS net")

json.dump(dict(
    exact=res["exact"], chosen=dict(name=chosen_name, params={k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in P_ch.items()}, **res[chosen_name]),
    best_base=dict(params={k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in P_bs.items()}, **res["best_base"]),
    no_filters=res["no_filters"], no_bias=res["no_bias"], random_exact=rc_exact, random_chosen=rc_ch,
    grid_pos=[n_pos_is, n_pos_oos, len(grid_out)], d200_first=str(d200_first), replay_day=str(best_day.date()),
    claims=[{k: v for k, v in c.items()} for c in claims]), open(os.path.join(HERE, "summary_raw.json"), "w"), indent=1, default=str)
pd.to_pickle(dict(t_exact=t_exact, t_ch=t_ch, t_nof=t_nof, t_bs=t_bs), os.path.join(HERE, "trades_cache.pkl"))
print("Moon Dev: wrote summary_raw.json. Next: write_result.py builds result.json after the numbers are read.")

# parts for result.json (finalize.py adds the plain-words headline + takeaways after Moon Dev reads the numbers)
lit = dict(BASE, align=2); t_lit = run(D, B, lit)
json.dump(dict(equity_curve=equity, trades=fmt_trades(t_ch.tail(300), setup_name), grid=grid_out,
               price_replay=dict(symbol="TSLA", tf="2m", day=str(best_day.date()),
                                 note="best OOS day of the IS-chosen version (picked as the best day, so it flatters the system)",
                                 bars=bars, trades=replay_trades),
               literal_all_rules=dict(params={k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in lit.items()}, **block(t_lit))),
          open(os.path.join(HERE, "parts.json"), "w"), default=str)
print("Moon Dev literal all-rules (align=2):", stats(t_lit))
