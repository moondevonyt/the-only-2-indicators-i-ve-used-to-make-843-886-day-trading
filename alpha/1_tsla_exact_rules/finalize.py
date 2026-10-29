"""
Moon Dev: build result.json (plan.md format) from summary_raw.json + parts.json + trades_cache.pkl.
Run after report.py. Adds the IS / OOS split of the random-entry control and the plain-words headline.
"""
import json, os
import numpy as np
import pandas as pd
from backtest import HERE, IS_END, FEE, SLIP, load

s = json.load(open(os.path.join(HERE, "summary_raw.json")))
p = json.load(open(os.path.join(HERE, "parts.json")))
tc = pd.read_pickle(os.path.join(HERE, "trades_cache.pkl"))
D = load()
rng = np.random.default_rng(11)


def rand_ctrl(t, reps=1000):
    """Moon Dev: same day, same side, same hold, random start minute. Net of the same costs."""
    c = D["c"]; first, last = D["dfirst"], D["dlast"]
    di = D["days"].get_indexer(t["date"]); hold = t["hold"].values.clip(1); sd = t["side"].values
    means = []
    for _ in range(reps):
        st = rng.integers(first[di], np.maximum(last[di] - hold, first[di] + 1))
        x = np.minimum(st + hold, last[di]); en, ex = c[st], c[x]
        means.append(((sd * (ex - en) - 2 * SLIP - FEE * (en + ex)) / en).mean())
    means = np.array(means); real = t["pnl"].mean()
    return dict(n=len(t), real_bp=round(real * 1e4, 1), random_bp=round(means.mean() * 1e4, 1),
                share_random_beats_real=round(float((means >= real).mean()), 3))


rc = {}
for name, key in (("exact", "t_exact"), ("chosen", "t_ch")):
    t = tc[key]
    rc[name] = {"is": rand_ctrl(t[t["date"] < IS_END]), "oos": rand_ctrl(t[t["date"] >= IS_END])}
print("Moon Dev random control:", rc)

E, C, NF, NB = s["exact"], s["chosen"], s["no_filters"], s["no_bias"]
LIT = p["literal_all_rules"]


def mini(x):
    return {k: x[k] for k in ("trades", "win_rate", "avg_R", "profit_factor", "total_return_pct", "max_dd_pct", "sharpe", "avg_pnl_bp")}


# trim the grid for the dashboard: every base row + pullback rows with block 0.3% (full grid in grid_rows.pkl)
grid = [g for g in p["grid"] if g["params"]["setup"] == "base" or (g["params"]["block"] == 0.003 and g["params"]["exit"] in ("R2", "trail"))]
for g in grid:
    if g["params"]["setup"] == "pullback":
        g["params"].pop("base_n"); g["params"].pop("base_w")
    else:
        for k in ("steep", "trig", "tmin", "block"):
            g["params"].pop(k)

# claim verdicts after reading the numbers (Moon Dev): the skip-first gap is about 1 standard error, so WEAK
claims = s["claims"]
for c in claims:
    c.pop("table", None)
    if c["timestamp"] == "0:22":
        c["verdict"] = "SUPPORTED in 2024-25, FADED in 2026 OOS (gap-direction hold 10:00-16:00 on 1-3% gaps: +65 bp/day IS, +6 bp/day OOS)"
    if c["timestamp"] == "0:15":
        c["verdict"] = "NOT SUPPORTED (real 200 rejects about as often as a fake level 1% away; target almost never hit)"
    if c["timestamp"] == "0:14":
        c["verdict"] = "WEAK (right direction, about 1 standard error, not proven)"
claims.append(dict(claim="Two moving averages are all you need (the MAs are the edge)", timestamp="0:01",
                   test="Random-entry control: same days, same side, same hold time as the real trades, random start minute, same costs. 1000 runs.",
                   result=(f"Exact rules: random beats real {rc['exact']['is']['share_random_beats_real']:.0%} of runs IS, {rc['exact']['oos']['share_random_beats_real']:.0%} OOS. "
                           f"IS-picked base breakout: random beats real {rc['chosen']['is']['share_random_beats_real']:.0%} IS, {rc['chosen']['oos']['share_random_beats_real']:.0%} OOS."),
                   verdict="NOT PROVEN: exact rules lost to random IS and beat it OOS on only 36 trades; the IS-best version lost to random 99% of runs OOS"))

TAKE = [
    f"Exact rules (gap 1%+, rising 20 SMA under price, 9 over 20, both timeframes agree, doji / tail trigger, 2 prior touches, skip first pullback after a 2.5 ATR stretch, no 200 SMA within 0.3%, trail under pivots): {E['is_']['trades']} IS trades {E['is_']['total_return_pct']:+.1f}%, {E['oos']['trades']} OOS trades {E['oos']['total_return_pct']:+.1f}% net. Small, not significant.",
    f"Stack EVERY literal rule (also the 'pullback to the 20 on the fast chart = 9 EMA on the slow chart' touch) and TSLA gives {LIT['all']['trades']} trades in 2.2 years. The rules are a filter so tight it barely trades.",
    f"The MA filters DO beat the raw pullback entry (raw: {NF['is_']['total_return_pct']:+.1f}% IS, {NF['oos']['total_return_pct']:+.1f}% OOS net). But random entries on the same days and side BEAT the exact rules IS ({rc['exact']['is']['random_bp']} bp random vs {rc['exact']['is']['real_bp']} bp real per trade) and lost to them OOS ({rc['exact']['oos']['random_bp']} vs {rc['exact']['oos']['real_bp']} bp, only 36 trades).",
    f"Best IS version ({C['name']}, picked from 16,524 combos by IS t-stat): {C['is_']['total_return_pct']:+.0f}% IS, then {C['oos']['total_return_pct']:+.1f}% OOS (PF {C['oos']['profit_factor']}). Random entries same day + side beat it 99% of runs OOS. The IS money came from the gap direction, not the MAs.",
    "Gap bias was the real piece in 2024-25: 1-2% gaps kept going 59% of days, 2-3% gaps 64%. Just holding the gap direction from 10:00 to 16:00 on 1-3% gap days made +65 bp a day IS but only +6 bp a day OOS. It faded in 2026.",
    "Extension is NOT an exit signal on TSLA: in a trend, the next 30 min return gets BIGGER the further price is from the 20 SMA (in zone -0.3 bp, over 3.5 ATR +4.8 bp). Flat vs sloped 20 SMA, touch count and the 200 SMA as support / resistance all showed no real edge vs placebo.",
    f"Costs matter: raw pullback entries make {NF['is0']['total_return_pct']:+.0f}% IS at zero cost and {NF['is_']['total_return_pct']:+.1f}% after 1 bp + half a cent per side. Tight 1m / 2m stops make the cost a big slice of 1R.",
]

result = dict(
    package="1_tsla_exact_rules",
    title="Emmanuel's exact 9 EMA / 20 SMA / 200 SMA day-trading rules on TSLA",
    market="TSLA, US stock, regular hours 09:30-16:00 ET, Nasdaq lit prints",
    data=("/Volumes/Moon 26/tick_data/databento/xnas_basic_trades/TSLA/ (TSLA_2024-07.parquet ... TSLA_2026-06.parquet + "
          "TSLA_2026-06-25_2026-09-24.parquet), publisher_id 81 only, 2024-07-01 -> 2026-09-23, 557 days, 217,186 1m bars. "
          "Degraded days 2024-07-25, 2025-12-15, 2026-01-12 skipped. Daily 200 SMA usable from " + s["d200_first"] + "."),
    headline=dict(label="Exact rules, net of costs: in-sample / out-of-sample return",
                  value=(f"{E['is_']['total_return_pct']:+.1f}% / {E['oos']['total_return_pct']:+.1f}% on {E['is_']['trades']}/{E['oos']['trades']} trades; "
                         f"t-stat {E['is_']['tstat']} / {E['oos']['tstat']} (not significant). Random entries same day + side beat it in {rc['exact']['is']['share_random_beats_real']:.0%} of runs IS, {rc['exact']['oos']['share_random_beats_real']:.0%} OOS"),
                  verdict="NOT PROVEN"),
    stats=dict(**mini(E["all"]),
               in_sample=dict(**mini(E["is_"]), zero_cost=mini(E["is0"])),
               out_of_sample=dict(**mini(E["oos"]), zero_cost=mini(E["oos0"])),
               costs="1 bp per side + $0.005 slippage per side on every fill (about 2.5-3 bp round trip at TSLA prices). Zero-cost numbers shown next to each.",
               sizing="1x notional per trade (no leverage), compounded. Video gives no size or risk per trade.",
               exact_params={k: v for k, v in s["exact"].items()} and None,
               chosen_on_is=dict(name=C["name"], params=C["params"], in_sample=mini(C["is_"]), out_of_sample=mini(C["oos"]),
                                 in_sample_0cost=mini(C["is0"]), out_of_sample_0cost=mini(C["oos0"])),
               best_pullback_on_is=None,
               literal_every_rule_stacked=dict(params=LIT["params"], in_sample=mini(LIT["is_"]), out_of_sample=mini(LIT["oos"])),
               control_no_ma_filters=dict(in_sample=mini(NF["is_"]), out_of_sample=mini(NF["oos"]),
                                          in_sample_0cost=mini(NF["is0"]), out_of_sample_0cost=mini(NF["oos0"])),
               control_no_gap_bias=dict(params="IS-best pullback, both sides every day", in_sample=mini(NB["is_"]), out_of_sample=mini(NB["oos"])),
               control_random_entry=rc,
               grid_positive=dict(is_net_positive=s["grid_pos"][0], oos_net_positive=s["grid_pos"][1], total=s["grid_pos"][2])),
    equity_curve=p["equity_curve"],
    equity_curve_note=f"IS-chosen version ({C['name']}), all trades 2024-07 -> 2026-09, net of costs, 1x notional compounded",
    trades=p["trades"],
    grid=grid,
    grid_note="IS stats in trades/win_rate/avg_R/profit_factor/return_pct, OOS stats in oos_*. Pullback rows shown with block=0.3% and exit R2 / trail only; the full 16,524-row grid is in grid_rows.pkl. Every swept number is one the video leaves out.",
    claims=claims,
    price_replay=dict(symbol="TSLA", bars=p["price_replay"]["bars"], trades=p["price_replay"]["trades"],
                      day=p["price_replay"]["day"], tf="2m", note=p["price_replay"]["note"]),
    takeaways=TAKE,
)
result["stats"].pop("exact_params"); result["stats"].pop("best_pullback_on_is")
result["stats"]["exact_params"] = {k: (None if v == float("inf") else v) for k, v in __import__("backtest").BASE.items()}
# Moon Dev: top-level equity + trades = the EXACT rules (matches the headline); the IS-chosen curve rides along
te = tc["t_exact"]
eq = np.cumprod(1 + te["pnl"].values)
result["equity_curve_chosen"] = result.pop("equity_curve"); result.pop("equity_curve_note")
result["equity_curve"] = [[pd.Timestamp(t).isoformat(), round(float(v), 5)] for t, v in zip(te["t_out"], eq)][:500]
result["equity_curve_note"] = "equity_curve = exact rules, net, 1x notional compounded. equity_curve_chosen = IS-chosen " + C["name"]
result["trades_chosen"] = result.pop("trades")[-150:]
result["trades"] = [dict(t_in=r.t_in.isoformat(), t_out=r.t_out.isoformat(), side="long" if r.side == 1 else "short",
                         entry=round(float(r.entry), 2), exit=round(float(r.exit), 2), R=round(float(r.R), 3),
                         pnl_pct=round(float(r.pnl * 100), 4), setup="pullback (exact rules)", symbol="TSLA", tf=f"{int(r.tf)}m")
                    for r in te.tail(300).itertuples()]
json.dump(result, open(os.path.join(HERE, "result.json"), "w"), indent=1, default=str)
print(f"Moon Dev: wrote result.json ({os.path.getsize(os.path.join(HERE, 'result.json')) / 1e6:.2f} MB, {len(grid)} grid rows)")
