"""
Moon Dev's Edge Finder dashboard builder.
Reads the 3 real result.json files in ../alpha, trims them, and bakes them into index.html.
Every number on the dashboard comes from these files. Nothing is made up here.
Run: python3 build_dashboard.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALPHA = HERE.parent / "alpha"
P1 = json.load(open(ALPHA / "1_tsla_exact_rules" / "result.json"))
P2 = json.load(open(ALPHA / "2_crypto_rules_and_claims" / "result.json"))
P3 = json.load(open(ALPHA / "3_moondev_edge_layer" / "result.json"))
BOT = (ALPHA / "3_moondev_edge_layer" / "bot_ma_pullback_flow.py").read_text()
print("🌙 Moon Dev: loaded 3 result.json files")


def rnd(v, d=4):
    return round(v, d) if isinstance(v, float) else v


def num(s):
    return float(s.replace("+", ""))


# ---------- package 1: TSLA ----------
s1 = P1["stats"]
P1_COLS = ["setup", "gap", "flat", "steep", "align", "trig", "skip", "tmin", "exit", "base_n", "base_w"]
P1_VALS = ["trades", "win_rate", "avg_R", "profit_factor", "return_pct", "oos_trades", "oos_avg_R",
           "oos_profit_factor", "oos_return_pct", "is_return_0cost"]
p1_grid = [[x["params"].get(c) for c in P1_COLS] + [rnd(x.get(v)) for v in P1_VALS] for x in P1["grid"]]

c1 = {c["timestamp"] + "|" + c["claim"][:20]: c for c in P1["claims"]}
cl1 = P1["claims"]
gap_txt = cl1[4]["result"]
gap_buckets = [{"b": m[0], "pct": int(m[1]), "n": int(m[2])}
               for m in re.findall(r"([<>]?[\d.]+(?:-[\d.]+)?%) gap (\d+)% of (\d+) days", gap_txt)]
gap_bp = [int(x) for x in re.findall(r"\+(\d+) bp/day", cl1[4]["verdict"])]  # IS, OOS
gap_with = re.search(r"with gap avg R \+([\d.]+) \(n=(\d+)\) vs against gap \+([\d.]+) \(n=(\d+)\)", gap_txt).groups()
ext_txt = cl1[2]["result"]
ext_b = [{"b": m[0].strip(), "bp": num(m[1]), "n": int(m[2])}
         for m in re.findall(r"(in zone <0\.5|[\d.]+-[\d.]+|>[\d.]+ ATR): ([+-][\d.]+) bp \(n=(\d+)\)", ext_txt)]
tch = [{"b": m[0], "R": num(m[1]), "n": int(m[2])}
       for m in re.findall(r"(\d\+?) touches: ([+-][\d.]+)R \(n=(\d+)\)", cl1[1]["result"])]
slope1 = re.search(r"flat \(<0\.03 ATR/bar\) avg R ([+-][\d.]+) \(n=(\d+)\) vs sloped 0\.03-0\.25 avg R ([+-][\d.]+); "
                   r"steep>0\.25 avg R ([+-][\d.]+)", cl1[0]["result"]).groups()
first1 = re.search(r"avg R ([+-][\d.]+) \(n=(\d+)\), win ([\d.]+)%; other pullbacks avg R ([+-][\d.]+) \(n=(\d+)\)",
                   cl1[3]["result"]).groups()
ma200_1 = re.search(r"1h 200: rejected ([\d.]+)% \(n=(\d+)\) vs placebo ([\d.]+)%.*?daily 200.*?rejected ([\d.]+)% "
                    r"\(n=(\d+)\) vs placebo ([\d.]+)%.*?hit it ([\d.]+)% of the time \(n=(\d+)\)",
                    cl1[5]["result"]).groups()
rc = s1["control_random_entry"]

p1 = {
    "headline": P1["headline"], "title": P1["title"], "market": P1["market"],
    "is": s1["in_sample"], "oos": s1["out_of_sample"], "costs": s1["costs"], "sizing": s1["sizing"],
    "chosen": s1["chosen_on_is"], "literal": s1["literal_every_rule_stacked"], "noma": s1["control_no_ma_filters"],
    "nogap": s1["control_no_gap_bias"], "rand": rc, "gridpos": s1["grid_positive"],
    "exact": s1["exact_params"], "claims": P1["claims"], "takeaways": P1["takeaways"],
    "eq": P1["equity_curve"], "eqc": P1["equity_curve_chosen"], "trades": P1["trades"],
    "trades_chosen": P1["trades_chosen"], "replay": P1["price_replay"],
    "grid": {"cols": P1_COLS + P1_VALS, "rows": p1_grid}, "grid_note": P1["grid_note"],
    "parsed": {"gap": gap_buckets, "gap_bp": gap_bp, "gap_with": gap_with, "ext": ext_b, "touch": tch,
               "slope": slope1, "first": first1, "ma200": ma200_1},
}

# ---------- package 2: crypto ----------
s2 = P2["stats"]
P2_COLS = ["setup", "tf", "slope_min", "block200_R", "exit", "htf_align", "skip_first", "base_bars",
           "base_width_atr", "trend", "max_retrace_bars", "market"]
P2_VALS = ["trades", "win_rate", "avg_R", "profit_factor", "is_avg_R", "oos_avg_R", "oos_trades", "avg_R_zero_cost"]
p2_grid = [[x["params"].get(c) for c in P2_COLS] + [rnd(x.get(v), 3) for v in P2_VALS] for x in P2["grid"]]

# share of configs that made money OOS (pooled ALL market rows), computed from the real grid
share = {}
for st in ["pullback", "base", "swing_reclaim20", "pullback_us_session"]:
    rows = [x for x in P2["grid"] if x["params"]["market"] == "ALL" and x["params"]["setup"] == st]
    share[st] = {"n": len(rows), "pos": sum(1 for x in rows if (x["oos_avg_R"] or 0) > 0)}
# zero cost avg R by time frame (pullback, ALL market, mean over every swept config)
tf0 = {}
for tf in ["5m", "15m", "1h", "4h"]:
    rows = [x["avg_R_zero_cost"] for x in P2["grid"]
            if x["params"]["market"] == "ALL" and x["params"]["setup"] == "pullback" and x["params"]["tf"] == tf]
    tf0[tf] = {"R": round(sum(rows) / len(rows), 4), "n": len(rows)}

p2 = {
    "headline": P2["headline"], "title": P2["title"], "market": P2["market"],
    "is": s2["in_sample"], "oos": s2["out_of_sample"], "zero": s2["zero_cost"],
    "rand": s2["random_entries_same_exits"], "costs": s2["costs"], "params": s2["params"],
    "setups": P2["setups"], "bh": P2["buy_and_hold_pct"], "spot": P2["spot_cross_check"],
    "ct": P2["claim_tables"], "ma200": P2["ma200_events"], "claims": P2["claims"], "takeaways": P2["takeaways"],
    "left_out": P2["left_out_by_video"], "eq": P2["equity_curve"], "trades": P2["trades"],
    "replay": P2["price_replay"], "grid": {"cols": P2_COLS + P2_VALS, "rows": p2_grid},
    "share": share, "tf0": tf0,
}

# ---------- package 3: Moon Dev edge layer ----------
s3 = P3["stats"]
buckets = [[x["params"]["market"], x["params"]["feature"], x["params"]["bucket"], x["params"]["period"],
            x["trades"], x["avg_R"], x.get("avg_R_zero_cost")]
           for x in P3["grid"] if x["params"]["view"] == "feature_bucket"]
better = [[x["params"]["market"], x["params"]["version"], x["params"]["filters"], x["params"]["period"],
           x["trades"], x["win_rate"], x["avg_R"], x.get("avg_R_zero_cost")]
          for x in P3["grid"] if x["params"]["view"] == "better_version"]
p3 = {
    "headline": P3["headline"], "title": P3["title"], "market": P3["market"], "stats": s3,
    "vbm": P3["versions_by_market"], "buckets": buckets, "better": better, "claims": P3["claims"],
    "takeaways": P3["takeaways"], "trades": P3["trades"], "replay": P3["price_replay"],
}

# ---------- the unified claims scoreboard (verdicts combine the agents' verdicts, proof = their numbers) ----------
ct1h = P2["claim_tables"]["1h"]
sb = ct1h["slope_buckets"]
ma200_1h = {e["line"]: e for e in P2["ma200_events"] if e["tf"] == "1h"}
plac_rej = [e["reject_pct"] for k, e in ma200_1h.items() if k != "sma200"]
plac_mag = [e["magnet_pct"] for k, e in ma200_1h.items() if k != "sma200"]
sw = P2["setups"]["swing_reclaim20"]
us = P2["setups"]["pullback_us_session"]
vb = P3["versions_by_market"]
clv = {(b[0], b[2], b[3]): b for b in buckets if b[1] == "clv"}
tail3 = re.search(r"no tail ([+-]?[\d.]+)R vs bottoming tail ([+-]?[\d.]+)R", P3["claims"][1]["result"]).groups()


def f2(v, d=2):
    return f"{v:+.{d}f}"


claims = [
    {"ts": "0:01", "ma": True, "claim": "Two moving averages are all I need to make money",
     "verdict": "BUSTED", "proof": f"{round(rc['exact']['is']['share_random_beats_real'] * 100)}%",
     "proof_label": "of random-entry runs beat his exact TSLA rules (in-sample)",
     "tsla": P1["claims"][6]["result"], "crypto": P2["headline"]["value"],
     "chart": {"title": "Avg bp a trade, his exact rules vs random entries (same day, side, hold)",
               "labels": ["Real IS", "Random IS", "Real OOS", "Random OOS"],
               "values": [rc["exact"]["is"]["real_bp"], rc["exact"]["is"]["random_bp"],
                          rc["exact"]["oos"]["real_bp"], rc["exact"]["oos"]["random_bp"]], "unit": "bp"}},
    {"ts": "0:10", "ma": True, "claim": "Rule 1: never trade a flat 20 SMA, it is chop",
     "verdict": "BUSTED", "proof": f2(sb["flat (0-0.03)"]["avg_R"], 3) + "R",
     "proof_label": "flat-slope trades on crypto 1h (before costs). Sloped buckets did no better",
     "tsla": P1["claims"][0]["result"], "crypto": P2["claims"][0]["result"],
     "chart": {"title": "Crypto 1h, avg R a trade (before costs) by 20 SMA slope",
               "labels": [k.split(" ")[0] for k in sb], "values": [v["avg_R"] for v in sb.values()],
               "unit": "R", "hi": 1}},
    {"ts": "0:11", "ma": True, "claim": "More touches of the 20 SMA that held = more reliable",
     "verdict": "BUSTED", "proof": f"{ct1h['touches']['0']['win_rate']}% / {ct1h['touches']['3+']['win_rate']}%",
     "proof_label": "win rate with 0 touches vs 3+ touches (crypto 1h). Flat",
     "tsla": P1["claims"][1]["result"], "crypto": P2["claims"][2]["result"],
     "chart": {"title": "Crypto 1h, avg R a trade (before costs) by prior touches",
               "labels": list(ct1h["touches"].keys()), "values": [v["avg_R"] for v in ct1h["touches"].values()],
               "unit": "R"}},
    {"ts": "0:12", "ma": True, "claim": "The best slope is about 45 degrees",
     "verdict": "BUSTED", "proof": f2(sb["very steep (>0.25)"]["avg_R"], 3) + "R",
     "proof_label": "the STEEPEST bucket was best on crypto 1h, not the middle",
     "tsla": "steep > 0.25 ATR/bar avg R " + slope1[3] + " was the best TSLA bucket", "crypto": P2["claims"][1]["result"],
     "chart": {"title": "Crypto 1h, avg R a trade (before costs) by slope",
               "labels": [k.split(" ")[0] if "very" not in k else "v.steep" for k in sb],
               "values": [v["avg_R"] for v in sb.values()], "unit": "R", "hi": 3}},
    {"ts": "0:13", "ma": True, "claim": "The entry zone is between the 9 EMA and the 20 SMA",
     "verdict": "MIXED", "proof": f2(ct1h["entry_kind"]["zone"]["avg_R"], 3) + "R",
     "proof_label": "in the zone vs " + f2(ct1h["entry_kind"]["shallow (above 9)"]["avg_R"], 3) + "R shallow, "
                    + f2(ct1h["entry_kind"]["deep (under 20)"]["avg_R"], 3) + "R deep (crypto 1h)",
     "tsla": "not split out on TSLA", "crypto": P2["claims"][7]["result"],
     "chart": {"title": "Crypto 1h, avg R a trade (before costs) by entry spot",
               "labels": ["zone", "shallow", "deep"], "values": [v["avg_R"] for v in ct1h["entry_kind"].values()],
               "unit": "R", "hi": 0}},
    {"ts": "0:13", "ma": True, "claim": "Far from the 20 SMA = exit zone, not entry zone",
     "verdict": "BUSTED", "proof": f"{ext_b[-1]['bp']:+.2f} bp",
     "proof_label": "next 30 min on TSLA when stretched over 3.5 ATR. In the zone: " + f"{ext_b[0]['bp']:+.2f} bp. Backwards",
     "tsla": P1["claims"][2]["result"], "crypto": P2["claims"][3]["result"],
     "chart": {"title": "TSLA 5m, next 30 min return in trend direction by stretch (bp)",
               "labels": ["zone", "0.5-1.5", "1.5-2.5", "2.5-3.5", ">3.5"], "values": [b["bp"] for b in ext_b],
               "unit": "bp", "hi": 4}},
    {"ts": "0:14", "ma": True, "claim": "After the trend speeds up, skip the first pullback",
     "verdict": "MIXED", "proof": f"{float(first1[0]):+.3f}R",
     "proof_label": f"first pullback on TSLA vs {float(first1[3]):+.3f}R for others. Crypto: no difference",
     "tsla": P1["claims"][3]["result"], "crypto": P2["claims"][4]["result"],
     "chart": {"title": "Avg R a trade, first pullback vs later",
               "labels": ["TSLA first", "TSLA later", "Crypto first", "Crypto later"],
               "values": [float(first1[0]), float(first1[3]), ct1h["first_pb"]["avg_R"], ct1h["later_pb"]["avg_R"]],
               "unit": "R"}},
    {"ts": "0:15", "ma": True, "claim": "The 200 SMA is support and resistance",
     "verdict": "BUSTED", "proof": f"{ma200_1h['sma200']['reject_pct']}%",
     "proof_label": f"of crypto 1h tags rejected. Fake lines 3-6% away: {min(plac_rej)}-{max(plac_rej)}%",
     "tsla": P1["claims"][5]["result"], "crypto": P2["claims"][5]["result"],
     "chart": {"title": "Crypto 1h, % of tags that bounce: real 200 SMA vs fake lines",
               "labels": ["200 SMA"] + [k.replace("placebo", "fake ") for k in ma200_1h if k != "sma200"],
               "values": [ma200_1h["sma200"]["reject_pct"]] + plac_rej, "unit": "%", "hi": 0, "base": 40}},
    {"ts": "0:16", "ma": True, "claim": "The 200 SMA is the magnet, use it as Target 1",
     "verdict": "BUSTED", "proof": f"{ma200_1[6]}%",
     "proof_label": f"of TSLA trades hit the 200 SMA target (n={ma200_1[7]}). Crypto magnet: {ma200_1h['sma200']['magnet_pct']}%, a coin flip",
     "tsla": P1["claims"][5]["result"], "crypto": P2["claims"][6]["result"],
     "chart": {"title": "Crypto 1h, % of times price reaches the line first",
               "labels": ["200 SMA"] + [k.replace("placebo", "fake ") for k in ma200_1h if k != "sma200"],
               "values": [ma200_1h["sma200"]["magnet_pct"]] + plac_mag, "unit": "%", "hi": 0, "base": 40}},
    {"ts": "0:20", "ma": False, "claim": "Pick the chart by time of day: 1m, 2m, 5m, 15m",
     "verdict": "NOT PROVEN", "proof": f2(us["oos_zero_cost"]["avg_R"], 2) + "R",
     "proof_label": f"crypto US session before costs, {f2(us['oos_net']['avg_R'], 2)}R after. TSLA exact rules: t-stat 1.27, not significant",
     "tsla": P1["headline"]["value"], "crypto": P2["takeaways"][3],
     "chart": {"title": "Crypto, his day-trade clock (1m/2m/5m), avg R a trade",
               "labels": ["IS 0 cost", "IS net", "OOS 0 cost", "OOS net"],
               "values": [us["is_zero_cost"]["avg_R"], us["is_net"]["avg_R"], us["oos_zero_cost"]["avg_R"],
                          us["oos_net"]["avg_R"]], "unit": "R"}},
    {"ts": "0:22", "ma": False, "claim": "Gap up = look long. Gap down = look short",
     "verdict": "MIXED", "proof": f"+{gap_bp[0]} bp",
     "proof_label": f"a day holding the gap side on 1-3% TSLA gaps (2024-25). 2026: +{gap_bp[1]} bp. It faded",
     "tsla": P1["claims"][4]["result"], "crypto": "crypto has no overnight gap, not tested",
     "chart": {"title": "TSLA: % of days the close follows the gap, by gap size",
               "labels": [g["b"] for g in gap_buckets], "values": [g["pct"] for g in gap_buckets], "unit": "%",
               "hi": 3, "base": 40}},
    {"ts": "0:24", "ma": False, "claim": "Enter above a doji or bottoming-tail bar, stop under its low",
     "verdict": "VERIFIED", "proof": f"{float(tail3[1]):+.3f}R",
     "proof_label": f"close in top third (his tail) vs {float(tail3[0]):+.3f}R bottom third, crypto 5m before costs. Best filter found. Can't pay fees",
     "tsla": "TSLA 2m OOS before costs: Q1 " + f"{clv[('tsla_2m','Q1','OOS')][6]:+.3f}R vs Q5 {clv[('tsla_2m','Q5','OOS')][6]:+.3f}R",
     "crypto": P3["claims"][1]["result"] + " (top third vs bottom third). " + P3["claims"][1]["verdict"],
     "chart": {"title": "Crypto 5m OOS, avg R before costs by where the trigger bar closed",
               "labels": ["Q1 low", "Q2", "Q3", "Q4", "Q5 high"],
               "values": [clv[("crypto_5m", q, "OOS")][6] for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]], "unit": "R", "hi": 4}},
    {"ts": "0:30", "ma": True, "claim": "Daily swing: buy when price reclaims the 20 SMA",
     "verdict": "BUSTED", "proof": f2(sw["oos_net"]["avg_R"], 2) + "R",
     "proof_label": f"a trade on new data, after costs. In-sample it was {f2(sw['is_net']['avg_R'], 2)}R. Random entries: {f2(sw['random_oos_net']['avg_R'], 2)}R",
     "tsla": "not tested on TSLA", "crypto": P2["headline"]["value"],
     "chart": {"title": "Crypto 1d swing reclaim, avg R a trade after costs",
               "labels": ["In-sample", "Out-of-sample", "Random OOS"],
               "values": [sw["is_net"]["avg_R"], sw["oos_net"]["avg_R"], sw["random_oos_net"]["avg_R"]], "unit": "R"}},
    {"ts": "0:30", "ma": False, "claim": "It works on gold, silver, crypto, anything",
     "verdict": "BUSTED", "proof": f"{share['pullback']['pos']} / {share['pullback']['n']}",
     "proof_label": "crypto pullback setups made money on new data, after costs",
     "tsla": "TSLA exact rules: not significant (see 0:01)", "crypto": P2["takeaways"][0],
     "chart": {"title": "Crypto, avg R a trade after costs, in-sample vs out-of-sample",
               "labels": ["pb IS", "pb OOS", "base IS", "base OOS", "swing IS", "swing OOS"],
               "values": [P2["setups"]["pullback"]["is_net"]["avg_R"], P2["setups"]["pullback"]["oos_net"]["avg_R"],
                          P2["setups"]["base"]["is_net"]["avg_R"], P2["setups"]["base"]["oos_net"]["avg_R"],
                          sw["is_net"]["avg_R"], sw["oos_net"]["avg_R"]], "unit": "R"}},
    {"ts": "0:36", "ma": False, "claim": "The MAs are only one part of the plan",
     "verdict": "VERIFIED", "proof": f"{vb['crypto_15m']['ma_only_OOS']['avg_R']:+.3f} → {vb['crypto_15m']['ma_plus_data_OOS']['avg_R']:+.3f}R",
     "proof_label": "crypto 15m OOS: MA only vs MA + Moon Dev flow. Extra info helps. Still loses",
     "tsla": f"TSLA 5m OOS: {vb['tsla_5m']['ma_only_OOS']['avg_R']:+.3f}R MA only vs {vb['tsla_5m']['ma_plus_data_OOS']['avg_R']:+.3f}R with flow",
     "crypto": P3["claims"][0]["result"],
     "chart": {"title": "OOS avg R after costs: MA only vs MA + taker flow + volume",
               "labels": ["c5m MA", "c5m +flow", "c15m MA", "c15m +flow", "T5m MA", "T5m +flow"],
               "values": [vb["crypto_5m"]["ma_only_OOS"]["avg_R"], vb["crypto_5m"]["ma_plus_data_OOS"]["avg_R"],
                          vb["crypto_15m"]["ma_only_OOS"]["avg_R"], vb["crypto_15m"]["ma_plus_data_OOS"]["avg_R"],
                          vb["tsla_5m"]["ma_only_OOS"]["avg_R"], vb["tsla_5m"]["ma_plus_data_OOS"]["avg_R"]], "unit": "R"}},
]
# Moon Dev's own data ideas (not in the video)
claims += [
    {"ts": "Moon Dev", "ma": False, "claim": c["claim"].replace("Moon Dev idea (not in the video): ", "").replace("Moon Dev idea: ", ""),
     "verdict": "NOT PROVEN" if "NOT PROVEN" in c["verdict"] else "BUSTED", "proof": c["result"].split(": ")[-1][:40],
     "proof_label": c["verdict"], "tsla": "n/a", "crypto": c["test"] + ". " + c["result"], "chart": None}
    for c in P3["claims"][4:6]
]

DATA = {
    "video": {"title": "The ONLY 2 Indicators I've Used to Make $843,886 Day Trading", "channel": "Emmanuel Malyarovich",
              "url": "https://www.youtube.com/watch?v=2e4WayvjLJo"},
    "p1": p1, "p2": p2, "p3": p3, "claims": claims, "bot": BOT,
}
js = json.dumps(DATA, separators=(",", ":"), allow_nan=False)
tpl = (HERE / "template.html").read_text()
out = tpl.replace("/*__DATA__*/null", js)
(HERE / "index.html").write_text(out)
print(f"🌙 Moon Dev: wrote index.html ({len(out) / 1e6:.2f} MB), {len(claims)} claims, "
      f"{len(p1_grid)} TSLA grid rows, {len(p2_grid)} crypto grid rows")
