"""
Moon Dev edge layer, step 2: score his MA pullback trades by Moon Dev's data, pick filters on
IN-SAMPLE only, then show the OUT-OF-SAMPLE truth. Writes result.json.
Splits: crypto IS 2020 -> 2024-06, OOS 2024-07 -> 2026-10.
Short datasets (funding 2024-01 -> 2026-08, liquidations 2024-06 -> 2025-12, TSLA 2024-07 -> 2026-09):
first 60% of their own span = IS, last 40% = OOS.
"""
import os
import json
import numpy as np
import pandas as pd
from common import apply_costs, stats, add_mas, run_signals
import build_crypto as bc

HERE = os.path.dirname(os.path.abspath(__file__))
CRYPTO_SPLIT = pd.Timestamp("2024-07-01", tz="UTC")
BASE = (0.3, "2.0")       # his rules, default sweep point picked BEFORE looking: slope >= 0.3 ATR / 5 bars, 2R target
CRYPTO_COST = 0.0005 + 0.0002          # 0.05% taker + 0.02% slip, per side
TSLA_COST, TSLA_SHARE = 0.0001, 0.005  # 1 bp per side + half a cent per share per side
RISK_PER_TRADE = 1.0                   # % of equity risked per trade for the equity curve (video gives none)
FEATS = {
    "imb_trig": "taker buy imbalance on the trigger bar (trade side +)",
    "imb_pb3": "taker imbalance during the 3 pullback bars (trade side +)",
    "imb12": "CVD trend over the last 12 bars (trade side +)",
    "cvd_div": "CVD divergence: price pulled back but CVD went our way (1 = yes)",
    "rvol": "trigger bar volume / last 20 bars",
    "tsize": "avg trade size on trigger bar / last 20 bars",
    "fund_crowd": "funding in bps, + = crowd already on our side",
    "liq_fuel": "same-side liquidations in the last 3 bars vs 30d normal (long liqs on a long pullback)",
    "liq_against": "other-side liquidations in the last 3 bars vs 30d normal",
    "clv": "HIS WORDS 0:25: bottoming tail = trigger bar closes near its high (long). Price action, not Moon Dev data",
    "blk3": "TSLA block prints (>=1k sh) signed shares, last 3 bars (trade side +)",
    "blk5k3": "TSLA block prints (>=5k sh) signed shares, last 3 bars (trade side +)",
}
SHORT_FEATS = {"fund_crowd", "liq_fuel", "liq_against"}


def split_6040(t):
    lo, hi = t.t_in.min(), t.t_in.max()
    cut = lo + (hi - lo) * 0.6
    return np.where(t.t_in < cut, "IS", "OOS"), cut


def bucket_table(t, feat, market):
    """Moon Dev: quintile buckets with edges from IS only, stats IS and OOS, net and zero cost."""
    d = t.dropna(subset=[feat])
    if len(d) < 100:
        return [], None
    if feat == "cvd_div":
        d = d.assign(b=d[feat].astype(int).astype(str))
        edges = None
    else:
        is_vals = d.loc[d.period == "IS", feat]
        edges = np.unique(np.quantile(is_vals, [0.2, 0.4, 0.6, 0.8]))
        d = d.assign(b=np.searchsorted(edges, d[feat].values, side="right"))
        d["b"] = d.b.map(lambda k: f"Q{k + 1}")
    rows = []
    for (b, p), g in d.groupby(["b", "period"]):
        s = stats(g.R, g.net_pct)
        s0 = stats(g.R0)
        rows.append({"params": {"market": market, "view": "feature_bucket", "feature": feat, "bucket": b,
                                "period": p, "smin": BASE[0], "tgt": BASE[1]},
                     **s, "avg_R_zero_cost": s0["avg_R"], "pf_zero_cost": s0["profit_factor"]})
    return rows, (None if edges is None else [round(float(x), 5) for x in edges])


def candidate_filters(t, feat):
    """Moon Dev: one-sided thresholds at the IS quintile edges."""
    d = t.loc[t.period == "IS", feat].dropna()
    if feat == "cvd_div":
        return [(feat, ">=", 1.0), (feat, "<=", 0.0)]
    qs = np.quantile(d, [0.2, 0.4, 0.6, 0.8])
    return [(feat, op, float(q)) for q in qs for op in (">=", "<=")]


def mask(t, f):
    feat, op, q = f
    v = t[feat]
    return (v >= q) if op == ">=" else (v <= q)


def pick_filters(t, feats, min_frac=0.3, max_n=2):
    """Moon Dev: greedy pick on IS avg R. Each filter must keep >= 30% of IS trades and beat the
    current IS avg R by >= 0.02R. Nothing here ever sees OOS."""
    is_t = t[t.period == "IS"]
    chosen, cur = [], is_t
    for _ in range(max_n):
        best, best_r = None, cur.R.mean() + 0.02
        for feat in feats:
            if is_t[feat].notna().mean() < 0.95:
                continue
            for f in candidate_filters(t, feat):
                g = cur[mask(cur, f)]
                if len(g) < min_frac * len(is_t) or len(g) < 200:
                    continue
                if g.R.mean() > best_r:
                    best, best_r = f, g.R.mean()
        if best is None:
            break
        chosen.append(best)
        cur = cur[mask(cur, best)]
    return chosen


def apply_filters(t, fs):
    m = pd.Series(True, index=t.index)
    for f in fs:
        m &= mask(t, f)
    return t[m]


def fs_label(fs):
    return " AND ".join(f"{a} {op} {q:.4g}" for a, op, q in fs) if fs else "none (MA only)"


def equity(t, start=100.0):
    t = t.sort_values("t_out")
    eq = start * np.cumprod(1 + np.maximum(t.R.values, -5) * RISK_PER_TRADE / 100)   # compounding, 1% risk a trade
    peak = np.maximum.accumulate(np.r_[start, eq])[1:]
    dd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    daily = pd.Series(t.R.values * RISK_PER_TRADE, index=pd.to_datetime(t.t_out.values)).resample("1D").sum()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0
    return t.t_out.values, eq, dd, sharpe


def summarize(t):
    s = stats(t.R, t.net_pct)
    _, eq, dd, sh = equity(t)
    s.update({"total_return_pct": round(float(eq[-1] - 100), 1) if len(eq) else 0.0,
              "max_dd_pct": round(dd, 1), "sharpe": round(sh, 2),
              "avg_R_zero_cost": round(float(t.R0.mean()), 3) if len(t) else 0.0})
    return s


def load_crypto():
    t = pd.read_parquet(f"{HERE}/cache/crypto_trades.parquet")
    t["t_in"] = pd.to_datetime(t.t_in, utc=True)
    t["t_out"] = pd.to_datetime(t.t_out, utc=True)
    t = apply_costs(t, CRYPTO_COST)
    t["period"] = np.where(t.t_in < CRYPTO_SPLIT, "IS", "OOS")
    return t


def load_tsla():
    t = pd.read_parquet(f"{HERE}/cache/tsla_trades.parquet")
    t["t_in"] = pd.to_datetime(t.t_in)
    t["t_out"] = pd.to_datetime(t.t_out)
    t = apply_costs(t, TSLA_COST, TSLA_SHARE)
    t["period"], _ = split_6040(t)
    return t


def main():
    grid, claims, take = [], [], []
    ct = load_crypto()
    tt = load_tsla()
    markets = {}
    for tf in ["5m", "15m"]:
        markets[f"crypto_{tf}"] = ct[ct.tf == tf]
    for tf in ["2m", "5m"]:
        markets[f"tsla_{tf}"] = tt[tt.tf == tf]

    # 1) parameter sweep (MA only), every market, IS / OOS, with and without cost
    for mk, t in markets.items():
        for (smin, tgt, p), g in t.groupby(["smin", "tgt", "period"]):
            s = stats(g.R, g.net_pct)
            grid.append({"params": {"market": mk, "view": "ma_only_sweep", "smin": smin, "tgt": tgt, "period": p},
                         **s, "avg_R_zero_cost": stats(g.R0)["avg_R"]})

    # 2) feature buckets + 3) filter pick on IS, judged on OOS
    report = {}
    for mk, t_all in markets.items():
        t = t_all[(t_all.smin == BASE[0]) & (t_all.tgt == BASE[1])].copy()
        feats = [f for f in FEATS if f in t.columns and t[f].notna().any()]
        for f in feats:
            tf_ = t
            if f in SHORT_FEATS:      # own 60/40 split on the short window
                tf_ = t.dropna(subset=[f]).copy()
                tf_["period"], _ = split_6040(tf_)
            rows, _ = bucket_table(tf_, f, mk)
            grid += rows
        long_feats = [f for f in feats if f not in SHORT_FEATS and f != "clv"]
        fs = pick_filters(t, long_feats)                      # his rules + best 1-2 Moon Dev data filters
        fs_tail = pick_filters(t, ["clv"], max_n=1)           # his rules + his own bottoming-tail words
        fs_both = fs_tail + pick_filters(apply_filters(t, fs_tail).assign(period=lambda x: x.period), long_feats, max_n=1, min_frac=0.3 * 0.6)
        versions = {"ma_only": [], "ma_plus_data": fs, "ma_plus_tail": fs_tail, "ma_tail_plus_data": fs_both}
        base_is, base_oos = t[t.period == "IS"], t[t.period == "OOS"]
        report[mk] = {"versions": {k: fs_label(v) for k, v in versions.items()}}
        for name, vf in versions.items():
            for p, g in [("IS", base_is), ("OOS", base_oos)]:
                g = apply_filters(g, vf)
                report[mk][f"{name}_{p}"] = summarize(g)
                grid.append({"params": {"market": mk, "view": "better_version", "version": name,
                                        "filters": fs_label(vf), "period": p},
                             **stats(g.R, g.net_pct), "avg_R_zero_cost": stats(g.R0)["avg_R"]})
        report[mk]["_filters"] = versions
        # short-data filters (funding, liq): own 60/40 split, compared to MA only on the SAME window
        for f in [x for x in feats if x in SHORT_FEATS]:
            d = t.dropna(subset=[f]).copy()
            d["period"], cut = split_6040(d)
            fs2 = pick_filters(d, [f], min_frac=0.3, max_n=1)
            o = d[d.period == "OOS"]
            report[mk][f"short_{f}"] = {"filter": fs_label(fs2), "split": str(cut)[:10],
                                        "ma_only_OOS": stats(o.R), "filtered_OOS": stats(apply_filters(o, fs2).R),
                                        "filtered_IS": stats(apply_filters(d[d.period == "IS"], fs2).R),
                                        "ma_only_IS": stats(d[d.period == "IS"].R)}
            for name, g in [("ma_only", o), ("ma_plus_" + f, apply_filters(o, fs2))]:
                grid.append({"params": {"market": mk, "view": "short_data_filter", "feature": f, "version": name,
                                        "filters": fs_label(fs2), "period": "OOS"}, **stats(g.R, g.net_pct)})
        print(f"🌙 Moon Dev {mk}: filters {fs_label(fs)}")
        for k, v in report[mk].items():
            print("   ", k, v)
    json.dump(report, open(f"{HERE}/cache/report.json", "w"), indent=1, default=str)
    return markets, report, grid, ct, tt


if __name__ == "__main__":
    main()
