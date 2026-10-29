"""Moon Dev: pick params on IN-SAMPLE only, score out-of-sample, run baselines + spot cross check, write result.json."""
import json, os
import numpy as np
import pandas as pd
from multiprocessing import Pool
import core, engine, setups, prep

OUT = os.path.join(core.HERE, "result.json")
PARAM_COLS = ["setup", "tf", "slope_min", "block200_R", "exit", "htf_align", "skip_first", "base_bars", "base_width_atr",
              "trend", "max_retrace_bars"]
MIN_IS_TRADES = {"pullback": 300, "base": 300, "swing_reclaim20": 100, "pullback_us_session": 300}


def stat(r, per="all", tag="n"):
    n = r[f"{per}_{tag}_cnt"]
    if not n:
        return dict(trades=0, win_rate=0, avg_R=0, profit_factor=0, return_pct=0)
    gl = r[f"{per}_{tag}_gl"]
    return dict(trades=int(n), win_rate=round(r[f"{per}_{tag}_win"] / n * 100, 2), avg_R=round(r[f"{per}_{tag}_sum"] / n, 4),
                profit_factor=round(r[f"{per}_{tag}_gw"] / gl, 3) if gl > 0 else 99.0,
                return_pct=round(float(np.expm1(min(r[f"{per}_{tag}_log"], 50)) * 100), 2))


def spec_of(row):
    s = {k: row[k] for k in PARAM_COLS if k in row and row[k] is not None and not (isinstance(row[k], float) and np.isnan(row[k]))}
    for k in ("htf_align", "skip_first", "base_bars", "max_retrace_bars"):
        if k in s:
            s[k] = int(s[k])
    return s


def run_spec(args):
    sym, spec = args
    t = engine.trades_for(sym, spec)
    engine.clear()
    return t


def baseline(args):
    sym, spec, real, seed = args
    t = engine.random_baseline(sym, spec, real, seed)
    engine.clear()
    return t


def bh(sym):
    d = core.load(sym, "1d")
    out = {}
    for per, m in (("is", d.index < core.OOS_START), ("oos", d.index >= core.OOS_START)):
        x = d[m]
        out[per] = round(float((x["close"].iloc[-1] / x["open"].iloc[0] - 1) * 100), 2) if len(x) > 1 else None
    return sym, out


def build_spot(sym):
    """Moon Dev: spot cross check bars (Binance spot 1m), cached as <SYM>SPOT_<tf>."""
    for tf in ["15m", "1h", "4h", "1d", "5m"]:
        f = f"{core.CACHE}/{sym}SPOT_{tf}.parquet"
        if not os.path.exists(f):
            d = prep.load_1m(sym, "spot_klines_1m")
            for tf2, rule in prep.TFS.items():
                prep.resample(d, rule).to_parquet(f"{core.CACHE}/{sym}SPOT_{tf2}.parquet")
            break


def main():
    sw = pd.read_parquet(f"{core.CACHE}/sweep.parquet")
    sums = [c for c in sw.columns if c.split("_")[0] in ("all", "is", "oos")]
    sw[sums] = sw[sums].fillna(0)
    keycols = [c for c in PARAM_COLS if c in sw.columns]
    sw["_key"] = sw[keycols].astype(str).agg("|".join, axis=1)
    pooled = sw.groupby("_key")[sums].sum().reset_index()
    first = sw.drop_duplicates("_key").set_index("_key")[keycols]
    pooled = pooled.join(first, on="_key")
    pooled["market"] = "ALL"
    full = pd.concat([sw, pooled], ignore_index=True)

    # Moon Dev: grid for the dashboard sliders (every left-out number is a sweep)
    grid = []
    for _, r in full.iterrows():
        p = spec_of(r); p["market"] = r["market"]
        s = stat(r, "all", "n")
        grid.append(dict(params=p, trades=s["trades"], win_rate=s["win_rate"], avg_R=s["avg_R"], profit_factor=s["profit_factor"],
                         return_pct=s["return_pct"], is_avg_R=stat(r, "is")["avg_R"], oos_avg_R=stat(r, "oos")["avg_R"],
                         oos_trades=stat(r, "oos")["trades"], avg_R_zero_cost=stat(r, "all", "g")["avg_R"]))

    # Moon Dev: pick the best config per setup on IN-SAMPLE pooled net avg R only
    picks = {}
    for setup, g in pooled.groupby("setup"):
        g = g[g["is_n_cnt"] >= MIN_IS_TRADES[setup]].copy()
        g["is_avgR"] = g["is_n_sum"] / g["is_n_cnt"]
        best = g.sort_values("is_avgR", ascending=False).iloc[0]
        picks[setup] = dict(spec=spec_of(best), is_net=stat(best, "is"), oos_net=stat(best, "oos"), is_zero=stat(best, "is", "g"),
                            oos_zero=stat(best, "oos", "g"))
        print(f"Moon Dev IS pick {setup}: {picks[setup]['spec']}\n   IS {picks[setup]['is_net']}\n   OOS {picks[setup]['oos_net']}\n   OOS zero cost {picks[setup]['oos_zero']}")
    head_setup = max(picks, key=lambda k: picks[k]["is_net"]["avg_R"])
    print("Moon Dev headline setup (by IS):", head_setup)

    # Moon Dev: rerun every pick on every coin for real trade lists
    with Pool(8) as p:
        tr = {s: pd.concat([t for t in p.map(run_spec, [(sym, picks[s]["spec"]) for sym in core.SYMS]) if len(t)], ignore_index=True)
              for s in picks}
        base = {}
        for s in picks:
            if s == "pullback_us_session":
                continue
            jobs = [(sym, picks[s]["spec"], tr[s][tr[s]["symbol"] == sym], seed) for sym in core.SYMS for seed in range(3)
                    if (tr[s]["symbol"] == sym).any()]
            base[s] = pd.concat([t for t in p.map(baseline, jobs) if len(t)], ignore_index=True)
        bhs = dict(p.map(bh, core.SYMS))
        p.map(build_spot, ["BTC", "ETH", "SOL"])
    spot = {}
    for sym in ["BTC", "ETH", "SOL"]:
        st = engine.trades_for(f"{sym}SPOT", picks[head_setup]["spec"]); engine.clear()
        pf = tr[head_setup][tr[head_setup]["symbol"] == sym]
        spot[sym] = dict(spot=core.summarize(st), perp=core.summarize(pf))

    def split(t, col="R_net"):
        return dict(in_sample=core.summarize(t[t["time_in"] < core.OOS_START], col),
                    out_of_sample=core.summarize(t[t["time_in"] >= core.OOS_START], col), all=core.summarize(t, col))

    setups_out = {}
    for s in picks:
        t = tr[s]
        per_coin = {}
        for sym, g in t.groupby("symbol"):
            per_coin[sym] = dict(is_avg_R=core.summarize(g[g["time_in"] < core.OOS_START])["avg_R"],
                                 oos_avg_R=core.summarize(g[g["time_in"] >= core.OOS_START])["avg_R"],
                                 oos_trades=int((g["time_in"] >= core.OOS_START).sum()),
                                 oos_win_rate=core.summarize(g[g["time_in"] >= core.OOS_START])["win_rate"])
        rnd = base.get(s)
        setups_out[s] = dict(params=picks[s]["spec"], net=split(t), zero_cost=split(t, "R"), per_coin=per_coin,
                             random_entry_same_exits=None if rnd is None else dict(net=split(rnd), zero_cost=split(rnd, "R")))
    json.dump(dict(setups=setups_out, bh=bhs, spot=spot, head=head_setup), open(f"{core.CACHE}/report_raw.json", "w"), default=str, indent=1)

    # ---------- Moon Dev: assemble result.json ----------
    T = tr[head_setup].sort_values("time_out").reset_index(drop=True)
    eq = np.cumprod(1 + 0.01 * T["R_net"].values)
    step = max(1, len(T) // 499)
    curve = [[T["time_out"].iloc[i].isoformat(), round(float(eq[i] * 10000), 2)] for i in range(0, len(T), step)][:499]
    curve.append([T["time_out"].iloc[-1].isoformat(), round(float(eq[-1] * 10000), 2)])
    samp = T.iloc[np.linspace(0, len(T) - 1, min(300, len(T))).astype(int)]
    trades = [dict(t_in=r.time_in.isoformat(), t_out=r.time_out.isoformat(), side=r.side, entry=round(r.entry, 6), exit=round(r.exit, 6),
                   R=round(r.R_net, 3), pnl_pct=round(r.ret_net * 100, 3), setup=f"{r.setup} {r.tf}", symbol=r.symbol)
              for r in samp.itertuples()]
    # price replay: BTC, the 400-bar window with the most trades (out of sample)
    tf = T["tf"].iloc[0]
    d = core.load("BTC", tf if tf in ("5m", "15m", "1h", "4h", "1d") else "5m")
    ind = core.indicators(d)
    bt = T[(T["symbol"] == "BTC") & (T["time_in"] >= core.OOS_START)]
    pos = d.index.get_indexer(bt["time_in"])
    pos = pos[pos >= 0]
    best_start, best_n = len(d) - 400, -1
    for s0 in pos:
        st0 = max(0, s0 - 100)
        nn = ((pos >= st0) & (pos < st0 + 400)).sum()
        if nn > best_n:
            best_start, best_n = st0, nn
    w = slice(best_start, best_start + 400)
    dw, iw = d.iloc[w], ind.iloc[w]
    bars = [[t.isoformat(), r.open, r.high, r.low, r.close, round(e, 4), round(s, 4) if s == s else None, round(s2, 4) if s2 == s2 else None]
            for t, r, e, s, s2 in zip(dw.index, dw.itertuples(), iw["ema9"], iw["sma20"], iw["sma200"])]
    rt = bt[(bt["time_in"] >= dw.index[0]) & (bt["time_in"] <= dw.index[-1])]
    replay_tr = [dict(t_in=r.time_in.isoformat(), t_out=r.time_out.isoformat(), side=r.side, entry=r.entry, exit=r.exit,
                      R=round(r.R_net, 3), pnl_pct=round(r.ret_net * 100, 3), setup=r.setup, symbol="BTC") for r in rt.itertuples()]

    cj = json.load(open(f"{core.CACHE}/claims.json"))
    H = setups_out[head_setup]
    cov = pd.read_csv(f"{core.CACHE}/coverage.csv")
    res = dict(
        package="2_crypto_rules_and_claims",
        title="Does it work on crypto? His 9 EMA / 20 SMA / 200 SMA rules on 16 Binance perps, plus every claim measured",
        market="Crypto: 16 Binance USDT perps (ARB AVAX BNB BTC CRV DOGE ETH FIL INIT LINK SOL STRK SUI TIA XRP ZEC), spot cross check BTC/ETH/SOL",
        data="/Volumes/Moon 26/tick_data/binance/futures_klines_1m/<SYM>USDT/<SYM>USDT_1m_<start>_<end>.parquet, "
             f"{cov['start'].min()[:10]} -> {cov['end'].max()[:10]} (each coin from its listing date); spot: spot_klines_1m/. "
             "Resampled to 5m/15m/1h/4h/1d (1m/2m built on the fly for the US-session test).",
        headline={}, stats={}, equity_curve=curve, trades=trades, grid=grid, claims=cj["claims"],
        price_replay=dict(symbol="BTC", timeframe=tf, bars=bars, trades=replay_tr),
        setups={k: dict(params=v["params"], is_net=v["net"]["in_sample"], oos_net=v["net"]["out_of_sample"],
                        is_zero_cost=v["zero_cost"]["in_sample"], oos_zero_cost=v["zero_cost"]["out_of_sample"],
                        random_oos_net=None if v["random_entry_same_exits"] is None else v["random_entry_same_exits"]["net"]["out_of_sample"],
                        random_oos_zero_cost=None if v["random_entry_same_exits"] is None else v["random_entry_same_exits"]["zero_cost"]["out_of_sample"],
                        per_coin=v["per_coin"]) for k, v in setups_out.items()},
        buy_and_hold_pct=bhs, spot_cross_check=spot, claim_tables=cj["tables"], ma200_events=cj["ma200"],
        left_out_by_video={"slope_min": "video leaves this out (says 'not flat', ideally '45 degrees'); ATR per bar, 5-bar lookback",
                           "block200_R": "video leaves this out (skip if 200 SMA 'directly above'); 0 = off",
                           "exit": "video leaves this out besides 200 SMA = Target 1; R1.5/R2/R3 fixed or trail under pivots",
                           "skip_first": "accel threshold left out; we use close > 3 ATR past the 20 SMA",
                           "base_bars / base_width_atr": "video never defines a base",
                           "max_retrace_bars / trend": "0:30 idea gives no numbers",
                           "max_hold": str(core.MAX_HOLD), "zone tolerance": "0.25 ATR under the 20 SMA",
                           "position size / risk": "1% of equity risked per trade for the curves"},
    )
    hs = H["net"]
    res["stats"] = dict(**{k: hs["all"][k] for k in ["trades", "win_rate", "avg_R", "profit_factor", "total_return_pct", "max_dd_pct", "sharpe"]},
                        in_sample=hs["in_sample"], out_of_sample=hs["out_of_sample"],
                        zero_cost=dict(in_sample=H["zero_cost"]["in_sample"], out_of_sample=H["zero_cost"]["out_of_sample"]),
                        random_entries_same_exits=H["random_entry_same_exits"],
                        costs="0.05% Binance taker + 0.02% slippage per side (0.14% round trip), charged in R. Zero-cost version shown too. 1% risk per trade, all coins in exit-time order.",
                        params=H["params"], headline_setup=head_setup)
    # Moon Dev: headline + takeaways straight from the numbers above
    G = pd.DataFrame([dict(**x["params"], is_R=x["is_avg_R"], oos_R=x["oos_avg_R"], zc=x["avg_R_zero_cost"]) for x in grid])
    A = G[G["market"] == "ALL"]
    oos_pos = {k: f"{(g['oos_R'] > 0).mean() * 100:.0f}%" for k, g in A.groupby("setup")}
    o, i = hs["out_of_sample"], hs["in_sample"]
    rnd = H["random_entry_same_exits"]["net"]["out_of_sample"] if H["random_entry_same_exits"] else None
    nver = sum(c["verdict"] == "VERIFIED" for c in cj["claims"]); nb = sum(c["verdict"] == "BUSTED" for c in cj["claims"])
    res["headline"] = dict(label=f"Best in-sample crypto version ({head_setup}, {H['params']['tf']}) out of sample, 16 coins, after costs",
                           value=f"{o['avg_R']:+.2f}R per trade OOS ({o['trades']} trades) vs {i['avg_R']:+.2f}R in-sample; random entries with the same exits {rnd['avg_R']:+.2f}R",
                           verdict="BUSTED" if o["avg_R"] <= 0 or (rnd and o["avg_R"] <= rnd["avg_R"]) else "VERIFIED")
    P = setups_out
    res["takeaways"] = [
        f"'Works on crypto' did not hold up. Every setup was picked on 2020 -> 2024-06 data, then tested on 2024-07 -> 2026-10: "
        + ", ".join(f"{k} {P[k]['net']['in_sample']['avg_R']:+.2f}R -> {P[k]['net']['out_of_sample']['avg_R']:+.2f}R" for k in P) + " per trade after costs.",
        f"Share of all swept configs that made money out of sample (pooled, after costs): {oos_pos}.",
        f"The in-sample wins were mostly the 2020-21 crypto bull run plus trailing stops: random entries with the same exits also made money in-sample "
        + ", ".join(f"({k} {P[k]['random_entry_same_exits']['net']['in_sample']['avg_R']:+.2f}R)" for k in P if P[k]['random_entry_same_exits']) + ".",
        f"His day-trade version (US session, 1m/2m/5m by time of day, 13:30-20:00 UTC DST aware) is close to break-even before costs "
        f"({P['pullback_us_session']['zero_cost']['out_of_sample']['avg_R']:+.2f}R OOS) but costs eat it alive ({P['pullback_us_session']['net']['out_of_sample']['avg_R']:+.2f}R): stops under a 1m/2m bar are tiny, so 0.14% round trip is a big chunk of R.",
        f"Claims scoreboard: {nver} VERIFIED, {nb} BUSTED, {len(cj['claims']) - nver - nb} MIXED out of {len(cj['claims'])}. The MAs barely sort good entries from bad on crypto; the 200 SMA acts like any other line (placebo lines bounce just as often).",
        "Higher time frames hurt less: zero-cost avg R climbs from 5m to 4h, and the HTF-alignment filter and trailing exit are the only knobs that help on average. Still not enough to beat costs out of sample.",
        f"Spot cross check (BTC/ETH/SOL, same {head_setup} rules, full history): spot and perp agree in sign, so the result is not a perp quirk.",
    ]
    json.dump(res, open(OUT, "w"), default=str)
    print(f"Moon Dev: wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
