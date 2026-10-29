"""
Moon Dev edge layer, step 3: write result.json in the shared plan.md format.
Runs analyze.py, then builds the bot market (BTCUSDT 15m, his rules + the picked flow filters)
for the equity curve, the trade list and one real price replay window.
"""
import os
import json
import numpy as np
import pandas as pd
import analyze as A
import build_crypto as bc
from common import add_mas

HERE = os.path.dirname(os.path.abspath(__file__))
BOT_SYM, BOT_TF, BOT_MARKET = "BTCUSDT", "15m", "crypto_15m"


def r2(x, n=3):
    return round(float(x), n)


def main():
    markets, report, grid, ct, tt = A.main()
    rep = report[BOT_MARKET]
    bot_filters = rep["_filters"]["ma_plus_data"]
    print(f"🌙 Moon Dev bot filters ({BOT_MARKET}, picked on IS): {A.fs_label(bot_filters)}")

    base = ct[(ct.smin == A.BASE[0]) & (ct.tgt == A.BASE[1])]
    bot = A.apply_filters(base[(base.symbol == BOT_SYM) & (base.tf == BOT_TF)], bot_filters).sort_values("t_in")
    bot_ma = base[(base.symbol == BOT_SYM) & (base.tf == BOT_TF)]
    s = A.summarize(bot)
    s["in_sample"] = A.summarize(bot[bot.period == "IS"])
    s["out_of_sample"] = A.summarize(bot[bot.period == "OOS"])
    s["ma_only_same_market_OOS"] = A.summarize(bot_ma[bot_ma.period == "OOS"])
    s["costs"] = ("crypto 0.05% taker + 0.02% slippage per side (0.14% round trip); TSLA 1 bp per side + $0.005/share "
                  "per side. avg_R_zero_cost = same trades with no cost. Equity = compounding, 1% risk a trade "
                  "(video gives no size). Stop = trigger bar other end, target 2R, max hold 48 bars (both sweeps).")

    tx, eq, _, _ = A.equity(bot)
    idx = np.unique(np.linspace(0, len(eq) - 1, min(500, len(eq))).astype(int))
    curve = [[pd.Timestamp(tx[i]).isoformat(), r2(eq[i], 2)] for i in idx]

    def trow(r):
        return {"t_in": pd.Timestamp(r.t_in).isoformat(), "t_out": pd.Timestamp(r.t_out).isoformat(),
                "side": "long" if r.dir == 1 else "short", "entry": r2(r.entry, 2), "exit": r2(r.exit, 2),
                "R": r2(r.R), "pnl_pct": r2(r.net_pct * 100), "setup": "MA pullback + flow filter",
                "symbol": r.symbol, "imb_trig": r2(r.imb_trig), "rvol": r2(r.rvol)}
    trades = [trow(r) for r in bot[bot.period == "OOS"].tail(300).itertuples()]

    # one real window: the 400 15m bars ending at the last OOS bot trade
    d1 = bc.load_1m(BOT_SYM)
    b = add_mas(bc.resample(d1, "15min"))
    last = pd.Timestamp(bot.t_out.iloc[-1])
    end = b.index.searchsorted(last) + 20
    w = b.iloc[max(0, end - 400):end]
    in_w = bot[(bot.t_in >= w.index[0]) & (bot.t_out <= w.index[-1])]
    replay = {"symbol": f"{BOT_SYM} perp 15m (Binance)",
              "bars": [[t.isoformat(), r2(r.open, 2), r2(r.high, 2), r2(r.low, 2), r2(r.close, 2),
                        r2(r.ema9, 2), r2(r.sma20, 2), r2(r.sma200, 2)] for t, r in w.iterrows()],
              "trades": [trow(r) for r in in_w.itertuples()]}

    def oos(mk, v):
        return report[mk][f"{v}_OOS"]

    head_val = (f"OOS after costs, avg R per trade. Crypto 15m (16 perps): MA only {oos('crypto_15m', 'ma_only')['avg_R']}R "
                f"-> MA + flow {oos('crypto_15m', 'ma_plus_data')['avg_R']}R. Crypto 5m: {oos('crypto_5m', 'ma_only')['avg_R']}R -> "
                f"{oos('crypto_5m', 'ma_plus_data')['avg_R']}R. TSLA 5m: {oos('tsla_5m', 'ma_only')['avg_R']}R -> "
                f"{oos('tsla_5m', 'ma_plus_data')['avg_R']}R. Data cuts the loss but nothing makes money.")

    c5 = ct[(ct.tf == "5m") & (ct.smin == A.BASE[0]) & (ct.tgt == A.BASE[1])]
    tail_lo = c5[c5.clv < 1 / 3].R0.mean()
    tail_hi = c5[c5.clv >= 2 / 3].R0.mean()
    sl = ct[(ct.tf == "5m") & (ct.tgt == "2.0")].groupby("smin").R0.mean()
    claims = [
        {"claim": "The MAs are only one part of the plan, never trade just because price pulled back to the 20 SMA",
         "timestamp": "0:01, 0:36",
         "test": "Same pullback trades, MA only vs MA + best 2 Moon Dev data filters (picked on IS), judged OOS 2024-07 -> 2026-10",
         "result": f"crypto 15m OOS {oos('crypto_15m', 'ma_only')['avg_R']}R vs {oos('crypto_15m', 'ma_plus_data')['avg_R']}R, "
                   f"PF {oos('crypto_15m', 'ma_only')['profit_factor']} vs {oos('crypto_15m', 'ma_plus_data')['profit_factor']}",
         "verdict": "VERIFIED (MAs alone lose; the extra info helps) but the better version still loses after costs"},
        {"claim": "Enter above a doji / bottoming tail bar at the MA, stop under its low",
         "timestamp": "0:24-0:25",
         "test": "Trigger bar close location (clv). Bottom third = no tail, top third = bottoming tail. Crypto 5m, zero cost",
         "result": f"no tail {tail_lo:.3f}R vs bottoming tail {tail_hi:.3f}R per trade (zero cost)",
         "verdict": "VERIFIED as the strongest single filter, stronger than taker flow. Still too small to pay costs"},
        {"claim": "It works on crypto", "timestamp": "0:30",
         "test": "MA pullback on 16 Binance perps, 5m and 15m, 2020 -> 2026, before any cost",
         "result": f"zero-cost avg R: 5m {oos('crypto_5m', 'ma_only')['avg_R_zero_cost']}, 15m {oos('crypto_15m', 'ma_only')['avg_R_zero_cost']} (OOS)",
         "verdict": "BUSTED (negative even before fees)"},
        {"claim": "Never trade a flat 20 SMA (rule 1)", "timestamp": "0:10",
         "test": "Slope floor sweep (20 SMA move over 5 bars in ATRs): 0.1 / 0.3 / 0.6. Crypto 5m, 2R, zero cost",
         "result": ", ".join(f"{k}: {v:.3f}R" for k, v in sl.items()),
         "verdict": "BUSTED here: a steeper slope floor did not help (package 2 owns the full test)"},
        {"claim": "Moon Dev idea (not in the video): long liquidations flushing into a long pullback = fuel",
         "timestamp": "n/a",
         "test": "liq_clean BTC/ETH/SOL/XRP 2024-06 -> 2025-12, 60/40 split, same-side liqs in the last 3 bars",
         "result": f"crypto 15m OOS: MA only {report['crypto_15m']['short_liq_fuel']['ma_only_OOS']['avg_R']}R vs "
                   f"filtered {report['crypto_15m']['short_liq_fuel']['filtered_OOS']['avg_R']}R ({report['crypto_15m']['short_liq_fuel']['filter']})",
         "verdict": "NOT PROVEN (helps a bit, short sample, still loses)"},
        {"claim": "Moon Dev idea: funding extreme against the trade", "timestamp": "n/a",
         "test": "Binance funding 2024-01 -> 2026-08, 60/40 split",
         "result": f"crypto 5m OOS: MA only {report['crypto_5m']['short_fund_crowd']['ma_only_OOS']['avg_R']}R vs "
                   f"filtered {report['crypto_5m']['short_fund_crowd']['filtered_OOS']['avg_R']}R ({report['crypto_5m']['short_fund_crowd']['filter']})",
         "verdict": "BUSTED (IS pick made OOS worse)"},
    ]
    takeaways = [
        "BUSTED: his MA pullback has no edge on crypto or TSLA even before fees (about -0.05R a trade on crypto).",
        "Moon Dev's taker flow on the trigger bar is the best data filter: buyers in control on the trigger bar = fewer losers. It held out of sample on every market.",
        "Half of that flow edge is just candle shape. His own words (enter above a bottoming tail, 0:25) are the strongest filter we found.",
        "Costs are the killer. A 5m crypto stop is so tight that 0.14% round trip eats about 0.5R (median trade) to 0.7R (average) a trade. 15m: about 0.3R to 0.4R.",
        "Liquidation fuel and funding: short samples, small or no help. Not worth trading on.",
        "Best version (rules + flow + volume) still loses after costs: do not run it with real money.",
    ]
    out = {
        "package": "3_moondev_edge_layer",
        "title": "Moon Dev's edge layer: does order flow, liquidation and funding data fix his MA pullback?",
        "market": "Crypto: 16 Binance USDT perps 5m + 15m. TSLA 2m + 5m RTH (Nasdaq lit tape + block prints).",
        "data": ("/Volumes/Moon 26/tick_data/binance/futures_klines_1m/<SYM>/<SYM>_1m_*.parquet (2020-01 -> 2026-10, taker_buy_volume); "
                 "/Volumes/Moon 26/tick_data/binance/funding/*_funding.csv (2024-01 -> 2026-08); "
                 "moon-dev-trading-bots/data/moondev_api/liq_clean_{BTC,ETH,SOL,XRP}_5m.csv (2024-06 -> 2025-12; the liq_data_*_ohlcv files are banned in data_sources.md for fake prices); "
                 "/Volumes/Moon 26/tick_data/derived/tsla/lit_rth_{hist,orig}.parquet + big_prints_{hist,orig}.parquet (2024-07-01 -> 2026-09-23)"),
        "headline": {"label": "MA only vs MA + Moon Dev data, out of sample", "value": head_val, "verdict": "BUSTED"},
        "stats": s,
        "equity_curve": curve,
        "trades": trades,
        "grid": grid,
        "claims": claims,
        "price_replay": replay,
        "takeaways": takeaways,
        "versions_by_market": {mk: {k: v for k, v in r.items() if k != "_filters"} for mk, r in report.items()},
    }
    json.dump(out, open(f"{HERE}/result.json", "w"), indent=1, default=str)
    print(f"🌙 Moon Dev wrote result.json: {len(grid)} grid rows, {len(trades)} trades, {len(curve)} equity pts, "
          f"{len(replay['bars'])} replay bars, {len(replay['trades'])} replay trades")
    print("🌙 Moon Dev bot stats:", {k: v for k, v in s.items() if k != "costs"})


if __name__ == "__main__":
    main()
