"""
Moon Dev edge layer, step 1 (TSLA).
Builds RTH 1m bars from Moon Dev's Nasdaq lit tape (2024-07 -> 2026-09) with tick-rule flow,
adds block prints (>= 1,000 shares, direction = printed above / below the last lit price),
then runs his MA pullback on 2m and 5m and scores each trade with the flow at the trigger bar.
Package 1 owns the full TSLA system. This file only needs the signal.
"""
import os
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit
from common import add_mas, run_signals

TS = "/Volumes/Moon 26/tick_data/derived/tsla"
HERE = os.path.dirname(os.path.abspath(__file__))
BARS = f"{HERE}/cache/tsla_1m.parquet"
GRID = [(s, t) for s in (0.1, 0.3, 0.6) for t in (1.0, 1.5, 2.0, 3.0, "200")]


@njit(cache=True)
def tick_sign(p, last_p, last_s):
    """Moon Dev tick rule: up-tick = buy (+1), down-tick = sell (-1), same price = keep last sign."""
    s = np.empty(len(p), dtype=np.int8)
    for k in range(len(p)):
        if p[k] > last_p:
            last_s = 1
        elif p[k] < last_p:
            last_s = -1
        s[k] = last_s
        last_p = p[k]
    return s, last_p, last_s


def build_1m():
    parts = []
    last_p, last_s = 0.0, 0
    for name in ["lit_rth_hist.parquet", "lit_rth_orig.parquet"]:
        f = pq.ParquetFile(f"{TS}/{name}")
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=["ns", "price"]).to_pandas()
            s, last_p, last_s = tick_sign(t.price.values, last_p, last_s)
            t["s"] = s
            t["m"] = t.ns.values // 60_000_000_000
            a = t.groupby("m").agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
                                   close=("price", "last"), n=("price", "size"), sgn=("s", "sum"))
            parts.append(a)
        print(f"🌙 Moon Dev read {name}", flush=True)
    b = pd.concat(parts)
    b = b.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last", "n": "sum", "sgn": "sum"})
    big = pd.concat([pd.read_parquet(f"{TS}/{x}", columns=["ns", "size", "dev_bp"])
                     for x in ["big_prints_hist.parquet", "big_prints_orig.parquet"]])
    big["m"] = big.ns.values // 60_000_000_000
    big["ss"] = np.sign(big.dev_bp.fillna(0)) * big["size"]
    big["ss5k"] = np.where(big["size"] >= 5000, big.ss, 0)
    bg = big.groupby("m")[["ss", "ss5k"]].sum()
    b = b.join(bg).fillna({"ss": 0, "ss5k": 0})
    b.index = pd.to_datetime(b.index * 60_000_000_000, utc=True).tz_convert("America/New_York")
    b.index.name = "minute"
    b.to_parquet(BARS)
    return b


def to_tf(b1, minutes):
    b1 = b1.copy()
    b1["day"] = b1.index.date
    g = b1.groupby(["day", b1.index.floor(f"{minutes}min")])
    b = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
              n=("n", "sum"), sgn=("sgn", "sum"), ss=("ss", "sum"), ss5k=("ss5k", "sum"))
    b.index = b.index.get_level_values(1)
    b["day_id"] = pd.factorize(b.index.date)[0]
    return b


def main():
    t0 = time.time()
    b1 = pd.read_parquet(BARS) if os.path.exists(BARS) else build_1m()
    print(f"🌙 Moon Dev TSLA 1m bars: {len(b1):,} {b1.index[0]} -> {b1.index[-1]} ({time.time() - t0:.0f}s)")
    rows = []
    for tf, mins in {"2m": 2, "5m": 5}.items():
        b = add_mas(to_tf(b1, mins))
        imb = b.sgn / b.n
        f = pd.DataFrame(index=b.index)
        f["imb_trig"] = imb
        f["imb_pb3"] = b.sgn.shift(1).rolling(3).sum() / b.n.shift(1).rolling(3).sum()
        f["imb6"] = b.sgn.rolling(6).sum() / b.n.rolling(6).sum()
        f["imb12"] = b.sgn.rolling(12).sum() / b.n.rolling(12).sum()
        f["ret6"] = b.close / b.close.shift(6) - 1
        f["rvol"] = b.n / b.n.shift(1).rolling(20).mean()
        f["blk3"] = b.ss.rolling(3).sum()
        f["blk5k3"] = b.ss5k.rolling(3).sum()
        for smin, tgt in GRID:
            t = run_signals(b, smin=smin, tgt_r=3.0 if tgt == "200" else tgt, use200tgt=(tgt == "200"),
                            skip200=True, max_hold=10_000, flat_eod=True)
            i = t.i_trig.values
            dr = t.dir.values
            for col in ["imb_trig", "imb_pb3", "imb6", "imb12", "blk3", "blk5k3"]:
                t[col] = f[col].values[i] * dr
            t["rvol"] = f.rvol.values[i]
            t["cvd_div"] = ((f.ret6.values[i] * dr < 0) & (f.imb6.values[i] * dr > 0)).astype(int)
            rng = (b.high.values[i] - b.low.values[i])
            t["clv"] = np.where(rng > 0, np.where(dr == 1, b.close.values[i] - b.low.values[i], b.high.values[i] - b.close.values[i]) / np.where(rng > 0, rng, 1), 0.5)   # candle close location (price action)
            t["ext"] = (b.close.values[i] - b.sma20.values[i]) / b.atr.values[i] * dr
            t["slope_v"] = b.slope.values[i] * dr
            t["symbol"], t["tf"], t["smin"], t["tgt"] = "TSLA", tf, smin, str(tgt)
            rows.append(t.drop(columns=["i_trig", "i_in", "i_out"]))
        print(f"🌙 Moon Dev TSLA {tf}: {len(b):,} bars, {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(f"{HERE}/cache/tsla_trades.parquet", index=False)
    print(f"🌙 Moon Dev saved {len(out):,} TSLA trade rows")


if __name__ == "__main__":
    main()
