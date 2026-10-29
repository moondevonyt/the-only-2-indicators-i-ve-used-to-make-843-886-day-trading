"""
Moon Dev edge layer, step 1 (crypto).
Builds 5m and 15m bars from Binance perp 1m klines, finds his MA pullback trades, and scores
every trade with Moon Dev's order flow / funding / liquidation data AT THE TRIGGER BAR CLOSE
(before entry, so no look-ahead). Saves one trades table to cache/crypto_trades.parquet.
Real data only. A missing file = crash, on purpose.
"""
import os
import time
import numpy as np
import pandas as pd
from common import add_mas, run_signals

KL = "/Volumes/Moon 26/tick_data/binance/futures_klines_1m"
FUND = "/Volumes/Moon 26/tick_data/binance/funding"
LIQ = "/Users/md/Dropbox/dev/github/moon-dev-trading-bots/data/moondev_api"
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT", "AVAXUSDT",
        "ZECUSDT", "CRVUSDT", "FILUSDT", "ARBUSDT", "SUIUSDT", "TIAUSDT", "STRKUSDT", "INITUSDT"]
FUND_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
LIQ_SYMS = ["BTC", "ETH", "SOL", "XRP"]          # liq_clean only (data_sources.md bans liq_data_*_ohlcv)
TFS = {"5m": "5min", "15m": "15min"}
# grid: slope floor for "not flat" (video: no number), target (video: 200 SMA or nothing)
GRID = [(s, t) for s in (0.1, 0.3, 0.6) for t in (1.0, 1.5, 2.0, 3.0, "200")]
MAX_HOLD = 48


def load_1m(sym):
    f = [x for x in os.listdir(f"{KL}/{sym}") if x.startswith(f"{sym}_1m_")]
    assert len(f) == 1, f"Moon Dev: expected one all-in-one 1m file for {sym}, found {f}"
    cols = ["open_time", "open", "high", "low", "close", "volume", "trades", "taker_buy_volume"]
    d = pd.read_parquet(f"{KL}/{sym}/{f[0]}", columns=cols).set_index("open_time").sort_index()
    return d


def resample(d, rule):
    b = d.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum", "trades": "sum", "taker_buy_volume": "sum"}).dropna(subset=["close"])
    return b[b.volume > 0]


def flow_features(b):
    """Moon Dev order flow per bar. imb = (taker buy - taker sell) / volume, in [-1, 1]."""
    v, tb = b.volume, b.taker_buy_volume
    delta = 2 * tb - v                                   # per bar CVD change
    f = pd.DataFrame(index=b.index)
    f["imb_trig"] = delta / v
    f["imb_pb3"] = delta.shift(1).rolling(3).sum() / v.shift(1).rolling(3).sum()
    f["imb6"] = delta.rolling(6).sum() / v.rolling(6).sum()
    f["imb12"] = delta.rolling(12).sum() / v.rolling(12).sum()
    f["ret6"] = b.close / b.close.shift(6) - 1
    f["rvol"] = v / v.shift(1).rolling(20).mean()
    ts = v / b.trades.replace(0, np.nan)
    f["tsize"] = ts / ts.shift(1).rolling(20).mean()
    return f


def load_funding(sym):
    fd = pd.read_csv(f"{FUND}/{sym}_funding.csv")
    fd["ts"] = pd.to_datetime(fd.ts, utc=True, format="ISO8601")
    return fd.sort_values("ts")


def load_liq(coin, rule):
    q = pd.read_csv(f"{LIQ}/liq_clean_{coin}_5m.csv", usecols=["datetime", "forced_sell_usd", "forced_buy_usd", "liq_usd", "coinm_suspect"])
    q["datetime"] = pd.to_datetime(q.datetime, utc=True)
    q = q[~q.coinm_suspect].set_index("datetime")      # drop the 61-bar coin-M inflation flag
    q = q[["forced_sell_usd", "forced_buy_usd", "liq_usd"]].resample(rule, label="left", closed="left").sum()
    per_day = int(pd.Timedelta("1D") / pd.Timedelta(rule))
    base = q.liq_usd.shift(1).rolling(30 * per_day, min_periods=7 * per_day).mean() * 3
    out = pd.DataFrame(index=q.index)
    out["long_liq3"] = q.forced_sell_usd.rolling(3).sum() / base     # longs flushed (price down)
    out["short_liq3"] = q.forced_buy_usd.rolling(3).sum() / base     # shorts flushed (price up)
    return out


def main():
    t0 = time.time()
    rows = []
    for sym in SYMS:
        d1 = load_1m(sym)
        for tf, rule in TFS.items():
            b = add_mas(resample(d1, rule))
            ff = flow_features(b)
            tclose = b.index + pd.Timedelta(rule)
            fund = None
            if sym in FUND_SYMS:
                fd = load_funding(sym)
                pos = np.searchsorted(fd.ts.values, tclose.values, side="right") - 1
                fund = np.where(pos >= 0, fd.rate.values[np.maximum(pos, 0)], np.nan)
                fund[tclose.values < fd.ts.values[0]] = np.nan
            liq = None
            coin = sym.replace("USDT", "")
            if coin in LIQ_SYMS:
                liq = load_liq(coin, rule).reindex(b.index)
            for smin, tgt in GRID:
                t = run_signals(b, smin=smin, tgt_r=3.0 if tgt == "200" else tgt,
                                use200tgt=(tgt == "200"), skip200=True, max_hold=MAX_HOLD)
                if len(t) == 0:
                    continue
                i = t.i_trig.values
                dr = t.dir.values
                for col in ["imb_trig", "imb_pb3", "imb6", "imb12", "rvol", "tsize"]:
                    v = ff[col].values[i]
                    t[col] = v * dr if col.startswith("imb") else v
                t["cvd_div"] = ((ff.ret6.values[i] * dr < 0) & (ff.imb6.values[i] * dr > 0)).astype(int)
                rng = (b.high.values[i] - b.low.values[i])
                t["clv"] = np.where(rng > 0, np.where(dr == 1, b.close.values[i] - b.low.values[i], b.high.values[i] - b.close.values[i]) / np.where(rng > 0, rng, 1), 0.5)   # candle close location, trade side (price action, not Moon Dev data)
                t["ext"] = (b.close.values[i] - b.sma20.values[i]) / b.atr.values[i] * dr
                t["slope_v"] = b.slope.values[i] * dr
                t["fund_crowd"] = fund[i] * dr * 1e4 if fund is not None else np.nan    # bps, + = crowd on our side
                if liq is not None:
                    ll, sl = liq.long_liq3.values[i], liq.short_liq3.values[i]
                    t["liq_fuel"] = np.where(dr == 1, ll, sl)      # longs flushed on a long pullback = fuel (0:36 idea)
                    t["liq_against"] = np.where(dr == 1, sl, ll)
                else:
                    t["liq_fuel"] = np.nan
                    t["liq_against"] = np.nan
                t["symbol"] = sym
                t["tf"] = tf
                t["smin"] = smin
                t["tgt"] = str(tgt)
                rows.append(t.drop(columns=["i_trig", "i_in", "i_out"]))
            print(f"🌙 Moon Dev {sym} {tf}: {len(b):,} bars {b.index[0].date()} -> {b.index[-1].date()}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(f"{HERE}/cache/crypto_trades.parquet", index=False)
    print(f"🌙 Moon Dev saved {len(out):,} trade rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
