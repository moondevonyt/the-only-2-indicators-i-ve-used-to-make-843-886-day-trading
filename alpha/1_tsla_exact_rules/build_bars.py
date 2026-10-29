"""
Moon Dev: build TSLA RTH bars ONCE from real Databento Nasdaq Basic trades.
- Source: /Volumes/Moon 26/tick_data/databento/xnas_basic_trades/TSLA/*.parquet
- Uses Nasdaq lit prints only (publisher_id 81). TRF / off-exchange prints are skipped (late + odd prices).
- RTH only: 09:30 to 16:00 America/New_York (DST handled by tz_convert).
- Skips Databento's flagged degraded days: 2024-07-25, 2025-12-15, 2026-01-12.
- Writes bars_1m.parquet in this folder. Everything else (2m, 5m, 15m, 1h, daily) is built from it in backtest.py.
"""
import glob, os, sys
from multiprocessing import Pool
import pandas as pd
import pyarrow.parquet as pq

SRC = "/Volumes/Moon 26/tick_data/databento/xnas_basic_trades/TSLA"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bars_1m.parquet")
DEGRADED = {"2024-07-25", "2025-12-15", "2026-01-12"}


def one_file(path):
    pf = pq.ParquetFile(path)
    parts = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["ts_recv", "price", "size", "publisher_id"]).to_pandas()
        t = t[t["publisher_id"] == 81]
        if t.empty:
            continue
        ts = t.index if "ts_recv" not in t.columns else t["ts_recv"]
        et = pd.DatetimeIndex(ts).tz_convert("America/New_York")
        mins = et.hour * 60 + et.minute
        keep = (mins >= 570) & (mins < 960)  # 09:30 <= t < 16:00 ET
        if not keep.any():
            continue
        d = pd.DataFrame({"price": t["price"].values[keep], "size": t["size"].values[keep]},
                         index=et[keep].floor("1min"))
        g = d.groupby(level=0)
        b = pd.DataFrame({"open": g["price"].first(), "high": g["price"].max(), "low": g["price"].min(),
                          "close": g["price"].last(), "volume": g["size"].sum(), "n": g["price"].size()})
        parts.append(b)
    if not parts:
        return None
    # row groups can split a minute, so merge the pieces in time order (Moon Dev)
    b = pd.concat(parts).sort_index(kind="stable")
    g = b.groupby(level=0)
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum(), "n": g["n"].sum()})
    print(f"Moon Dev bars: {os.path.basename(path)} -> {len(b):,} minutes", flush=True)
    return b


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SRC, "TSLA_*.parquet")))
    if not files:
        sys.exit(f"Moon Dev: NO TSLA files found in {SRC}. Is Moon 26 mounted?")
    with Pool(6) as p:
        res = [r for r in p.map(one_file, files) if r is not None]
    bars = pd.concat(res).sort_index(kind="stable")
    g = bars.groupby(level=0)  # overlap between TSLA_2026-06 and the 3-month file gets merged here
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].max(), "n": g["n"].max()})
    day = bars.index.strftime("%Y-%m-%d")
    bars = bars[~day.isin(DEGRADED)]
    bars.index.name = "t_et"
    bars.to_parquet(OUT)
    days = bars.index.normalize().unique()
    print(f"Moon Dev: wrote {OUT}: {len(bars):,} 1m bars, {len(days)} days, {days[0].date()} -> {days[-1].date()}")
