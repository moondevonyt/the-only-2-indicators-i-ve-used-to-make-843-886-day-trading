"""Moon Dev: load Binance 1m perp klines (real data only) and cache 5m / 15m / 1h / 4h / 1d bars.
Fails loudly if the drive or a coin is missing."""
import os, glob, sys
import pandas as pd
from multiprocessing import Pool

ROOT = "/Volumes/Moon 26/tick_data/binance"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
SYMS = ["ARB", "AVAX", "BNB", "BTC", "CRV", "DOGE", "ETH", "FIL", "INIT", "LINK", "SOL", "STRK", "SUI", "TIA", "XRP", "ZEC"]
TFS = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def all_in_one(sym, kind="futures_klines_1m"):
    """Moon Dev: the one big parquet per coin (name has a date range, not a single month)."""
    files = [f for f in glob.glob(f"{ROOT}/{kind}/{sym}USDT/{sym}USDT_1m_*_*.parquet") if os.path.basename(f).count("_") == 3]
    assert len(files) == 1, f"Moon Dev: expected 1 all-in-one file for {sym} in {kind}, found {files}"
    return files[0]


def load_1m(sym, kind="futures_klines_1m"):
    d = pd.read_parquet(all_in_one(sym, kind), columns=["open_time", "open", "high", "low", "close", "volume"])
    d = d.drop_duplicates("open_time").set_index("open_time").sort_index()
    return d


def resample(d, rule):
    r = d.resample(rule, label="left", closed="left").agg(AGG)
    return r.dropna(subset=["close"])  # Moon Dev: Binance outage holes stay holes, no fake fill


def build(sym):
    d = load_1m(sym)
    for tf, rule in TFS.items():
        resample(d, rule).to_parquet(f"{CACHE}/{sym}_{tf}.parquet")
    print(f"Moon Dev cached {sym}: {d.index[0]} -> {d.index[-1]}, {len(d):,} 1m bars", flush=True)
    return sym, str(d.index[0]), str(d.index[-1]), len(d)


if __name__ == "__main__":
    assert os.path.isdir(ROOT), "Moon Dev: Moon 26 drive not mounted, no data, stopping"
    os.makedirs(CACHE, exist_ok=True)
    with Pool(8) as p:
        out = p.map(build, SYMS)
    pd.DataFrame(out, columns=["sym", "start", "end", "bars_1m"]).to_csv(f"{CACHE}/coverage.csv", index=False)
    print("Moon Dev: all bars cached")
