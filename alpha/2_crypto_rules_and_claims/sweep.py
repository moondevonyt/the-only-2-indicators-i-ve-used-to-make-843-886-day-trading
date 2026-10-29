"""Moon Dev: run every config on every coin (16 Binance perps), save additive stats to cache/sweep.parquet."""
import time
import pandas as pd
from multiprocessing import Pool
import core, engine, setups


def one_coin(sym):
    t0 = time.time()
    rows = []
    specs = setups.grid_specs()
    order = ["5m", "15m", "1h", "4h", "1d", "1m/2m/5m"]
    for tf in order:                              # Moon Dev: one TF at a time, free memory after
        for spec in [s for s in specs if s["tf"] == tf]:
            t = engine.trades_for(sym, spec)
            row = dict(spec); row["market"] = sym
            row.update(engine.sums(t) if len(t) else {})
            rows.append(row)
        engine.clear()
    print(f"Moon Dev sweep {sym} done in {time.time() - t0:.0f}s, {len(rows)} configs", flush=True)
    return rows


if __name__ == "__main__":
    with Pool(8) as p:
        out = p.map(one_coin, core.SYMS)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_parquet(f"{core.CACHE}/sweep.parquet")
    print("Moon Dev: sweep saved", df.shape)
