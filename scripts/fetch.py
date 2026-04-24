"""Download historical klines from Binance for training.

Usage:
    python scripts/fetch.py --symbols BTC,ETH,SOL --interval 5m \
        --start 2022-01-01 --end 2025-12-31 --out data/

Note: 5m data is ~12× larger than 1h. Expect ~500k bars per symbol over 3 years.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.data import fetch_klines, save, cache_path


def parse_date(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT",
                    help="Comma-separated Binance symbols (e.g. BTCUSDT,ETHUSDT)")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    start_ms = parse_date(args.start)
    end_ms = parse_date(args.end)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    for sym in tqdm(symbols, desc="Symbols"):
        path = cache_path(out_dir, sym, args.interval)
        print(f"  Fetching {sym} {args.interval} {args.start}→{args.end} …")
        klines = fetch_klines(sym, args.interval, start_ms, end_ms, session)
        save(klines, path)
        print(f"  Saved {len(klines)} bars → {path}")


if __name__ == "__main__":
    main()
