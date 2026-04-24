"""Market data fetching and caching.

Training data comes from Binance (public, no auth needed).
Live data comes from Hyperliquid via ccxt.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

BINANCE_BASE = "https://api.binance.com"
KLINES_MAX = 1000


@dataclass
class Kline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float        # base-asset volume (e.g. BTC)
    close_time: int
    quote_volume: float  # quote-asset volume (e.g. USDT)
    trades: int

    @classmethod
    def from_binance_row(cls, row: list) -> "Kline":
        return cls(
            open_time=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            close_time=int(row[6]),
            quote_volume=float(row[7]),
            trades=int(row[8]),
        )


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    session: requests.Session | None = None,
    throttle_s: float = 0.25,
) -> list[Kline]:
    """Paginate Binance public klines endpoint between [start_ms, end_ms]."""
    s = session or requests.Session()
    out: list[Kline] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": KLINES_MAX,
        }
        r = s.get(f"{BINANCE_BASE}/api/v3/klines", params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out.extend(Kline.from_binance_row(row) for row in rows)
        cursor = out[-1].close_time + 1
        time.sleep(throttle_s)

    return out


def save(klines: list[Kline], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(k) for k in klines]))


def load(path: Path) -> list[Kline]:
    return [Kline(**d) for d in json.loads(path.read_text())]


def cache_path(data_dir: str | Path, symbol: str, interval: str) -> Path:
    return Path(data_dir) / f"{symbol}_{interval}.json"
