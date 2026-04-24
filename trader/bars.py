"""Information-driven bar construction (Lopez de Prado, AFML ch. 2).

Aggregates OHLCV klines into volume or dollar bars, then applies
the symmetric CUSUM filter to identify candidate entry events.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Kline


@dataclass
class Bar:
    timestamp: int       # open_time of first constituent kline (ms)
    open: float
    high: float
    low: float
    close: float
    volume: float        # total base-asset volume
    dollar_volume: float # total quote-asset volume
    tick_count: int      # number of constituent klines
    vwap: float


def _flush(op: float, hi: float, lo: float, cl: float, vol: float, dv: float, ticks: int, ts: int) -> Bar:
    return Bar(
        timestamp=ts, open=op, high=hi, low=lo, close=cl,
        volume=vol, dollar_volume=dv, tick_count=ticks,
        vwap=dv / vol if vol > 1e-12 else cl,
    )


def make_volume_bars(klines: list[Kline], threshold: float) -> list[Bar]:
    bars: list[Bar] = []
    vol = dv = 0.0
    hi = -np.inf
    lo = np.inf
    op = ts = None
    ticks = 0

    for k in klines:
        if op is None:
            op, ts = k.open, k.open_time
        vol += k.volume
        dv += k.quote_volume
        hi = max(hi, k.high)
        lo = min(lo, k.low)
        ticks += 1
        if vol >= threshold:
            bars.append(_flush(op, hi, lo, k.close, vol, dv, ticks, ts))
            vol = dv = 0.0; hi = -np.inf; lo = np.inf; op = ts = None; ticks = 0

    return bars


def make_dollar_bars(klines: list[Kline], threshold: float) -> list[Bar]:
    bars: list[Bar] = []
    vol = dv = 0.0
    hi = -np.inf
    lo = np.inf
    op = ts = None
    ticks = 0

    for k in klines:
        if op is None:
            op, ts = k.open, k.open_time
        vol += k.volume
        dv += k.quote_volume
        hi = max(hi, k.high)
        lo = min(lo, k.low)
        ticks += 1
        if dv >= threshold:
            bars.append(_flush(op, hi, lo, k.close, vol, dv, ticks, ts))
            vol = dv = 0.0; hi = -np.inf; lo = np.inf; op = ts = None; ticks = 0

    return bars


def cusum_filter(bars: list[Bar], h: float) -> list[int]:
    """Symmetric CUSUM filter on bar log-returns.

    Returns indices into `bars` where cumulative log-return crosses ±h.
    These become candidate entry points for triple-barrier labeling.
    """
    if len(bars) < 2:
        return []
    log_rets = np.diff(np.log([b.close for b in bars]))
    events: list[int] = []
    s_pos = s_neg = 0.0
    for i, r in enumerate(log_rets):
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos >= h:
            s_pos = 0.0
            events.append(i + 1)
        elif s_neg <= -h:
            s_neg = 0.0
            events.append(i + 1)
    return events


def auto_threshold(klines: list[Kline], target_bars_per_day: float, kind: str = "volume") -> float:
    """Compute bar threshold so construction yields ~target_bars_per_day on average."""
    if not klines:
        raise ValueError("empty klines")
    days = max((klines[-1].close_time - klines[0].open_time) / 86_400_000, 1.0)
    total = sum(k.volume if kind == "volume" else k.quote_volume for k in klines)
    return total / days / target_bars_per_day
