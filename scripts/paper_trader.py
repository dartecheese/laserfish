#!/usr/bin/env python3
"""
Paper trader: oracle-lag signal detector and simulated P&L tracker for
WIF, AR, and SNX on Hyperliquid.

Signal (delta-based, not level-based):
  - Watch Binance price change over a rolling N-second window
  - Watch HL allMids change over the same window
  - When Binance moved > entry_threshold% AND HL lagged > lag_threshold%
    behind that move → enter in the direction of Binance's move
  - Exit on: HL take-profit, stop-loss, or timeout

This avoids the structural perp discount trap: allMids sits ~0.03-0.05% below
HL's own oracle (which tracks Binance closely). That persistent gap is NOT a
lag signal. Only delta divergence — Binance moved, HL didn't yet — is real.

Break-even round-trip: 0.070% (HL taker 0.035% × 2)

Usage:
    python scripts/paper_trader.py
    python scripts/paper_trader.py --assets WIF AR --duration 1800
    python scripts/paper_trader.py --assets WIF AR SNX --size 5000 --output data/paper_trades.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import websockets

# ---------------------------------------------------------------------------
# Config per asset
# ---------------------------------------------------------------------------

ASSET_CONFIG: dict[str, dict] = {
    "WIF": {
        "binance_stream": "wifusdt@aggTrade",
        "signal_lookback_s": 5.0,      # rolling window for delta measurement
        "entry_threshold_pct": 0.12,   # Binance must move at least this much in window
        "lag_threshold_pct": 0.08,     # HL must lag at least this far behind Binance move
        "take_profit_pct": 0.07,       # HL moves this far in our direction → exit profitable
        "stop_loss_pct": 0.15,         # HL moves this far against us → cut loss
        "max_hold_ms": 25_000,         # hard timeout (p50 lag ~9.8s, use 25s cap)
        "min_batch_buffer_ms": 150,    # ms of HL window remaining required to enter
    },
    "AR": {
        "binance_stream": "arusdt@aggTrade",
        "signal_lookback_s": 8.0,
        "entry_threshold_pct": 0.15,
        "lag_threshold_pct": 0.10,
        "take_profit_pct": 0.09,
        "stop_loss_pct": 0.20,
        "max_hold_ms": 50_000,         # p50 lag ~18.9s
        "min_batch_buffer_ms": 150,
    },
    "SNX": {
        "binance_stream": "snxusdt@aggTrade",
        "signal_lookback_s": 15.0,
        "entry_threshold_pct": 0.15,
        "lag_threshold_pct": 0.10,
        "take_profit_pct": 0.09,
        "stop_loss_pct": 0.20,
        "max_hold_ms": 100_000,        # p50 lag ~52s
        "min_batch_buffer_ms": 150,
    },
}

HL_WS_URL = "wss://api.hyperliquid.xyz/ws"
BINANCE_WS_URL = "wss://stream.binance.com:9443/stream"
HL_BATCH_PERIOD_MS = 1022  # measured mean from 6,466-sample probe

TAKER_FEE_PCT = 0.035
ROUND_TRIP_COST_PCT = TAKER_FEE_PCT * 2  # 0.070%

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    asset: str
    direction: int               # +1 long, -1 short
    binance_delta_pct: float     # Binance move over lookback window that triggered signal
    hl_delta_pct: float          # HL move over same window at entry (how much it lagged)
    lag_pct: float               # binance_delta - hl_delta (the gap we're betting will close)
    hl_entry_price: float
    entry_ns: int
    batch_offset_ms: float
    next_batch_ms: float
    size_usd: float

    # Filled on close
    hl_exit_price: float = 0.0
    exit_ns: int = 0
    exit_reason: str = ""
    hold_ms: float = 0.0
    gross_pnl_pct: float = 0.0
    net_pnl_pct: float = 0.0
    net_pnl_usd: float = 0.0


@dataclass
class BatchClock:
    last_batch_ns: int = 0
    intervals_ms: list[float] = field(default_factory=list)

    def record(self, ts_ns: int) -> None:
        if self.last_batch_ns > 0:
            interval = (ts_ns - self.last_batch_ns) / 1e6
            if 200 < interval < 3000:
                self.intervals_ms.append(interval)
        self.last_batch_ns = ts_ns

    def ms_since_last(self, now_ns: int) -> float:
        if self.last_batch_ns == 0:
            return HL_BATCH_PERIOD_MS
        return (now_ns - self.last_batch_ns) / 1e6

    def ms_until_next(self, now_ns: int) -> float:
        return max(0.0, HL_BATCH_PERIOD_MS - self.ms_since_last(now_ns))

    @property
    def mean_period_ms(self) -> float:
        if not self.intervals_ms:
            return HL_BATCH_PERIOD_MS
        return sum(self.intervals_ms[-200:]) / len(self.intervals_ms[-200:])


# ---------------------------------------------------------------------------
# Paper trader
# ---------------------------------------------------------------------------

class PaperTrader:
    def __init__(self, assets: list[str], duration_s: int, size_usd: float, output: Optional[Path]):
        self.assets = assets
        self.duration_s = duration_s
        self.size_usd = size_usd
        self.output = output

        # Rolling price histories: deque of (perf_counter_ns, price)
        max_history = 500
        self._binance_hist: dict[str, deque] = {a: deque(maxlen=max_history) for a in assets}
        self._hl_hist: dict[str, deque] = {a: deque(maxlen=max_history) for a in assets}

        self._binance: dict[str, float] = {}
        self._hl: dict[str, float] = {}

        self._open: dict[str, Optional[Trade]] = {a: None for a in assets}
        self.closed: list[Trade] = []
        self._clock = BatchClock()

        self._signals_fired = 0
        self._signals_skipped_window = 0
        self._signals_skipped_no_data = 0

    # ------------------------------------------------------------------ #
    # Delta helpers                                                        #
    # ------------------------------------------------------------------ #

    def _ref_price(self, hist: deque, lookback_ns: int, now_ns: int) -> Optional[float]:
        """Find the most-recent price that is at least lookback_ns old."""
        target = now_ns - lookback_ns
        ref = None
        for ts, price in hist:
            if ts <= target:
                ref = price
        return ref

    def _delta_pct(self, hist: deque, lookback_s: float, now_ns: int) -> Optional[float]:
        """Return % change from lookback_s seconds ago to most-recent tick."""
        if not hist:
            return None
        lookback_ns = int(lookback_s * 1e9)
        ref = self._ref_price(hist, lookback_ns, now_ns)
        if ref is None or ref == 0:
            return None
        current = hist[-1][1]
        return (current - ref) / ref * 100

    # ------------------------------------------------------------------ #
    # Signal logic                                                         #
    # ------------------------------------------------------------------ #

    def _try_entry(self, asset: str, now_ns: int) -> None:
        if self._open[asset] is not None:
            return

        cfg = ASSET_CONFIG[asset]
        lookback_s = cfg["signal_lookback_s"]

        b_delta = self._delta_pct(self._binance_hist[asset], lookback_s, now_ns)
        h_delta = self._delta_pct(self._hl_hist[asset], lookback_s, now_ns)

        if b_delta is None or h_delta is None:
            self._signals_skipped_no_data += 1
            return

        # How much has HL lagged behind Binance's move?
        # Positive lag means Binance moved up more than HL did (or down less)
        lag = b_delta - h_delta  # positive = HL behind Binance (long signal)

        if abs(b_delta) < cfg["entry_threshold_pct"]:
            return
        if abs(lag) < cfg["lag_threshold_pct"]:
            return
        # Direction: lag>0 means HL hasn't moved up enough → go long on HL
        # lag<0 means HL hasn't moved down enough → go short on HL
        direction = 1 if lag > 0 else -1

        # Clock check
        next_batch_ms = self._clock.ms_until_next(now_ns)
        if next_batch_ms < cfg["min_batch_buffer_ms"]:
            self._signals_skipped_window += 1
            return

        hl_price = self._hl.get(asset)
        if hl_price is None:
            return

        self._signals_fired += 1
        dir_str = "LONG " if direction == 1 else "SHORT"
        print(
            f"  ENTRY  {asset:<5} {dir_str} | "
            f"HL={hl_price:.5f}  Binance_Δ={b_delta:+.3f}%  HL_Δ={h_delta:+.3f}%  "
            f"lag={lag:+.3f}% | window={next_batch_ms:.0f}ms"
        )

        t = Trade(
            asset=asset,
            direction=direction,
            binance_delta_pct=b_delta,
            hl_delta_pct=h_delta,
            lag_pct=lag,
            hl_entry_price=hl_price,
            entry_ns=now_ns,
            batch_offset_ms=self._clock.ms_since_last(now_ns),
            next_batch_ms=next_batch_ms,
            size_usd=self.size_usd,
        )
        self._open[asset] = t

    def _try_exit(self, asset: str, now_ns: int) -> None:
        trade = self._open[asset]
        if trade is None:
            return

        hl = self._hl.get(asset)
        if hl is None or hl == 0:
            return

        cfg = ASSET_CONFIG[asset]
        hold_ms = (now_ns - trade.entry_ns) / 1e6

        gross_pnl_pct = trade.direction * (hl - trade.hl_entry_price) / trade.hl_entry_price * 100

        # Take profit: HL moved in our direction enough to cover fees and make a profit
        take_profit = gross_pnl_pct >= cfg["take_profit_pct"]
        # Stop loss: HL moved against us
        stop_loss = gross_pnl_pct <= -cfg["stop_loss_pct"]
        # Timeout
        timed_out = hold_ms >= cfg["max_hold_ms"]

        reason = None
        if take_profit:
            reason = "take_profit"
        elif stop_loss:
            reason = "stop_loss"
        elif timed_out:
            reason = "timeout"

        if reason is None:
            return

        net_pnl_pct = gross_pnl_pct - ROUND_TRIP_COST_PCT
        net_pnl_usd = net_pnl_pct / 100 * self.size_usd

        trade.hl_exit_price = hl
        trade.exit_ns = now_ns
        trade.exit_reason = reason
        trade.hold_ms = hold_ms
        trade.gross_pnl_pct = gross_pnl_pct
        trade.net_pnl_pct = net_pnl_pct
        trade.net_pnl_usd = net_pnl_usd

        self.closed.append(trade)
        self._open[asset] = None

        sign = "+" if net_pnl_pct >= 0 else ""
        print(
            f"  EXIT   {asset:<5} {reason:<11} | "
            f"entry={trade.hl_entry_price:.5f} exit={hl:.5f} "
            f"hold={hold_ms/1000:.1f}s | "
            f"net: {sign}{net_pnl_pct:.3f}% (${net_pnl_usd:+.2f})"
        )

    # ------------------------------------------------------------------ #
    # Listeners                                                            #
    # ------------------------------------------------------------------ #

    async def _listen_hl(self) -> None:
        while True:
            try:
                async with websockets.connect(HL_WS_URL, ping_interval=20) as ws:
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "allMids"},
                    }))
                    prev_prices: dict[str, float] = {}
                    async for raw in ws:
                        ts = time.perf_counter_ns()
                        msg = json.loads(raw)
                        if msg.get("channel") != "allMids":
                            continue
                        mids: dict[str, str] = msg.get("data", {}).get("mids", {})
                        if mids:
                            self._clock.record(ts)
                        for sym in self.assets:
                            if sym not in mids:
                                continue
                            price = float(mids[sym])
                            self._hl[sym] = price
                            self._hl_hist[sym].append((ts, price))
                            # Only check exits/entries when price actually changes
                            if price != prev_prices.get(sym, 0.0):
                                self._try_exit(sym, ts)
                                self._try_entry(sym, ts)
                            prev_prices[sym] = price
            except Exception as exc:
                print(f"[HL WS] reconnecting: {exc}")
                await asyncio.sleep(2)

    async def _listen_binance(self) -> None:
        streams = "/".join(ASSET_CONFIG[a]["binance_stream"] for a in self.assets)
        url = f"{BINANCE_WS_URL}?streams={streams}"
        reverse = {
            ASSET_CONFIG[a]["binance_stream"].split("@")[0].replace("usdt", "").upper(): a
            for a in self.assets
        }
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for raw in ws:
                        ts = time.perf_counter_ns()
                        msg = json.loads(raw)
                        data = msg.get("data", {})
                        if data.get("e") != "aggTrade":
                            continue
                        ticker = data["s"].replace("USDT", "").upper()
                        sym = reverse.get(ticker)
                        if sym is None:
                            continue
                        price = float(data["p"])
                        self._binance[sym] = price
                        self._binance_hist[sym].append((ts, price))
                        self._try_entry(sym, ts)
            except Exception as exc:
                print(f"[Binance WS] reconnecting: {exc}")
                await asyncio.sleep(2)

    # ------------------------------------------------------------------ #
    # Stats display                                                        #
    # ------------------------------------------------------------------ #

    async def _print_stats(self) -> None:
        while True:
            await asyncio.sleep(60)
            self._report()

    def _report(self) -> None:
        trades = self.closed
        print(f"\n{'═'*80}")
        print(f"  PAPER TRADE SUMMARY  |  assets={self.assets}  |  size=${self.size_usd:,.0f}/trade")
        print(f"  Batch clock: mean={self._clock.mean_period_ms:.0f}ms  "
              f"n={len(self._clock.intervals_ms)}")
        print(f"  Signals fired={self._signals_fired}  "
              f"skipped(window)={self._signals_skipped_window}  "
              f"skipped(no_data)={self._signals_skipped_no_data}")
        print(f"{'─'*80}")

        if not trades:
            print("  No completed trades yet.")
            print(f"{'═'*80}\n")
            return

        by_asset: dict[str, list[Trade]] = {}
        for t in trades:
            by_asset.setdefault(t.asset, []).append(t)

        total_pnl = 0.0
        print(f"  {'Asset':<6} {'n':>4} {'wins':>5} {'win%':>6} "
              f"{'mean net%':>10} {'total $':>10} {'avg hold':>9} {'exits'}")
        print(f"  {'─'*76}")
        for asset, ts in sorted(by_asset.items()):
            wins = [t for t in ts if t.net_pnl_pct > 0]
            mean_net = sum(t.net_pnl_pct for t in ts) / len(ts)
            total = sum(t.net_pnl_usd for t in ts)
            avg_hold = sum(t.hold_ms for t in ts) / len(ts) / 1000
            reasons: dict[str, int] = {}
            for t in ts:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            reason_str = " ".join(f"{k}:{v}" for k, v in reasons.items())
            total_pnl += total
            print(
                f"  {asset:<6} {len(ts):>4} {len(wins):>5} "
                f"{len(wins)/len(ts)*100:>5.0f}% "
                f"{mean_net:>+9.3f}% {total:>+9.2f}$ "
                f"{avg_hold:>7.1f}s  {reason_str}"
            )
        print(f"  {'─'*76}")
        print(f"  {'TOTAL':<6} {len(trades):>4}{'':>36} {total_pnl:>+9.2f}$")

        print(f"\n  Recent trades (last 10):")
        for t in trades[-10:]:
            sign = "+" if t.net_pnl_pct >= 0 else ""
            dir_str = "L" if t.direction == 1 else "S"
            print(
                f"    {t.asset:<5} {dir_str} "
                f"BΔ={t.binance_delta_pct:+.3f}% HΔ={t.hl_delta_pct:+.3f}% lag={t.lag_pct:+.3f}% "
                f"hold={t.hold_ms/1000:.1f}s "
                f"net={sign}{t.net_pnl_pct:.3f}% ({sign}${t.net_pnl_usd:.2f}) [{t.exit_reason}]"
            )
        print(f"{'═'*80}\n")

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        print(f"\n{'═'*80}")
        print(f"  ORACLE LAG PAPER TRADER  (delta-signal)")
        print(f"  Assets: {self.assets}")
        print(f"  Duration: {self.duration_s}s | Size: ${self.size_usd:,.0f}/trade")
        print(f"  Break-even: {ROUND_TRIP_COST_PCT:.3f}% round-trip")
        print(f"{'─'*80}")
        for a in self.assets:
            cfg = ASSET_CONFIG[a]
            print(f"  {a}: lookback={cfg['signal_lookback_s']:.0f}s  "
                  f"B_move>{cfg['entry_threshold_pct']:.2f}%  "
                  f"lag>{cfg['lag_threshold_pct']:.2f}%  "
                  f"TP={cfg['take_profit_pct']:.2f}%  "
                  f"SL={cfg['stop_loss_pct']:.2f}%  "
                  f"timeout={cfg['max_hold_ms']//1000}s")
        print(f"{'═'*80}\n")
        print("Connecting... (first stats in 60s)\n")

        tasks = [
            asyncio.create_task(self._listen_hl(), name="hl"),
            asyncio.create_task(self._listen_binance(), name="binance"),
            asyncio.create_task(self._print_stats(), name="stats"),
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.duration_s,
            )
        except asyncio.TimeoutError:
            pass
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Close any open positions at last known price
        for asset, trade in self._open.items():
            if trade is not None:
                hl = self._hl.get(asset, trade.hl_entry_price)
                now_ns = time.perf_counter_ns()
                gross = trade.direction * (hl - trade.hl_entry_price) / trade.hl_entry_price * 100
                net = gross - ROUND_TRIP_COST_PCT
                trade.hl_exit_price = hl
                trade.exit_ns = now_ns
                trade.exit_reason = "session_end"
                trade.hold_ms = (now_ns - trade.entry_ns) / 1e6
                trade.gross_pnl_pct = gross
                trade.net_pnl_pct = net
                trade.net_pnl_usd = net / 100 * self.size_usd
                self.closed.append(trade)

        self._report()
        self._save()

    def _save(self) -> None:
        if not self.output or not self.closed:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(asdict(self.closed[0]).keys())
        with open(self.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in self.closed:
                w.writerow(asdict(t))
        print(f"Saved {len(self.closed)} trades → {self.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle-lag paper trader (delta signal)")
    parser.add_argument("--assets", nargs="+", default=["WIF", "AR", "SNX"])
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--size", type=float, default=5000.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    assets = [a.upper() for a in args.assets]
    unknown = [a for a in assets if a not in ASSET_CONFIG]
    if unknown:
        print(f"Unknown assets: {unknown}. Available: {list(ASSET_CONFIG)}")
        return

    asyncio.run(PaperTrader(
        assets=assets,
        duration_s=args.duration,
        size_usd=args.size,
        output=args.output,
    ).run())


if __name__ == "__main__":
    main()
