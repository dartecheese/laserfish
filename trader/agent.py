"""Live trading agent — laserfish 5m perp scalper.

Runs a continuous loop:
  1. Fetch 5m candles → construct volume bars
  2. Build feature sequence for each symbol
  3. Run ONNX Transformer → alpha signal
  4. Risk checks → size position
  5. Execute bracket order on Hyperliquid perps (entry + TP + SL)
  6. Sleep until next 5m bar
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .bars import Bar, make_volume_bars, auto_threshold
from .data import Kline
from .exchanges.base import Exchange, BracketParams
from .features import live_sequence, SEQ_LEN
from .model import load_onnx, infer_alpha
from .risk import RiskConfig, RiskManager

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    symbols: list[str]           # base assets, e.g. ["BTC", "ETH", "SOL"]
    model_path: str              # path to ONNX file
    interval: str = "5m"         # candle interval — laserfish default is 5m
    candle_lookback: int = 500   # candles to fetch per tick (~41h of 5m history)
    target_bars_day: float = 48.0  # more vol-bars per day at 5m resolution
    bar_type: str = "volume"
    cusum_h: float = 0.003       # smaller CUSUM threshold for 5m returns

    # Triple-barrier params — 1h hold horizon at 5m resolution
    pt: float = 0.015
    sl: float = 0.015
    t_max_bars: int = 12         # 12 × 5m = 1h max hold

    leverage: int = 2
    poll_seconds: int = 300      # one poll per 5m candle
    risk: RiskConfig = field(default_factory=RiskConfig)
    dry_run: bool = True         # if True, log decisions but don't place orders


def _candles_to_klines(candles) -> list[Kline]:
    """Convert exchange Candle objects to Kline format for bar construction."""
    out = []
    for c in candles:
        out.append(Kline(
            open_time=int(c.timestamp),
            open=float(c.open),
            high=float(c.high),
            low=float(c.low),
            close=float(c.close),
            volume=float(c.volume),
            close_time=int(c.timestamp) + 3_600_000,  # approx
            quote_volume=float(c.quote_volume),
            trades=0,
        ))
    return out


class TradingAgent:
    def __init__(self, exchange: Exchange, cfg: AgentConfig):
        self.exchange = exchange
        self.cfg = cfg
        self.session = load_onnx(cfg.model_path)
        self.risk = RiskManager(cfg.risk)
        self._bars_cache: dict[str, list[Bar]] = {}

    def _fetch_bars(self, symbol: str) -> list[Bar]:
        candles = self.exchange.get_candles(symbol, self.cfg.interval, self.cfg.candle_lookback)
        klines = _candles_to_klines(candles)
        if not klines:
            return []
        threshold = auto_threshold(klines, self.cfg.target_bars_day, self.cfg.bar_type)
        if self.cfg.bar_type == "volume":
            return make_volume_bars(klines, threshold)
        from .bars import make_dollar_bars
        return make_dollar_bars(klines, threshold)

    def _get_alpha(self, bars: list[Bar]) -> float:
        if len(bars) < SEQ_LEN:
            return 0.0
        seq = live_sequence(bars)
        return infer_alpha(self.session, seq)

    def tick(self) -> None:
        """One trading cycle: fetch → signal → execute."""
        try:
            balance = self.exchange.get_balance()
            positions = self.exchange.get_positions()
            equity = balance.total_usd
        except Exception as e:
            log.error(f"Balance/positions fetch failed: {e}")
            return

        if self.risk.check_drawdown(equity):
            log.warning(f"Max drawdown breached — halting new positions (equity={equity:.0f})")
            return

        for symbol in self.cfg.symbols:
            try:
                self._process_symbol(symbol, equity, balance, positions)
            except Exception as e:
                log.error(f"{symbol}: error in tick — {e}")

    def _process_symbol(self, symbol: str, equity: float, balance, positions: list) -> None:
        bars = self._fetch_bars(symbol)
        if len(bars) < SEQ_LEN + 5:
            log.debug(f"{symbol}: insufficient bars ({len(bars)})")
            return

        self._bars_cache[symbol] = bars
        alpha = self._get_alpha(bars)
        current_price = bars[-1].close
        log.info(f"{symbol}: α={alpha:.3f}  price={current_price:.2f}")

        open_pos = {p.symbol: p for p in positions}

        # Exit logic: if we have a position and signal flips or goes neutral
        if symbol in open_pos:
            pos = open_pos[symbol]
            signal_flip = (pos.side == "long" and alpha < -self.cfg.risk.min_alpha) or \
                          (pos.side == "short" and alpha > self.cfg.risk.min_alpha)
            if signal_flip:
                log.info(f"{symbol}: signal flip → close {pos.side}")
                if not self.cfg.dry_run:
                    self.exchange.close_position(symbol)
            return   # don't open a new position this tick

        # Entry logic
        if abs(alpha) < self.cfg.risk.min_alpha:
            return

        if not self.risk.can_open_position(symbol, positions, balance, equity):
            log.debug(f"{symbol}: risk limits prevent new position")
            return

        qty = self.risk.size_position(alpha, equity, current_price, self.cfg.leverage)
        if qty <= 0:
            return

        side = "buy" if alpha > 0 else "sell"
        log.info(f"{symbol}: OPEN {side.upper()} qty={qty:.4f}  α={alpha:.3f}")

        if not self.cfg.dry_run:
            self.exchange.place_bracket_order(BracketParams(
                symbol=symbol,
                side=side,
                quantity=qty,
                price=None,   # market order
                take_profit_pct=self.cfg.pt,
                stop_loss_pct=self.cfg.sl,
                leverage=self.cfg.leverage,
            ))

    def run(self) -> None:
        log.info(f"Agent started | exchange={self.exchange.name} | symbols={self.cfg.symbols}")
        log.info(f"Model: {self.cfg.model_path} | dry_run={self.cfg.dry_run}")
        while True:
            self.tick()
            time.sleep(self.cfg.poll_seconds)
