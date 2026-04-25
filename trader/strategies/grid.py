"""Grid trading strategy for ranging markets.

Places a ladder of limit buy/sell orders around a center price.
Each oscillation within the grid earns the spread minus fees.
Auto-closes when regime shifts from RANGE to TREND or CRISIS.

Design:
  - N levels above center (sell orders) + N levels below (buy orders)
  - When a buy fills → immediately place sell one level up
  - When a sell fills → immediately place buy one level down
  - Net position tracks directional exposure from imbalanced fills
  - Auto-recenters if price drifts >drift_pct from center
  - Paper mode: simulates limit fills via price crossings on each check()

Live order flow on Hyperliquid:
  - Plain limit orders (no TP/SL on the grid legs)
  - Order IDs tracked in self._live_orders
  - check() polls status; filled orders trigger re-placement
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)

FEE = 0.00015   # 1.5bps maker (limit orders) — Hyperliquid maker rebate territory


@dataclass
class GridConfig:
    symbol: str = "BTC"

    # Grid geometry
    spacing_pct: float = 0.005    # 0.5% between each level
    n_levels: int = 10            # levels each side (total 20 orders)

    # Sizing — each level gets order_size_pct × equity / mark_price contracts
    order_size_pct: float = 0.04  # 4% of equity per level (20 levels × 4% = 80% deployed)
    max_leverage: float = 2.0

    # Safety
    drift_pct: float = 0.04       # re-center if price moves this far from center
    max_net_position_pct: float = 0.30  # close grid if net exposure > 30% of equity


@dataclass
class GridLevel:
    """One resting limit order in the grid."""
    level_idx: int        # signed: negative = below center, positive = above
    side: str             # "buy" | "sell"
    price: float
    qty: float
    order_id: str = ""    # live mode: exchange order ID; paper: ""
    filled: bool = False
    fill_price: float = 0.0
    fill_time: float = 0.0


class GridFill(NamedTuple):
    level_idx: int
    side: str
    price: float
    qty: float
    pnl: float            # realized PnL for a completed round trip (0 if just half)


class GridStrategy:
    """
    Grid market maker for ranging regimes.

    Usage in run.py:
        grid = GridStrategy(exchange, GridConfig(symbol="BTC"))

        # When HMM enters RANGE regime:
        grid.open(mark_price, equity)

        # Every 60s:
        fills = grid.check(mark_price, equity)

        # When HMM exits RANGE regime:
        grid.close()
    """

    def __init__(self, exchange, cfg: GridConfig | None = None):
        self.ex = exchange
        self.cfg = cfg or GridConfig()
        self._levels: dict[int, GridLevel] = {}   # level_idx → GridLevel
        self._center: float = 0.0
        self._active: bool = False
        self._realized_pnl: float = 0.0
        self._n_round_trips: int = 0
        self._net_qty: float = 0.0    # positive = net long, negative = net short
        self._open_time: float = 0.0
        # Track paired fills for PnL: buy_idx → fill_price (awaiting matching sell)
        self._pending_buy_fills: dict[int, float] = {}
        self._pending_sell_fills: dict[int, float] = {}

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def n_round_trips(self) -> int:
        return self._n_round_trips

    def open(self, mark_price: float, equity: float) -> None:
        """Place the initial grid around mark_price."""
        if self._active:
            logger.warning("Grid already active — call close() first")
            return

        self._center = mark_price
        self._active = True
        self._open_time = time.time()
        self._levels.clear()
        self._net_qty = 0.0
        self._realized_pnl = 0.0
        self._n_round_trips = 0

        qty = self._order_qty(mark_price, equity)
        placed = 0
        for i in range(1, self.cfg.n_levels + 1):
            buy_price  = mark_price * (1 - self.cfg.spacing_pct * i)
            sell_price = mark_price * (1 + self.cfg.spacing_pct * i)
            self._place_level(-i, "buy",  buy_price,  qty)
            self._place_level(+i, "sell", sell_price, qty)
            placed += 2

        logger.info(
            "Grid OPEN | %s | center=%.4f | levels=%d×2 | spacing=%.2f%% | qty/level=%.4f",
            self.cfg.symbol, mark_price, self.cfg.n_levels,
            self.cfg.spacing_pct * 100, qty,
        )

    def check(self, mark_price: float, equity: float) -> list[GridFill]:
        """
        Check for fills (paper: price crossings; live: order status poll).
        Re-places orders for each fill. Returns list of fill events.
        """
        if not self._active:
            return []

        fills: list[GridFill] = []

        # Detect fills
        for idx, lvl in list(self._levels.items()):
            if lvl.filled:
                continue
            filled = self._check_fill(lvl, mark_price)
            if not filled:
                continue

            lvl.filled = True
            lvl.fill_price = mark_price if not lvl.fill_price else lvl.fill_price
            lvl.fill_time = time.time()

            if lvl.side == "buy":
                self._net_qty += lvl.qty
                # Re-place sell one level up; that sell pairs with this buy
                new_idx = idx + 1
                new_price = self._center * (1 + self.cfg.spacing_pct * new_idx)
                pnl = 0.0
                # Check if the sell one level up already filled (price whipsawed)
                pair_key = new_idx
                if pair_key in self._pending_sell_fills:
                    sell_px = self._pending_sell_fills.pop(pair_key)
                    pnl = (sell_px - lvl.fill_price) * lvl.qty - FEE * 2 * lvl.qty * lvl.fill_price
                    self._realized_pnl += pnl
                    self._n_round_trips += 1
                else:
                    self._pending_buy_fills[new_idx] = lvl.fill_price  # keyed by paired sell idx
                fills.append(GridFill(idx, "buy", lvl.fill_price, lvl.qty, pnl))
                self._place_level(new_idx, "sell", new_price, lvl.qty)

            else:  # sell
                self._net_qty -= lvl.qty
                # Re-place buy one level down; check if that buy already filled
                new_idx = idx - 1
                new_price = self._center * (1 - self.cfg.spacing_pct * abs(new_idx))
                pnl = 0.0
                # This sell completes the round trip started by buy at (idx)
                pair_key = idx
                if pair_key in self._pending_buy_fills:
                    buy_px = self._pending_buy_fills.pop(pair_key)
                    pnl = (lvl.fill_price - buy_px) * lvl.qty - FEE * 2 * lvl.qty * buy_px
                    self._realized_pnl += pnl
                    self._n_round_trips += 1
                else:
                    self._pending_sell_fills[idx] = lvl.fill_price
                fills.append(GridFill(idx, "sell", lvl.fill_price, lvl.qty, pnl))
                self._place_level(new_idx, "buy", new_price, lvl.qty)

        if fills:
            logger.info(
                "Grid %s | %d fill(s) | net_qty=%+.4f | realized_pnl=%+.2f | trips=%d",
                self.cfg.symbol, len(fills), self._net_qty,
                self._realized_pnl, self._n_round_trips,
            )

        # Re-center if price drifted too far
        if abs(mark_price - self._center) / self._center > self.cfg.drift_pct:
            logger.info("Grid %s: price drifted %.1f%% from center — recentering",
                        self.cfg.symbol,
                        abs(mark_price - self._center) / self._center * 100)
            self._cancel_all_live_orders()
            self._flatten_net_position(mark_price)
            self.open(mark_price, equity)

        # Safety: close if net exposure too large
        net_notional = abs(self._net_qty) * mark_price
        if equity > 0 and net_notional / equity > self.cfg.max_net_position_pct:
            logger.warning("Grid %s: net exposure %.1f%% > limit — closing grid",
                           self.cfg.symbol, net_notional / equity * 100)
            self.close(mark_price)

        return fills

    def close(self, mark_price: float | None = None) -> float:
        """Cancel all orders, flatten net position. Returns total realized PnL."""
        if not self._active:
            return self._realized_pnl
        self._cancel_all_live_orders()
        if mark_price and self._net_qty != 0:
            self._flatten_net_position(mark_price)
        self._active = False
        held = time.time() - self._open_time
        logger.info(
            "Grid CLOSE | %s | pnl=%+.2f | trips=%d | held=%.1fh",
            self.cfg.symbol, self._realized_pnl, self._n_round_trips, held / 3600,
        )
        return self._realized_pnl

    def status(self, mark_price: float) -> str:
        if not self._active:
            return f"Grid {self.cfg.symbol}: INACTIVE"
        n_pending = sum(1 for l in self._levels.values() if not l.filled)
        drift = (mark_price - self._center) / self._center * 100
        return (
            f"Grid {self.cfg.symbol} | center={self._center:.2f} | "
            f"mark={mark_price:.2f} ({drift:+.1f}%) | "
            f"pending={n_pending} | net={self._net_qty:+.4f} | "
            f"pnl={self._realized_pnl:+.2f} | trips={self._n_round_trips}"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _order_qty(self, mark_price: float, equity: float) -> float:
        notional = equity * self.cfg.order_size_pct * self.cfg.max_leverage
        return notional / mark_price if mark_price > 0 else 0.0

    def _place_level(self, idx: int, side: str, price: float, qty: float) -> None:
        """Place one grid level (paper: record for simulation; live: submit order)."""
        order_id = ""
        if not getattr(self.ex, 'paper', True):
            # Live mode: place actual limit order
            try:
                from trader.exchanges.base import BracketParams
                result = self.ex.place_bracket_order(BracketParams(
                    symbol=self.cfg.symbol,
                    side=side,
                    quantity=qty,
                    price=price,
                    take_profit_pct=None,
                    stop_loss_pct=None,
                    leverage=int(self.cfg.max_leverage),
                ))
                order_id = result.order_id
            except Exception as e:
                logger.debug("Grid place_level failed idx=%d: %s", idx, e)
                return

        self._levels[idx] = GridLevel(
            level_idx=idx, side=side, price=price, qty=qty, order_id=order_id,
        )

    def _check_fill(self, lvl: GridLevel, mark_price: float) -> bool:
        """Return True if this level should be considered filled."""
        if not getattr(self.ex, 'paper', True):
            # Live: poll order status
            if not lvl.order_id:
                return False
            try:
                status = self.ex.get_order_status(self.cfg.symbol, lvl.order_id)
                if status in ("closed", "filled"):
                    lvl.fill_price = lvl.price  # approximate
                    return True
            except Exception:
                pass
            return False
        else:
            # Paper: simulate — buy fills when price drops to/below, sell when price rises to/above
            if lvl.side == "buy"  and mark_price <= lvl.price:
                lvl.fill_price = lvl.price
                return True
            if lvl.side == "sell" and mark_price >= lvl.price:
                lvl.fill_price = lvl.price
                return True
            return False

    def _cancel_all_live_orders(self) -> None:
        if not getattr(self.ex, 'paper', True):
            try:
                self.ex.cancel_all_orders(self.cfg.symbol)
            except Exception as e:
                logger.warning("Grid cancel_all_orders: %s", e)
        self._levels.clear()

    def _flatten_net_position(self, mark_price: float) -> None:
        """Close net directional exposure from imbalanced grid fills."""
        if abs(self._net_qty) < 1e-8:
            return
        close_side = "sell" if self._net_qty > 0 else "buy"
        qty = abs(self._net_qty)
        # PnL from closing at market
        avg_entry = mark_price  # simplified — real PnL already tracked per fill
        if not getattr(self.ex, 'paper', True):
            try:
                from trader.exchanges.base import BracketParams
                self.ex.place_bracket_order(BracketParams(
                    symbol=self.cfg.symbol, side=close_side,
                    quantity=qty, price=None,
                    leverage=int(self.cfg.max_leverage),
                ))
            except Exception as e:
                logger.warning("Grid flatten position: %s", e)
        self._net_qty = 0.0
