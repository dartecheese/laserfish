"""In-memory paper trading portfolio.

Tracks simulated positions, margin, unrealized PnL, and funding payments
so that paper-mode scripts behave identically to live trading — positions
persist between ticks, equity updates with mark price, funding accrues.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from trader.exchanges.base import Balance, Order, Position


@dataclass
class _PaperPos:
    symbol: str
    side: str           # "long" | "short"
    qty: float          # positive base units
    entry_price: float
    leverage: int
    funding_pnl: float = 0.0

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side == "long" else -self.qty

    def unrealized_pnl(self, current_price: float) -> float:
        return self.signed_qty * (current_price - self.entry_price)

    def to_position(self) -> Position:
        return Position(
            symbol=self.symbol,
            side=self.side,
            quantity=self.qty,
            entry_price=self.entry_price,
            unrealized_pnl=self.unrealized_pnl(self.entry_price),  # filled by caller
            leverage=float(self.leverage),
        )


class PaperPortfolio:
    """
    Simulates a perpetuals trading account with no real exchange calls.

    Used by HyperliquidExchange in paper mode to give strategies realistic
    feedback: positions persist, equity changes, funding accrues every tick.
    """

    def __init__(self, initial_equity: float = 10_000.0) -> None:
        self.initial_equity = initial_equity
        self._cash = initial_equity
        self._margin: dict[str, float] = {}
        self._positions: dict[str, _PaperPos] = {}
        self._realized_pnl = 0.0
        self._funding_total = 0.0
        self._trades: list[dict] = []

    # ------------------------------------------------------------------ #
    # Order execution                                                      #
    # ------------------------------------------------------------------ #

    def open(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        leverage: int = 1,
    ) -> Order:
        notional = qty * price
        margin   = notional / max(leverage, 1)

        # Scale down if not enough free cash
        if margin > self._cash:
            qty    = self._cash * leverage / price
            margin = self._cash

        if qty <= 0:
            return Order(f"paper-skip-{int(time.time()*1000)}", symbol, side, 0.0, price, "cancelled")

        if symbol in self._positions:
            self.close(symbol, price)

        self._cash           -= margin
        self._margin[symbol]  = margin
        self._positions[symbol] = _PaperPos(
            symbol=symbol, side=side, qty=qty,
            entry_price=price, leverage=leverage,
        )
        self._trades.append({
            "action": "open", "symbol": symbol, "side": side,
            "qty": qty, "price": price, "ts": time.time(),
        })
        return Order(
            order_id=f"paper-{int(time.time()*1000)}",
            symbol=symbol, side=side, quantity=qty,
            price=price, status="filled",
        )

    def close(self, symbol: str, current_price: float) -> float:
        pos    = self._positions.pop(symbol, None)
        margin = self._margin.pop(symbol, 0.0)
        if pos is None:
            return 0.0
        pnl = pos.unrealized_pnl(current_price) + pos.funding_pnl
        self._cash          += margin + pnl
        self._realized_pnl  += pnl
        self._trades.append({
            "action": "close", "symbol": symbol,
            "pnl": pnl, "price": current_price, "ts": time.time(),
        })
        return pnl

    # ------------------------------------------------------------------ #
    # Funding accrual (call every tick for open positions)                #
    # ------------------------------------------------------------------ #

    def apply_funding(self, symbol: str, funding_rate_8h: float, current_price: float) -> float:
        """
        Apply one 8-hour funding payment to an open position.
        Shorts receive when funding_rate > 0; longs pay.
        Returns the payment amount (positive = received).
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0
        notional = pos.qty * current_price
        payment  = -pos.signed_qty * funding_rate_8h * notional
        pos.funding_pnl  += payment
        self._cash       += payment
        self._funding_total += payment
        return payment

    # ------------------------------------------------------------------ #
    # Account queries                                                      #
    # ------------------------------------------------------------------ #

    def get_balance(self, prices: dict[str, float]) -> Balance:
        equity    = self._equity(prices)
        available = max(0.0, self._cash)
        return Balance(total_usd=equity, available_usd=available)

    def get_positions(self, prices: dict[str, float]) -> list[Position]:
        result = []
        for sym, pos in self._positions.items():
            price = prices.get(sym, pos.entry_price)
            p = pos.to_position()
            p.unrealized_pnl = pos.unrealized_pnl(price) + pos.funding_pnl
            result.append(p)
        return result

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def summary(self, prices: dict[str, float]) -> str:
        equity = self._equity(prices)
        ret_pct = (equity - self.initial_equity) / self.initial_equity * 100
        lines = [
            f"  Equity   ${equity:>10,.2f}   ({ret_pct:+.2f}%)",
            f"  Realized ${self._realized_pnl:>+10,.2f}   Funding ${self._funding_total:>+,.2f}",
        ]
        for sym, pos in self._positions.items():
            price = prices.get(sym, pos.entry_price)
            upnl  = pos.unrealized_pnl(price)
            lines.append(
                f"  {'SHORT' if pos.side=='short' else 'LONG ':5s} {sym:8s}"
                f"  {pos.qty:.4f}@{pos.entry_price:.2f}"
                f"  mark={price:.2f}"
                f"  uPnL={upnl:>+8.2f}"
                f"  fund={pos.funding_pnl:>+7.2f}"
            )
        if not self._positions:
            lines.append("  (no open positions)")
        return "\n".join(lines)

    def _equity(self, prices: dict[str, float]) -> float:
        unrealized = sum(
            pos.unrealized_pnl(prices.get(s, pos.entry_price)) + pos.funding_pnl
            for s, pos in self._positions.items()
        )
        margin_total = sum(self._margin.values())
        return self._cash + margin_total + unrealized
