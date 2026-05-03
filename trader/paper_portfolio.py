"""In-memory paper trading portfolio.

Tracks simulated positions, margin, unrealized PnL, funding payments,
trading fees, and slippage so paper-mode results approximate live behavior.

Cost model (defaults match Hyperliquid 2026):
  - Taker fee: 0.045% (4.5 bps) per fill — applied on both open and close.
  - Slippage: per-symbol bps loss vs mark price on entry and exit.
    Majors  (BTC/ETH/SOL) ........ 3 bps
    Mid-cap (BNB/AVAX/LINK/DOGE/XRP) 7 bps
    Alts    (everything else) ...... 15 bps
  - Funding: applied via apply_funding(); shorts receive when rate>0.

To disable any cost, set fee_bps / slippage_bps to 0 at construction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from trader.exchanges.base import Balance, Order, Position


# Per-symbol slippage in basis points (1 bp = 0.01%).
# Reflects approx Hyperliquid 1m-window mid-to-fill cost for $5–10K orders.
_DEFAULT_SLIPPAGE_BPS: dict[str, float] = {
    "BTC": 3, "ETH": 3, "SOL": 3,
    "BNB": 7, "AVAX": 7, "LINK": 7, "DOGE": 7, "XRP": 7,
    "ARB": 7, "OP": 7, "MATIC": 7, "LTC": 7, "ADA": 7, "DOT": 7,
}
_FALLBACK_SLIPPAGE_BPS = 15.0   # alts / illiquid


@dataclass
class _PaperPos:
    symbol: str
    side: str           # "long" | "short"
    qty: float          # positive base units
    entry_price: float
    leverage: int
    funding_pnl: float = 0.0
    fees_paid: float = 0.0   # cumulative taker fee paid on this position

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
    Simulates a perpetuals trading account with realistic costs.

    Default cost model approximates live Hyperliquid execution:
      taker fee 4.5 bps + per-symbol slippage (3-15 bps).
    Set fee_bps=0 / slippage_bps_override=0 to disable.
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        fee_bps: float = 4.5,
        slippage_bps: dict[str, float] | None = None,
        slippage_default_bps: float = _FALLBACK_SLIPPAGE_BPS,
    ) -> None:
        self.initial_equity = initial_equity
        self._cash = initial_equity
        self._margin: dict[str, float] = {}
        self._positions: dict[str, _PaperPos] = {}
        self._realized_pnl = 0.0
        self._funding_total = 0.0
        self._fees_total = 0.0
        self._slippage_total = 0.0
        self._trades: list[dict] = []

        self._fee_rate = fee_bps / 10_000.0
        self._slip_table = dict(_DEFAULT_SLIPPAGE_BPS)
        if slippage_bps:
            self._slip_table.update(slippage_bps)
        self._slip_default = slippage_default_bps

    # ------------------------------------------------------------------ #
    # Cost helpers                                                        #
    # ------------------------------------------------------------------ #

    def _slip_rate(self, symbol: str) -> float:
        return self._slip_table.get(symbol, self._slip_default) / 10_000.0

    def _fill_price(self, symbol: str, side: str, mid_price: float) -> float:
        """Mid price adjusted for slippage in adverse direction."""
        slip = self._slip_rate(symbol)
        if side == "buy" or side == "long":
            return mid_price * (1.0 + slip)
        return mid_price * (1.0 - slip)

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
        # Normalize side: strategies pass "buy"/"sell" or "long"/"short"
        long_side = side in ("buy", "long")
        position_side = "long" if long_side else "short"

        fill_price = self._fill_price(symbol, side, price)
        notional   = qty * fill_price
        fee        = notional * self._fee_rate
        slip_cost  = qty * abs(fill_price - price)
        margin     = notional / max(leverage, 1)

        # Scale down if margin + fee exceed free cash
        max_outlay = self._cash - fee
        if margin > max_outlay and max_outlay > 0:
            qty       = max_outlay * leverage / fill_price
            notional  = qty * fill_price
            fee       = notional * self._fee_rate
            slip_cost = qty * abs(fill_price - price)
            margin    = notional / max(leverage, 1)

        if qty <= 0:
            return Order(f"paper-skip-{int(time.time()*1000)}", symbol, side, 0.0, price, "cancelled")

        if symbol in self._positions:
            self.close(symbol, price)

        self._cash           -= margin + fee
        self._fees_total     += fee
        self._slippage_total += slip_cost
        self._margin[symbol]  = margin
        self._positions[symbol] = _PaperPos(
            symbol=symbol, side=position_side, qty=qty,
            entry_price=fill_price, leverage=leverage,
            fees_paid=fee,
        )
        self._trades.append({
            "action": "open", "symbol": symbol, "side": position_side,
            "qty": qty, "price": fill_price, "fee": fee,
            "slippage": slip_cost, "ts": time.time(),
        })
        return Order(
            order_id=f"paper-{int(time.time()*1000)}",
            symbol=symbol, side=side, quantity=qty,
            price=fill_price, status="filled",
        )

    def close(self, symbol: str, current_price: float) -> float:
        pos    = self._positions.pop(symbol, None)
        margin = self._margin.pop(symbol, 0.0)
        if pos is None:
            return 0.0

        # Exit slippage: pay spread on the opposite side of entry
        exit_side = "sell" if pos.side == "long" else "buy"
        fill_price = self._fill_price(symbol, exit_side, current_price)
        slip_cost  = pos.qty * abs(fill_price - current_price)
        notional   = pos.qty * fill_price
        fee        = notional * self._fee_rate

        gross_pnl = pos.signed_qty * (fill_price - pos.entry_price)
        pnl       = gross_pnl + pos.funding_pnl - fee

        self._cash           += margin + pnl
        self._realized_pnl   += pnl
        self._fees_total     += fee
        self._slippage_total += slip_cost
        self._trades.append({
            "action": "close", "symbol": symbol,
            "pnl": pnl, "fee": fee, "slippage": slip_cost,
            "price": fill_price, "ts": time.time(),
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
            f"  Fees     ${self._fees_total:>10,.2f}   Slippage ${self._slippage_total:>+,.2f}",
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
