"""Abstract exchange interface.

All adapters expose the same six methods. The agent only imports this type.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Candle:
    timestamp: int   # ms
    open: float
    high: float
    low: float
    close: float
    volume: float    # base-asset (e.g. BTC)
    quote_volume: float


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str        # "buy" | "sell"
    quantity: float
    price: float
    status: str      # "open" | "filled" | "cancelled"


@dataclass
class Position:
    symbol: str
    side: str        # "long" | "short"
    quantity: float
    entry_price: float
    unrealized_pnl: float
    leverage: float


@dataclass
class Balance:
    total_usd: float
    available_usd: float


@dataclass
class BracketParams:
    """Parameters for a bracket (entry + TP + SL) order."""
    symbol: str
    side: str            # "buy" (long) | "sell" (short)
    quantity: float
    price: float | None  # None = market order
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    leverage: int = 1


@dataclass
class FundingData:
    symbol: str
    funding_rate_8h: float          # raw fraction (e.g. 0.0001 = 0.01%)
    funding_rate_annualized: float  # annualized fraction (rate_8h * 1095)
    mark_price: float
    index_price: float
    basis_pct: float                # (mark - index) / index * 100


class Exchange(ABC):
    """Abstract DeFi/CEX exchange adapter."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        """Fetch recent OHLCV candles. symbol in exchange-native format."""
        ...

    @abstractmethod
    def get_balance(self) -> Balance: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def place_bracket_order(self, params: BracketParams) -> Order:
        """Open a position with optional take-profit and stop-loss orders."""
        ...

    @abstractmethod
    def close_position(self, symbol: str) -> None:
        """Close the entire position for `symbol` at market."""
        ...

    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> None: ...
