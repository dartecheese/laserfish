"""Funding Rate Mean-Reversion Arbitrage Strategy.

Based on two papers:
  [1] He, Manela, Ross & von Wachter — "Fundamentals of Perpetual Futures" (2022)
  [2] "Exploring Risk and Return Profiles of Funding Rate Arbitrage on CEX and DEX"
      (ScienceDirect, 2024) — 60-scenario empirical study across Binance, Bitmex,
      ApolloX, and Drift for BTC/ETH/XRP/BNB/SOL.

Core signal from [1]:
  - No-arbitrage perp price: F_t = [κ/(κ-(r-r'))] × S_t ≈ S_t
  - Basis (mark - index) mean-reverts; crypto deviations are 60-90% annualized
  - Funding rate IS the basis signal (positive funding = F > S = overpriced perp)
  - Momentum explains >50% of basis gap (R²); use as conviction amplifier
  - Implied arbitrage earns Sharpe 1.8-3.5 across crypto

Three calibration improvements from [2]:
  1. 20bps minimum basis threshold — below this, only 40% of opportunities are
     profitable after transaction costs (from their 60-scenario analysis).
  2. Per-asset minimum funding thresholds — alt-coins (XRP, BNB, DOGE) exhibit
     systematically higher funding rates than BTC/ETH and need lower bars to enter.
  3. Consecutive-readings filter — require the basis to stay elevated for N periods,
     reducing entries on transient spikes that reverse before costs are recovered.

Signal construction:
  1. Load 90-period rolling window of 8-hour funding rates (~30 days)
  2. z_score = (current_rate - rolling_mean) / rolling_std
  3. Apply consecutive-readings gate: skip unless basis exceeded min_basis_pct
     for `require_consecutive` periods in a row
  4. Momentum amplifier: 1d return direction × sign(z_score) → add up to 0.3
  5. alpha = clip(z_score / Z_ENTRY + momentum_bonus, -1, 1)
  6. Side: alpha > 0 → SHORT; alpha < 0 → LONG
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from trader.exchanges.base import BracketParams, FundingData, Order
from trader.exchanges.hyperliquid import SPOT_HEDGEABLE, HyperliquidExchange

logger = logging.getLogger(__name__)


@dataclass
class HedgedPosition:
    """Tracks a delta-neutral position: short perp + long spot."""
    symbol: str
    perp_qty: float
    spot_qty: float
    hedged: bool          # True if spot leg was successfully opened


# Per-asset minimum annualized funding thresholds.
# Require meaningful carry before entering — at <8% annualized the funding
# payment (~0.006% per 8h) does not cover spread + slippage after ~3 periods.
# Alts get slightly lower floors because they exhibit higher rate volatility,
# so even a 6% annualized reading represents a genuine regime shift.
_DEFAULT_PER_ASSET_MIN_FUNDING: dict[str, float] = {
    "BTC":  0.08,   # 8% ann — major; tight spread so need real carry
    "ETH":  0.08,
    "BNB":  0.06,
    "XRP":  0.06,
    "SOL":  0.07,
    "DOGE": 0.06,
    "KPEPE": 0.06,
    "WIF":  0.06,
    "HYPE": 0.06,
}
# Assets not in the dict fall back to FundingArbConfig.min_funding_annualized.


@dataclass
class FundingArbConfig:
    symbols: list[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "AVAX", "LINK", "ARB", "OP", "DOGE",
        "BNB", "XRP", "NEAR", "WIF", "KPEPE", "SUI", "HYPE", "INJ",
    ])
    z_entry: float = 1.5            # z-score to open a position
    z_exit: float = 0.2             # z-score to close (normalize)
    history_window: int = 90        # 8h periods ≈ 30 days
    min_funding_annualized: float = 0.08   # default fallback — 8% ann covers costs in ~5 periods
    min_alpha: float = 0.25
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.025
    max_leverage: int = 3
    momentum_window: int = 6        # candles for 1d momentum (6 × 4h = 24h)
    momentum_bonus: float = 0.30

    # ── ScienceDirect 2024 improvements ──────────────────────────────── #
    min_basis_pct: float = 0.20
    # Minimum |basis_pct| to consider entry. Below 20bps only 40% of top
    # opportunities are profitable after transaction costs (60-scenario study).

    per_asset_min_funding: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_PER_ASSET_MIN_FUNDING)
    )
    # Override default min_funding_annualized per symbol.

    require_consecutive: int = 2
    # Basis must exceed min_basis_pct for this many consecutive 8h periods
    # before entry. Filters transient spikes that reverse before costs recover.

    hedge_spot: bool = True
    # When True, open a matching long spot position alongside each short perp
    # to eliminate directional exposure. Only applies to SPOT_HEDGEABLE symbols.
    # Unhedgeable symbols (ARB, OP, DOGE, etc.) are skipped when hedge_spot=True
    # unless they meet a higher conviction bar (alpha > 0.7).


class Signal(NamedTuple):
    symbol: str
    side: str               # "buy" | "sell"
    alpha: float            # 0-1 conviction
    z_score: float
    funding_rate_8h: float
    funding_annualized: float
    basis_pct: float
    mark_price: float
    momentum_1d: float      # 24h price return


class FundingArbStrategy:
    """Scans Hyperliquid perps and produces funding-arb signals."""

    def __init__(
        self,
        exchange: HyperliquidExchange,
        cfg: FundingArbConfig | None = None,
    ):
        self.ex = exchange
        self.cfg = cfg or FundingArbConfig()
        self._history: dict[str, deque[float]] = {
            s: deque(maxlen=self.cfg.history_window)
            for s in self.cfg.symbols
        }
        self._initialized: set[str] = set()
        # Consecutive-readings counter: tracks how many periods in a row
        # |basis_pct| has exceeded min_basis_pct (ScienceDirect 2024 filter).
        self._consecutive_above: dict[str, int] = defaultdict(int)
        # Tracks open spot hedge legs: symbol → HedgedPosition
        self._hedged: dict[str, HedgedPosition] = {}

    # ------------------------------------------------------------------ #
    # Initialization                                                       #
    # ------------------------------------------------------------------ #

    def warm_up(self) -> None:
        """Fetch funding rate history for all symbols to seed rolling windows."""
        logger.info("Warming up funding rate history (%d symbols)…", len(self.cfg.symbols))
        for sym in self.cfg.symbols:
            try:
                history = self.ex.get_funding_rate_history(sym, limit=self.cfg.history_window)
                for _, rate in history:
                    self._history[sym].append(rate)
                self._initialized.add(sym)
                logger.debug("%s: loaded %d history points", sym, len(self._history[sym]))
            except Exception as exc:
                logger.warning("Could not load history for %s: %s", sym, exc)

    # ------------------------------------------------------------------ #
    # Signal generation                                                    #
    # ------------------------------------------------------------------ #

    def _z_score(self, sym: str, current_rate: float) -> float | None:
        """Z-score of current_rate vs rolling window. None if insufficient data."""
        hist = list(self._history[sym])
        if len(hist) < 10:
            return None
        mu = float(np.mean(hist))
        sigma = float(np.std(hist))
        if sigma < 1e-9:
            return 0.0
        return (current_rate - mu) / sigma

    def _momentum_1d(self, sym: str) -> float:
        """24h price return from 4h candles (6 periods). 0.0 on failure."""
        try:
            candles = self.ex.get_candles(sym, "4h", self.cfg.momentum_window + 1)
            if len(candles) < 2:
                return 0.0
            return (candles[-1].close - candles[0].close) / candles[0].close
        except Exception:
            return 0.0

    def _asset_min_funding(self, sym: str) -> float:
        """Per-asset minimum annualized funding threshold (ScienceDirect 2024)."""
        return self.cfg.per_asset_min_funding.get(sym, self.cfg.min_funding_annualized)

    def scan(self) -> list[Signal]:
        """
        Fetch current funding data for all symbols and return ranked signals.

        Filters applied (in order):
          1. Per-asset minimum annualized funding (ScienceDirect 2024: alts lower)
          2. 20bps minimum |basis_pct| (ScienceDirect 2024: below this only 40% profitable)
          3. Consecutive-readings gate: basis must be elevated N periods in a row
          4. Z-score entry threshold
          5. Momentum conviction bonus (He et al.: momentum explains >50% of basis gap)

        Returns signals sorted descending by |alpha|.
        """
        fd_list = self.ex.get_all_funding_data(self.cfg.symbols)
        signals: list[Signal] = []

        for fd in fd_list:
            sym = fd.symbol
            rate = fd.funding_rate_8h

            self._history[sym].append(rate)
            if sym not in self._initialized:
                self._initialized.add(sym)

            # ── Filter 1: per-asset minimum funding magnitude ─────────────
            if abs(fd.funding_rate_annualized) < self._asset_min_funding(sym):
                self._consecutive_above[sym] = 0
                continue

            # ── Filter 2: 20bps minimum basis (ScienceDirect 2024) ────────
            if abs(fd.basis_pct) < self.cfg.min_basis_pct:
                self._consecutive_above[sym] = 0
                continue

            # ── Filter 3: consecutive-readings gate ───────────────────────
            self._consecutive_above[sym] += 1
            if self._consecutive_above[sym] < self.cfg.require_consecutive:
                continue

            # ── Filter 4: z-score entry threshold ────────────────────────
            z = self._z_score(sym, rate)
            if z is None or abs(z) < self.cfg.z_entry:
                continue

            # ── Conviction: momentum amplifier (He et al.) ────────────────
            mom = self._momentum_1d(sym)
            alignment = 1.0 if (z > 0 and mom > 0) or (z < 0 and mom < 0) else 0.0
            mom_bonus = alignment * self.cfg.momentum_bonus * min(abs(mom) / 0.05, 1.0)

            raw_alpha = abs(z) / self.cfg.z_entry
            alpha = float(np.clip(raw_alpha + mom_bonus, 0.0, 1.0))

            if alpha < self.cfg.min_alpha:
                continue

            # ── Hedge filter: skip unhedgeable unless very high conviction ──
            # Without a spot leg the trade is directional, not arb.
            # Allow unhedgeable symbols through only at alpha > 0.7.
            if self.cfg.hedge_spot and sym not in SPOT_HEDGEABLE and alpha <= 0.7:
                continue

            side = "sell" if z > 0 else "buy"

            signals.append(Signal(
                symbol=sym,
                side=side,
                alpha=alpha,
                z_score=z,
                funding_rate_8h=rate,
                funding_annualized=fd.funding_rate_annualized,
                basis_pct=fd.basis_pct,
                mark_price=fd.mark_price,
                momentum_1d=mom,
            ))

        signals.sort(key=lambda s: s.alpha, reverse=True)
        return signals

    # ------------------------------------------------------------------ #
    # Exit check                                                           #
    # ------------------------------------------------------------------ #

    def should_exit(self, sym: str) -> bool:
        """True if z-score has returned to neutral — time to close the position."""
        try:
            fd = self.ex.get_funding_data(sym)
        except Exception:
            return False
        self._history[sym].append(fd.funding_rate_8h)
        z = self._z_score(sym, fd.funding_rate_8h)
        if z is None:
            return False
        return abs(z) <= self.cfg.z_exit

    # ------------------------------------------------------------------ #
    # Hedged position management                                           #
    # ------------------------------------------------------------------ #

    def open_hedged_position(
        self,
        sig: Signal,
        qty: float,
    ) -> tuple[Order, Order | None]:
        """
        Open a delta-neutral position:
          1. Short perp (collect funding when rate is positive)
          2. Long spot of equal notional (offsets directional exposure)

        Returns (perp_order, spot_order). spot_order is None if the symbol
        has no spot market or hedge_spot is disabled.
        """
        perp_order = self.ex.place_bracket_order(BracketParams(
            symbol=sig.symbol,
            side=sig.side,
            quantity=qty,
            price=None,
            stop_loss_pct=self.cfg.stop_loss_pct,
            take_profit_pct=self.cfg.take_profit_pct,
            leverage=self.cfg.max_leverage,
        ))

        spot_order: Order | None = None
        hedged = False

        if self.cfg.hedge_spot and sig.symbol in SPOT_HEDGEABLE:
            try:
                # Spot hedge: always long spot (offsets short perp when funding > 0,
                # and offsets long perp when funding < 0).
                spot_side = "buy" if sig.side == "sell" else "sell"
                spot_order = self.ex.place_spot_order(sig.symbol, spot_side, qty)
                hedged = True
                logger.info(
                    "  SPOT HEDGE %s %s  qty=%.4f @ %.4f",
                    spot_side.upper(), sig.symbol, qty, spot_order.price,
                )
            except Exception as exc:
                logger.warning("Spot hedge failed for %s: %s — running unhedged", sig.symbol, exc)

        self._hedged[sig.symbol] = HedgedPosition(
            symbol=sig.symbol,
            perp_qty=qty,
            spot_qty=qty if hedged else 0.0,
            hedged=hedged,
        )
        return perp_order, spot_order

    def close_hedged_position(self, symbol: str) -> None:
        """Close both perp and spot legs for `symbol`."""
        self.ex.close_position(symbol)

        hp = self._hedged.pop(symbol, None)
        if hp and hp.hedged and hp.spot_qty > 0:
            try:
                self.ex.close_spot_position(symbol)
                logger.info("  SPOT CLOSE %s  qty=%.4f", symbol, hp.spot_qty)
            except Exception as exc:
                logger.warning("Failed to close spot leg for %s: %s", symbol, exc)

    def is_hedged(self, symbol: str) -> bool:
        """True if the open position for `symbol` has an active spot hedge."""
        hp = self._hedged.get(symbol)
        return hp is not None and hp.hedged

    # ------------------------------------------------------------------ #
    # Formatted output                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_signal(s: Signal) -> str:
        direction = "SHORT" if s.side == "sell" else "LONG"
        return (
            f"{s.symbol:6s} {direction:5s}  "
            f"α={s.alpha:.2f}  z={s.z_score:+.2f}  "
            f"fund_8h={s.funding_rate_8h*100:+.4f}%  "
            f"ann={s.funding_annualized*100:+.1f}%  "
            f"basis={s.basis_pct:+.3f}%  "
            f"mom1d={s.momentum_1d*100:+.2f}%  "
            f"mark={s.mark_price:.4f}"
        )
