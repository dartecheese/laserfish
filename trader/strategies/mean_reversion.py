"""Mean-reversion + funding carry strategy for ranging/bear markets.

Generates alpha when cross-sectional momentum is absent by exploiting two signals:

1. OVERSOLD BOUNCE (mean reversion)
   - Enter LONG when 7d z-score < -MR_Z_ENTRY (deeply oversold)
   - Tight TP 2% / SL 1.5% — captures the snapback, not trend continuation
   - Only if funding is not deeply negative (avoid catching falling knives)

2. FUNDING CARRY (positive carry harvest)
   - Enter LONG when funding_rate_8h < -CARRY_THRESHOLD (you earn rate by being long)
   - Enter SHORT when funding_rate_8h > +CARRY_THRESHOLD (you earn rate by being short)
   - Direction confirmed by near-zero or same-direction momentum z-score
   - These are held until funding normalizes (funding z-score returns to 0)

Both signals are ORTHOGONAL to momentum — they fire when momentum z is low,
providing alpha during choppy/ranging/bear market regimes (regime 1 or 2).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from trader.exchanges.base import FundingData

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionConfig:
    symbols: list[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "AVAX", "LINK", "BNB", "XRP", "DOGE",
        "NEAR", "WIF", "KPEPE", "SUI", "HYPE", "INJ",
    ])

    # Lookback (same as momentum for reuse)
    momentum_window: int = 2016    # 7d in 5m bars
    vol_window: int = 8640         # 30d
    funding_window: int = 21       # 7 days of 8h rates

    # Mean reversion signal
    mr_z_entry: float = 2.0        # z-score for oversold/overbought entry
    mr_z_exit: float = 0.5         # exit when z-score reverts toward 0

    # Carry signal
    carry_threshold: float = 0.0003   # 0.03% per 8h = 0.33%/day — meaningful carry
    carry_z_veto: float = 2.5         # don't fade if funding z-score is extreme in same direction

    # Risk — tighter than momentum (bounce trades, not trend)
    stop_loss_pct: float = 0.015      # 1.5%
    take_profit_pct: float = 0.025    # 2.5%

    # Max simultaneous MR positions (separate from momentum positions)
    max_positions: int = 2


class MRSignal(NamedTuple):
    symbol: str
    side: str               # "buy" | "sell"
    signal_type: str        # "mean_reversion" | "carry"
    alpha: float            # conviction 0-1
    z_score: float
    funding_rate: float
    funding_z: float
    mark_price: float


class MeanReversionStrategy:
    """
    Mean-reversion + funding carry — fires in low-momentum environments.

    Designed to run in parallel with MomentumStrategy. Check sig.signal_type
    to distinguish signal sources when sizing and logging.
    """

    def __init__(self, exchange, cfg: MeanReversionConfig | None = None):
        self.ex = exchange
        self.cfg = cfg or MeanReversionConfig()
        self._price_hist: dict[str, deque[float]] = {
            s: deque(maxlen=self.cfg.vol_window + self.cfg.momentum_window + 2)
            for s in self.cfg.symbols
        }
        self._funding_hist: dict[str, deque[float]] = {
            s: deque(maxlen=self.cfg.funding_window)
            for s in self.cfg.symbols
        }

    def warm_up(self) -> None:
        """Reuse price/funding history from exchange (same calls as momentum warm_up)."""
        needed = self.cfg.vol_window + self.cfg.momentum_window + 2
        batch = 500
        for sym in self.cfg.symbols:
            try:
                all_closes: list[float] = []
                fetched = 0
                while fetched < needed:
                    candles = self.ex.get_candles(sym, "5m", min(batch, needed - fetched))
                    if not candles:
                        break
                    all_closes = [c.close for c in candles] + all_closes
                    fetched += len(candles)
                    if len(candles) < batch:
                        break
                for close in all_closes[-(needed):]:
                    self._price_hist[sym].append(close)
            except Exception as e:
                logger.debug("MR warm_up prices %s: %s", sym, e)

            try:
                hist = self.ex.get_funding_rate_history(sym, limit=self.cfg.funding_window)
                for _, rate in hist:
                    self._funding_hist[sym].append(rate)
            except Exception as e:
                logger.debug("MR warm_up funding %s: %s", sym, e)

    def _momentum_z(self, sym: str) -> float | None:
        prices = list(self._price_hist[sym])
        if len(prices) < self.cfg.vol_window + 1:
            return None
        mom_bars = min(self.cfg.momentum_window, len(prices) - 1)
        ret = (prices[-1] - prices[-mom_bars - 1]) / prices[-mom_bars - 1]
        rolling = [
            (prices[i] - prices[max(0, i - mom_bars)]) / prices[max(0, i - mom_bars)]
            for i in range(mom_bars, min(len(prices), self.cfg.vol_window + 1))
        ]
        if len(rolling) < 5:
            return None
        return (ret - float(np.mean(rolling))) / (float(np.std(rolling)) + 1e-9)

    def _funding_z(self, sym: str) -> float | None:
        hist = list(self._funding_hist[sym])
        if len(hist) < 5:
            return None
        mu, sd = float(np.mean(hist)), float(np.std(hist)) + 1e-9
        return (hist[-1] - mu) / sd

    def scan(self) -> list[MRSignal]:
        """Fetch latest data and return mean-reversion + carry signals."""
        signals: list[MRSignal] = []

        for sym in self.cfg.symbols:
            try:
                fd: FundingData = self.ex.get_funding_data(sym)
                self._price_hist[sym].append(fd.mark_price)
                self._funding_hist[sym].append(fd.funding_rate_8h)
            except Exception as e:
                logger.debug("MR scan skip %s: %s", sym, e)
                continue

            z = self._momentum_z(sym)
            fz = self._funding_z(sym)
            if z is None:
                continue

            # ── Mean reversion signal ──────────────────────────────────
            # Deeply oversold → expect bounce up
            if z < -self.cfg.mr_z_entry:
                # Don't buy into a crash with extreme negative funding
                if fz is not None and fz < -self.cfg.carry_z_veto:
                    logger.debug("%s: MR long vetoed — extreme negative funding (falling knife)", sym)
                else:
                    alpha = float(np.clip(abs(z) / self.cfg.mr_z_entry - 1.0, 0.0, 1.0))
                    signals.append(MRSignal(
                        symbol=sym, side="buy", signal_type="mean_reversion",
                        alpha=alpha, z_score=z,
                        funding_rate=fd.funding_rate_8h,
                        funding_z=fz or 0.0, mark_price=fd.mark_price,
                    ))

            # Deeply overbought → expect fade
            elif z > self.cfg.mr_z_entry:
                if fz is not None and fz > self.cfg.carry_z_veto:
                    logger.debug("%s: MR short vetoed — extreme positive funding", sym)
                else:
                    alpha = float(np.clip(abs(z) / self.cfg.mr_z_entry - 1.0, 0.0, 1.0))
                    signals.append(MRSignal(
                        symbol=sym, side="sell", signal_type="mean_reversion",
                        alpha=alpha, z_score=z,
                        funding_rate=fd.funding_rate_8h,
                        funding_z=fz or 0.0, mark_price=fd.mark_price,
                    ))

            # ── Funding carry signal ───────────────────────────────────
            # Negative funding → longs earn carry; only enter if not in a downtrend
            if fd.funding_rate_8h < -self.cfg.carry_threshold and z > -1.0:
                carry_alpha = float(np.clip(abs(fd.funding_rate_8h) / self.cfg.carry_threshold - 1.0, 0.0, 1.0))
                signals.append(MRSignal(
                    symbol=sym, side="buy", signal_type="carry",
                    alpha=carry_alpha * 0.7,  # carry is lower conviction than MR
                    z_score=z,
                    funding_rate=fd.funding_rate_8h,
                    funding_z=fz or 0.0, mark_price=fd.mark_price,
                ))

            # Positive funding → shorts earn carry; only enter if not in an uptrend
            elif fd.funding_rate_8h > self.cfg.carry_threshold and z < 1.0:
                carry_alpha = float(np.clip(fd.funding_rate_8h / self.cfg.carry_threshold - 1.0, 0.0, 1.0))
                signals.append(MRSignal(
                    symbol=sym, side="sell", signal_type="carry",
                    alpha=carry_alpha * 0.7,
                    z_score=z,
                    funding_rate=fd.funding_rate_8h,
                    funding_z=fz or 0.0, mark_price=fd.mark_price,
                ))

        # Sort by alpha descending, dedupe by symbol (keep highest alpha per symbol)
        seen: set[str] = set()
        deduped: list[MRSignal] = []
        for sig in sorted(signals, key=lambda s: s.alpha, reverse=True):
            if sig.symbol not in seen:
                seen.add(sig.symbol)
                deduped.append(sig)

        return deduped[:self.cfg.max_positions]

    def should_exit(self, sym: str, entry_side: str, signal_type: str) -> bool:
        """Exit mean-reversion when z reverts; exit carry when funding normalizes."""
        z = self._momentum_z(sym)
        hist = list(self._funding_hist[sym])
        if z is None:
            return False

        if signal_type == "mean_reversion":
            # Exit when z reverts back past mr_z_exit toward zero
            if entry_side == "buy"  and z > -self.cfg.mr_z_exit:
                return True
            if entry_side == "sell" and z <  self.cfg.mr_z_exit:
                return True

        elif signal_type == "carry":
            # Exit when funding normalizes
            if len(hist) >= 3:
                current_rate = hist[-1]
                if entry_side == "buy"  and current_rate > -self.cfg.carry_threshold * 0.5:
                    return True
                if entry_side == "sell" and current_rate <  self.cfg.carry_threshold * 0.5:
                    return True

        return False

    @staticmethod
    def format_signal(s: MRSignal) -> str:
        direction = "LONG " if s.side == "buy" else "SHORT"
        return (
            f"{s.symbol:6s} {direction}  [{s.signal_type.upper():<15}]  "
            f"α={s.alpha:.2f}  z={s.z_score:+.2f}  "
            f"fund={s.funding_rate*100:+.4f}%  fund_z={s.funding_z:+.2f}  "
            f"mark={s.mark_price:.4f}"
        )
