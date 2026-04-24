"""Cross-sectional momentum strategy with regime gating — 5m perp edition.

Signal construction follows Liu & Tsyvinski (2021) "Risks and Returns of
Cryptocurrency" (Review of Financial Studies):
  - 7-day momentum (2016 × 5m bars) as primary signal — the horizon with
    documented positive Sharpe in crypto. Shorter windows show reversal.
  - 30-day volatility window (8640 × 5m bars) for normalization
  - Funding rate confirms/contra-indicates directional trades
  - 5m candles used for execution precision, not signal generation

Strategy:
  1. Rank all symbols by their 7d return z-score vs 30d vol
  2. Long top tercile, short bottom tercile (cross-sectional long-short)
  3. Gate: only enter when |momentum| > volatility-adjusted threshold
     AND funding rate direction is consistent with trade direction
  4. Size: proportional to momentum z-score, capped at max_leverage
  5. Exit: when momentum z-score reverses or TP/SL hit

Regime gating (two conditions must hold to enter):
  - Momentum gate: 7d return z-score exceeds entry threshold
  - Funding gate: for longs, funding not anomalously negative (longs get charged);
                  for shorts, funding not anomalously positive (shorts receive)
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
class MomentumConfig:
    symbols: list[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "AVAX", "LINK", "BNB", "XRP", "DOGE",
        "NEAR", "WIF", "KPEPE", "SUI", "HYPE", "INJ",
    ])

    # Lookback windows (in 5m bars) — calibrated from backtest sweep
    momentum_window: int = 2016   # 2016 × 5m = 7d momentum (Liu & Tsyvinski signal)
    vol_window: int = 8640        # 8640 × 5m = 30d realized vol for z-score normalization
    funding_window: int = 21      # 21 funding readings (~7 days) for z-score

    # Entry / exit
    z_entry: float = 1.2          # momentum z-score to enter
    z_exit: float = 0.3           # momentum z-score to exit
    funding_z_veto: float = 1.5   # veto longs when funding z-score < -1.5 (longs charged)

    # Sizing
    max_leverage: float = 2.0
    top_n: int = 3                # number of long and short positions simultaneously
    min_alpha: float = 0.20       # minimum |z_score / z_entry| to size

    # Risk — from backtest: 3% SL / 5% TP optimal at 7d signal horizon
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.05


class MomentumSignal(NamedTuple):
    symbol: str
    side: str               # "buy" | "sell"
    alpha: float            # 0-1 conviction
    momentum_z: float       # z-score of 24h return vs vol window
    return_24h: float       # raw 24h return
    funding_rate: float     # latest 8h funding rate
    funding_z: float        # funding rate z-score
    mark_price: float
    vol_1w: float           # 7-day realized volatility


class MomentumStrategy:
    """
    Cross-sectional momentum with regime gates.

    At each scan, ranks all symbols by their 24h momentum z-score and
    generates long/short signals for the top/bottom N symbols when the
    momentum AND funding gates both pass.
    """

    def __init__(self, exchange, cfg: MomentumConfig | None = None):
        self.ex = exchange
        self.cfg = cfg or MomentumConfig()

        # Price history per symbol: deque of 4h close prices
        self._price_hist: dict[str, deque[float]] = {
            s: deque(maxlen=max(self.cfg.momentum_window + 1, self.cfg.vol_window + 1))
            for s in self.cfg.symbols
        }
        # Funding rate history per symbol
        self._funding_hist: dict[str, deque[float]] = {
            s: deque(maxlen=self.cfg.funding_window)
            for s in self.cfg.symbols
        }

    # ------------------------------------------------------------------ #
    # Warm-up                                                              #
    # ------------------------------------------------------------------ #

    def warm_up(self) -> None:
        """Populate price and funding histories from exchange.

        Hyperliquid caps fetch_ohlcv at 500 bars per call, so we paginate
        to collect the full vol_window + momentum_window history needed for
        the 7d/30d signal (vol_window=8640 bars ≈ 30 days of 5m data).
        """
        needed = self.cfg.vol_window + self.cfg.momentum_window + 2
        batch = 500   # Hyperliquid CCXT per-call limit
        logger.info("Warming up momentum strategy (%d symbols, %d bars needed)…",
                    len(self.cfg.symbols), needed)
        for sym in self.cfg.symbols:
            try:
                # Paginate backwards to collect enough history
                all_closes: list[float] = []
                fetched = 0
                while fetched < needed:
                    candles = self.ex.get_candles(sym, "5m", min(batch, needed - fetched))
                    if not candles:
                        break
                    all_closes = [c.close for c in candles] + all_closes
                    fetched += len(candles)
                    if len(candles) < batch:
                        break  # exchange returned less than requested — no more history
                for close in all_closes[-(needed):]:
                    self._price_hist[sym].append(close)
                logger.debug("%s: loaded %d bars", sym, len(self._price_hist[sym]))
            except Exception as e:
                logger.warning("Could not warm up prices for %s: %s", sym, e)

            try:
                hist = self.ex.get_funding_rate_history(sym, limit=self.cfg.funding_window)
                for _, rate in hist:
                    self._funding_hist[sym].append(rate)
            except Exception as e:
                logger.warning("Could not warm up funding for %s: %s", sym, e)

    # ------------------------------------------------------------------ #
    # Signal generation                                                    #
    # ------------------------------------------------------------------ #

    def _momentum_z(self, sym: str) -> tuple[float, float, float] | None:
        """Returns (z_score, ret_7d, vol_30d) or None if insufficient data."""
        prices = list(self._price_hist[sym])
        if len(prices) < self.cfg.vol_window + 1:
            return None

        # 7-day momentum = return over last 2016 × 5m bars
        mom_bars = min(self.cfg.momentum_window, len(prices) - 1)
        ret_7d = (prices[-1] - prices[-mom_bars - 1]) / prices[-mom_bars - 1]

        # 30-day realized volatility (std of 5m log returns), annualized
        log_rets = np.diff(np.log(prices[-self.cfg.vol_window:]))
        vol = float(np.std(log_rets)) * np.sqrt(288 * 365)  # 288 × 5m bars per day

        # Vol-adjusted z-score vs rolling distribution of 7d returns
        n = min(len(prices) - 1, self.cfg.vol_window)
        rolling_rets = [
            (prices[i] - prices[max(0, i - mom_bars)]) / prices[max(0, i - mom_bars)]
            for i in range(mom_bars, n + 1)
        ]
        if len(rolling_rets) < 5:
            return None
        mu = float(np.mean(rolling_rets))
        sd = float(np.std(rolling_rets)) + 1e-9
        z = (ret_7d - mu) / sd

        return z, ret_7d, vol

    def _funding_z(self, sym: str) -> float | None:
        """Z-score of current funding rate vs history."""
        hist = list(self._funding_hist[sym])
        if len(hist) < 5:
            return None
        mu = float(np.mean(hist))
        sd = float(np.std(hist)) + 1e-9
        return (hist[-1] - mu) / sd if hist else None

    def scan(self) -> list[MomentumSignal]:
        """
        Fetch latest prices + funding, update history, and return
        ranked momentum signals (best to worst by |alpha|).
        """
        candidates: list[tuple[float, MomentumSignal]] = []

        for sym in self.cfg.symbols:
            try:
                fd: FundingData = self.ex.get_funding_data(sym)
                self._price_hist[sym].append(fd.mark_price)
                self._funding_hist[sym].append(fd.funding_rate_8h)
            except Exception as e:
                logger.debug("Skipping %s: %s", sym, e)
                continue

            result = self._momentum_z(sym)
            if result is None:
                continue
            z, ret_7d, vol = result

            fz = self._funding_z(sym)

            # ── Momentum gate ─────────────────────────────────────────
            if abs(z) < self.cfg.z_entry:
                continue

            # ── Funding gate ─────────────────────────────────────────
            # Veto longs when funding is deeply negative (longs get charged excessively).
            # Veto shorts when funding is deeply positive (confirms bearish sentiment
            # already priced in; mean-reversion more likely than continuation).
            if z > 0 and fz is not None and fz < -self.cfg.funding_z_veto:
                logger.debug("%s: long vetoed — funding z=%.2f (longs charged)", sym, fz)
                continue
            if z < 0 and fz is not None and fz > self.cfg.funding_z_veto:
                logger.debug("%s: short vetoed — funding z=%.2f (sentiment already negative)", sym, fz)
                continue

            alpha = float(np.clip(abs(z) / self.cfg.z_entry, 0.0, 1.0))
            if alpha < self.cfg.min_alpha:
                continue

            side = "buy" if z > 0 else "sell"
            sig = MomentumSignal(
                symbol=sym,
                side=side,
                alpha=alpha,
                momentum_z=z,
                return_24h=ret_7d,
                funding_rate=fd.funding_rate_8h,
                funding_z=fz if fz is not None else 0.0,
                mark_price=fd.mark_price,
                vol_1w=vol,
            )
            candidates.append((abs(z), sig))

        # Sort by conviction and return top_n longs + top_n shorts
        candidates.sort(key=lambda x: x[0], reverse=True)
        longs  = [s for _, s in candidates if s.side == "buy" ][: self.cfg.top_n]
        shorts = [s for _, s in candidates if s.side == "sell"][: self.cfg.top_n]
        signals = longs + shorts
        signals.sort(key=lambda s: s.alpha, reverse=True)
        return signals

    def should_exit(self, sym: str, entry_side: str) -> bool:
        """True if the momentum signal has reversed below z_exit threshold."""
        result = self._momentum_z(sym)
        if result is None:
            return False
        z, _, _ = result
        if entry_side == "buy"  and z < self.cfg.z_exit:
            return True
        if entry_side == "sell" and z > -self.cfg.z_exit:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Formatted output                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_signal(s: MomentumSignal) -> str:
        direction = "LONG " if s.side == "buy" else "SHORT"
        return (
            f"{s.symbol:6s} {direction}  "
            f"α={s.alpha:.2f}  mom_z={s.momentum_z:+.2f}  "
            f"ret24h={s.return_24h*100:+.2f}%  "
            f"fund_z={s.funding_z:+.2f}  "
            f"vol1w={s.vol_1w*100:.1f}%  "
            f"mark={s.mark_price:.4f}"
        )
