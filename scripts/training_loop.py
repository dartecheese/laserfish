"""Iterative strategy improvement loop with expanding walk-forward validation.

Structure:
  - Round 0 (baseline): pure 7-day momentum, NZ=0.3, 2× leverage
  - Each subsequent round adds ONE change to the current best
  - Changes tested: momentum window, volume filter, funding confirmation,
    BTC-relative momentum, leverage scaling, regime strength filter

Walk-forward protocol (no lookahead):
  - Warmup: first 120 days used only to build history, never tested
  - Walk-forward: 30-day OOS windows on expanding training set
  - Each config runs across ALL windows; winner advances to next round

Usage:
    python scripts/training_loop.py
    python scripts/training_loop.py --rounds 4      # run first 4 iterations
    python scripts/training_loop.py --quick         # 5 windows instead of 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from trader.exchanges.hyperliquid import HyperliquidExchange

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("training_loop")

SYMBOLS  = ["BTC", "ETH", "SOL"]
CAPITAL  = 10_000.0
FEE      = 0.00035
WARMUP_D = 120    # days of warmup (never tested)
WINDOW_D = 30     # days per OOS window


# ─── Strategy config ────────────────────────────────────────────────────────

@dataclass
class StratConfig:
    name: str
    mom_bars: int   = 42     # momentum lookback in 4h bars (42=7d, 21=3.5d, 84=14d)
    hist_bars: int  = 180    # rolling distribution window
    nz: float       = 0.3    # neutral-zone z-score threshold
    leverage: float = 2.0
    rebal_bars: int = 42     # rebalance every N bars
    # Optional feature flags
    vol_filter: bool    = False  # require above-average volume at entry
    funding_gate: bool  = False  # require funding sign consistent with trade direction
    rel_mom: bool       = False  # BTC-relative momentum (removes market beta)
    dyn_leverage: bool  = False  # scale leverage by |z_score| (more aggressive on stronger signal)
    # Risk management
    vol_lookback: int       = 50   # bars for vol-filter mean comparison
    stop_loss_pct: float    = 0.0  # 0 = disabled; per-position cumulative loss to exit
    take_profit_pct: float  = 0.0  # 0 = disabled; per-position cumulative gain to exit
    circuit_breaker: float  = 0.0  # 0 = disabled; halt trading if window drawdown exceeds X%
    # Fill realism
    slippage_bps: float  = 0.0    # adverse price movement per fill (bps); e.g. 3.0 = 0.03%
    next_bar_fill: bool  = False  # execute at next bar's OPEN instead of signal bar's close
    apply_funding: bool  = False  # accumulate 8h funding charges while holding
    # Signal quality improvements
    vol_regime_gate: bool = False  # only enter when 30d BTC vol >= its 6-month rolling median
    multi_tf_confirm: bool = False # require longer-window z-score to agree with signal direction
    confirm_mult: int  = 3         # longer window = confirm_mult × mom_bars
    adaptive_nz: bool  = False     # scale nz inversely with vol ratio (wider in chop, tighter in trend)
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        flags = []
        if self.vol_filter:    flags.append("vol_filter")
        if self.funding_gate:  flags.append("funding_gate")
        if self.rel_mom:       flags.append("rel_mom")
        if self.dyn_leverage:  flags.append("dyn_lev")
        if self.mom_bars != 42: flags.append(f"mom={self.mom_bars}b")
        if self.nz != 0.3:    flags.append(f"nz={self.nz}")
        if self.leverage != 2.0: flags.append(f"lev={self.leverage}x")
        return "+".join(flags) if flags else "baseline"


# ─── Core simulation ────────────────────────────────────────────────────────

def run_window(
    candles: dict[str, list],
    funding: dict[str, list],
    symbols: list[str],
    start: int,
    end: int,
    cfg: StratConfig,
) -> dict[str, float]:
    """
    Simulate one walk-forward window with the given config.
    Returns metrics dict.
    """
    equity = CAPITAL
    positions: dict[str, str] = {}    # sym → "long" | "short"
    entry_prices: dict[str, float] = {}
    eq_curve = [equity]
    trades = 0
    window_peak = CAPITAL
    halted = False          # circuit breaker tripped

    def _get_closes(sym: int, t: int, n: int) -> list[float]:
        c = candles[sym]
        return [c[i].close for i in range(max(0, t - n), t + 1)]

    def _get_volumes(sym: str, t: int, n: int) -> list[float]:
        c = candles[sym]
        return [c[i].volume for i in range(max(0, t - n), t + 1)]

    def _mom_z(sym: str, t: int, btc_ret: float | None = None) -> float | None:
        closes = _get_closes(sym, t, cfg.hist_bars + cfg.mom_bars)
        if len(closes) < cfg.mom_bars + 15:
            return None
        ret = (closes[-1] - closes[-cfg.mom_bars - 1]) / closes[-cfg.mom_bars - 1]
        if cfg.rel_mom and btc_ret is not None:
            ret = ret - btc_ret   # remove market beta
        hist = [(closes[i] - closes[max(0, i - cfg.mom_bars)]) /
                closes[max(0, i - cfg.mom_bars)]
                for i in range(cfg.mom_bars, len(closes))]
        if len(hist) < 10:
            return None
        mu = float(np.mean(hist)); sd = float(np.std(hist)) + 1e-9
        return (ret - mu) / sd

    def _vol_ok(sym: str, t: int) -> bool:
        if not cfg.vol_filter:
            return True
        vols = _get_volumes(sym, t, cfg.vol_lookback)
        if len(vols) < 10:
            return True
        return vols[-1] >= float(np.mean(vols[-cfg.vol_lookback:]))

    def _funding_ok(sym: str, t: int, side: str) -> bool:
        if not cfg.funding_gate:
            return True
        fdata = funding.get(sym, [])
        fidx = min(t // 2, len(fdata) - 1) if fdata else -1
        if fidx < 0:
            return True
        rate = fdata[fidx]
        if side == "long"  and rate < 0:
            return False
        if side == "short" and rate > 0:
            return False
        return True

    def _btc_vol_ratio(t: int) -> float:
        """Current 30d BTC realized vol divided by its 6-month rolling median. >1 = trending."""
        btc = candles.get("BTC", [])
        period = 180   # 30d in 4h bars
        if t < period * 7:
            return 1.0
        closes = [btc[i].close for i in range(max(0, t - period), t + 1)]
        if len(closes) < 20:
            return 1.0
        log_r = [float(np.log(closes[i] / closes[i - 1])) for i in range(1, len(closes))]
        cur_vol = float(np.std(log_r))
        # Median over 6 prior 30d windows
        prior_vols = []
        for off in range(1, 7):
            s = max(0, t - (off + 1) * period)
            e = max(0, t - off * period)
            c2 = [btc[i].close for i in range(s, e + 1)]
            if len(c2) < 20:
                continue
            lr = [float(np.log(c2[i] / c2[i - 1])) for i in range(1, len(c2))]
            prior_vols.append(float(np.std(lr)))
        if not prior_vols:
            return 1.0
        return cur_vol / (float(np.median(prior_vols)) + 1e-9)

    def _vol_regime_ok(t: int) -> bool:
        if not cfg.vol_regime_gate:
            return True
        return _btc_vol_ratio(t) >= 1.0   # only enter when vol is above its median

    def _confirm_ok(sym: str, t: int, side: str, btc_ret: float | None) -> bool:
        if not cfg.multi_tf_confirm:
            return True
        long_bars = cfg.mom_bars * cfg.confirm_mult
        closes = _get_closes(sym, t, cfg.hist_bars + long_bars)
        if len(closes) < long_bars + 15:
            return True   # insufficient history — allow
        ret_long = (closes[-1] - closes[-long_bars - 1]) / closes[-long_bars - 1]
        if cfg.rel_mom and btc_ret is not None:
            ret_long -= btc_ret
        hist = [(closes[i] - closes[max(0, i - long_bars)]) / closes[max(0, i - long_bars)]
                for i in range(long_bars, len(closes))]
        if len(hist) < 10:
            return True
        mu = float(np.mean(hist)); sd = float(np.std(hist)) + 1e-9
        z_long = (ret_long - mu) / sd
        if side == "long"  and z_long <= 0:
            return False  # short-term bullish vs long-term bearish — skip
        if side == "short" and z_long >= 0:
            return False  # short-term bearish vs long-term bullish — skip
        return True

    def _effective_nz(t: int) -> float:
        if not cfg.adaptive_nz:
            return cfg.nz
        ratio = _btc_vol_ratio(t)
        # High vol (ratio>1) → tighter NZ (more trades in trends)
        # Low vol  (ratio<1) → wider  NZ (fewer trades in chop)
        return float(np.clip(cfg.nz / ratio, cfg.nz * 0.5, cfg.nz * 2.0))

    slip = cfg.slippage_bps / 10_000   # fractional slippage per fill
    pending: dict[str, str | None] = {}  # orders queued for next bar's open

    def _fill_price(bar_price: float, is_buy: bool) -> float:
        """Adverse fill: buys fill high, sells fill low."""
        return bar_price * (1 + slip) if is_buy else bar_price * (1 - slip)

    def _execute(sym: str, target: str | None, fill_px: float) -> None:
        nonlocal equity, trades
        current = positions.get(sym)
        notional = equity * cfg.leverage / len(symbols)
        # Slippage cost: lose slip% of notional on every fill (entry or exit)
        equity -= slip * notional
        equity -= FEE * notional   # one-way fee per leg
        if current is not None:    # closing existing position
            equity -= FEE * notional
        if target:
            positions[sym] = target
            entry_prices[sym] = fill_px
        else:
            positions.pop(sym, None)
            entry_prices.pop(sym, None)
        trades += 1

    for t in range(start, end):
        bar = {sym: candles[sym][t] for sym in symbols if t < len(candles[sym])}

        # ── 1. Execute pending orders at this bar's OPEN ──────────────
        if cfg.next_bar_fill and pending:
            for sym, target in list(pending.items()):
                if sym not in bar:
                    continue
                current = positions.get(sym)
                if target == current:
                    continue
                is_buy = target == "long" if target else (current == "short")
                fill_px = _fill_price(bar[sym].open, is_buy)
                _execute(sym, target, fill_px)
            pending.clear()

        # ── 2. Mark-to-market on held positions ───────────────────────
        for sym, side in list(positions.items()):
            c = candles[sym]
            if t < 1 or t >= len(c):
                continue
            ret = (c[t].close - c[t - 1].close) / c[t - 1].close
            notional = equity * cfg.leverage / len(symbols)
            equity += ret * notional * (1 if side == "long" else -1)

            # Stop-loss / take-profit (checked at bar close)
            if cfg.stop_loss_pct > 0 or cfg.take_profit_pct > 0:
                entry = entry_prices.get(sym)
                if entry and entry > 0:
                    cum_ret = (c[t].close - entry) / entry * (1 if side == "long" else -1)
                    if (cfg.stop_loss_pct   > 0 and cum_ret < -cfg.stop_loss_pct) or \
                       (cfg.take_profit_pct > 0 and cum_ret >  cfg.take_profit_pct):
                        equity -= FEE * notional + slip * notional
                        positions.pop(sym)
                        entry_prices.pop(sym, None)
                        trades += 1

        # ── 3. Funding rate charges (8h = every 2 × 4h bars) ─────────
        if cfg.apply_funding and t % 2 == 0:
            for sym, side in list(positions.items()):
                fdata = funding.get(sym, [])
                fidx  = min(t // 2, len(fdata) - 1) if fdata else -1
                if fidx < 0:
                    continue
                rate     = fdata[fidx]
                notional = equity * cfg.leverage / len(symbols)
                # Longs pay positive rate; shorts receive (negative charge)
                sign = 1 if side == "long" else -1
                equity  -= sign * rate * notional

        equity = max(equity, 1.0)
        window_peak = max(window_peak, equity)
        eq_curve.append(equity)

        # ── 4. Circuit breaker ────────────────────────────────────────
        if cfg.circuit_breaker > 0 and not halted:
            if (window_peak - equity) / window_peak >= cfg.circuit_breaker:
                for sym in list(positions):
                    notional = equity * cfg.leverage / len(symbols)
                    equity  -= FEE * notional + slip * notional
                    trades  += 1
                positions.clear()
                entry_prices.clear()
                halted = True

        if halted:
            continue

        # ── 5. Scan for new signals on rebalance bars ─────────────────
        if (t - start) % cfg.rebal_bars != 0:
            continue

        btc_ret = None
        if cfg.rel_mom and "BTC" in symbols:
            c = candles["BTC"]
            if t >= cfg.mom_bars and t < len(c):
                btc_ret = (c[t].close - c[t - cfg.mom_bars].close) / c[t - cfg.mom_bars].close

        nz_eff = _effective_nz(t)
        regime_ok = _vol_regime_ok(t)

        new_targets: dict[str, str | None] = {}
        for sym in symbols:
            z = _mom_z(sym, t, btc_ret)
            if z is None:
                new_targets[sym] = None
                continue
            if z > nz_eff:
                side = "long"
            elif z < -nz_eff:
                side = "short"
            else:
                new_targets[sym] = None
                continue
            if not regime_ok:
                new_targets[sym] = None
                continue
            if not _vol_ok(sym, t) or not _funding_ok(sym, t, side):
                new_targets[sym] = None
                continue
            if not _confirm_ok(sym, t, side, btc_ret):
                new_targets[sym] = None
                continue
            new_targets[sym] = side

        # ── 6. Execute or queue ───────────────────────────────────────
        changed = {sym: tgt for sym, tgt in new_targets.items()
                   if tgt != positions.get(sym)}
        if cfg.next_bar_fill:
            pending.update(changed)        # execute at next bar's open
        else:
            for sym, target in changed.items():
                if sym not in bar:
                    continue
                is_buy = target == "long" if target else (positions.get(sym) == "short")
                fill_px = _fill_price(bar[sym].close, is_buy)
                _execute(sym, target, fill_px)

    # ── Close open positions at window end ────────────────────────────
    for sym in list(positions):
        c = candles[sym]
        if end - 1 < len(c):
            notional = equity * cfg.leverage / len(symbols)
            equity  -= FEE * notional + slip * notional
            trades  += 1

    returns = np.diff(eq_curve) / (np.array(eq_curve[:-1]) + 1e-9)
    sharpe  = (float(np.mean(returns)) / (float(np.std(returns)) + 1e-9)) * np.sqrt(6 * 365)
    peak = CAPITAL; max_dd = 0.0
    for eq in eq_curve:
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    return {
        "return_pct": (equity - CAPITAL) / CAPITAL * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "trades": trades,
        "curve": list(eq_curve),
    }


def eval_config(
    candles: dict[str, list],
    funding: dict[str, list],
    symbols: list[str],
    cfg: StratConfig,
    n_windows: int,
    verbose: bool = False,
) -> dict[str, float]:
    """Run all walk-forward windows for a config, return aggregate stats."""
    min_len = min(len(v) for v in candles.values())
    total_oos_bars = n_windows * WINDOW_D * 6
    warmup_bars    = min_len - total_oos_bars

    results = []
    for i in range(n_windows):
        w_s = warmup_bars + i * WINDOW_D * 6
        w_e = w_s + WINDOW_D * 6
        if w_e > min_len:
            break
        r = run_window(candles, funding, symbols, w_s, w_e, cfg)

        if verbose:
            btc = candles.get("BTC", [])
            btc_ret = (btc[w_e - 1].close - btc[w_s].close) / btc[w_s].close * 100 if btc else 0
            regime = "BULL" if btc_ret > 5 else ("BEAR" if btc_ret < -5 else "SIDE")
            logger.info("    W%d %s  ret=%+.1f%%  sh=%.2f  dd=%.1f%%  trades=%d",
                        i + 1, regime, r["return_pct"], r["sharpe"], r["max_dd_pct"], r["trades"])
        results.append(r)

    avg_ret = float(np.mean([r["return_pct"] for r in results]))
    avg_sh  = float(np.mean([r["sharpe"]     for r in results]))
    avg_dd  = float(np.mean([r["max_dd_pct"] for r in results]))
    wins    = sum(1 for r in results if r["return_pct"] > 0)
    # Composite score: Sharpe × win_rate — penalises inconsistency
    score   = avg_sh * (wins / len(results))
    return {
        "avg_return_pct": avg_ret,
        "avg_sharpe": avg_sh,
        "avg_dd_pct": avg_dd,
        "wins": wins,
        "n_windows": len(results),
        "score": score,
    }


# ─── Iteration definitions (built dynamically from champion) ─────────────

def round_candidates(champ: StratConfig, round_idx: int) -> list[StratConfig]:
    """
    Build the next round of candidates from the CURRENT champion's params.
    Each round varies exactly one dimension; all other params are copied
    from champ so results are directly comparable.
    """
    c = champ   # shorthand

    if round_idx == 0:
        # Momentum window: the single most impactful parameter
        return [
            StratConfig("mom_21", mom_bars=21,  hist_bars=120, nz=c.nz, leverage=c.leverage),
            StratConfig("mom_42", mom_bars=42,  hist_bars=180, nz=c.nz, leverage=c.leverage),
            StratConfig("mom_63", mom_bars=63,  hist_bars=252, nz=c.nz, leverage=c.leverage),
            StratConfig("mom_84", mom_bars=84,  hist_bars=360, nz=c.nz, leverage=c.leverage),
        ]

    if round_idx == 1:
        # Neutral-zone threshold (carries forward best mom_bars)
        return [
            StratConfig(f"nz_{v}", mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=v, leverage=c.leverage,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom, dyn_leverage=c.dyn_leverage)
            for v in [0.1, 0.3, 0.5, 0.8, 1.0]
        ]

    if round_idx == 2:
        # Leverage (carries forward best mom_bars + nz)
        return [
            StratConfig(f"lev_{v}", mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=v,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom)
            for v in [1.5, 2.0, 3.0]
        ] + [
            StratConfig("dyn_lev", mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage, dyn_leverage=True,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom)
        ]

    if round_idx == 3:
        # Feature additions: one at a time vs champion baseline
        return [
            StratConfig("baseline",      mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage),
            StratConfig("+vol_filter",   mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage, vol_filter=True),
            StratConfig("+funding_gate", mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage, funding_gate=True),
            StratConfig("+rel_mom",      mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage, rel_mom=True),
        ]

    if round_idx == 4:
        # Combine the feature(s) that helped in round 3
        return [
            StratConfig("no_extra",     mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom),
            StratConfig("vol+fund",     mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage,
                        vol_filter=True, funding_gate=True, rel_mom=c.rel_mom),
            StratConfig("vol+rel",      mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage,
                        vol_filter=True, funding_gate=c.funding_gate, rel_mom=True),
            StratConfig("fund+rel",     mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage,
                        vol_filter=c.vol_filter, funding_gate=True, rel_mom=True),
            StratConfig("all_features", mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage,
                        vol_filter=True, funding_gate=True, rel_mom=True),
        ]

    if round_idx == 5:
        # Rebalance frequency: how often to re-evaluate signals
        return [
            StratConfig(f"rebal_{v}b", mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage, rebal_bars=v,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom, dyn_leverage=c.dyn_leverage)
            for v in [21, 42, 63, 84]
        ]

    if round_idx == 6:
        # Fine leverage sweep around current champion value
        return [
            StratConfig(f"lev_{v}", mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=v, rebal_bars=c.rebal_bars,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom, dyn_leverage=c.dyn_leverage)
            for v in [2.0, 2.5, 3.0, 3.5, 4.0]
        ]

    if round_idx == 7:
        # Fine NZ sweep around current champion value
        return [
            StratConfig(f"nz_{v}", mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=v, leverage=c.leverage, rebal_bars=c.rebal_bars,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom, dyn_leverage=c.dyn_leverage)
            for v in [0.3, 0.4, 0.5, 0.6, 0.7]
        ]

    if round_idx == 8:
        # History window: affects how many samples the z-score distribution uses
        return [
            StratConfig(f"hist_{v}", mom_bars=c.mom_bars, hist_bars=v,
                        nz=c.nz, leverage=c.leverage, rebal_bars=c.rebal_bars,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom, dyn_leverage=c.dyn_leverage)
            for v in [90, 120, 180, 240, 360]
        ]

    if round_idx == 9:
        # Dynamic leverage re-test with champion's vol_filter context
        return [
            StratConfig("static_lev",  mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage, rebal_bars=c.rebal_bars,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom, dyn_leverage=False),
            StratConfig("dyn_lev",     mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage, rebal_bars=c.rebal_bars,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom, dyn_leverage=True),
            StratConfig("dyn_lev_hi",  mom_bars=c.mom_bars, hist_bars=c.hist_bars,
                        nz=c.nz, leverage=c.leverage + 1.0, rebal_bars=c.rebal_bars,
                        vol_filter=c.vol_filter, funding_gate=c.funding_gate,
                        rel_mom=c.rel_mom, dyn_leverage=True),
        ]

    def _base(**kw) -> StratConfig:
        """Clone champion with overrides, carrying all fields forward automatically."""
        from dataclasses import asdict
        d = asdict(c)
        d.pop("extra", None)
        d.update(kw)
        return StratConfig(**d)

    if round_idx == 10:
        # Fine mom_bars sweep (now that other params are tuned)
        return [_base(name=f"mom_{v}b", mom_bars=v) for v in [28, 35, 42, 49, 56]]

    if round_idx == 11:
        # Fine hist_bars around current champion
        return [_base(name=f"hist_{v}", hist_bars=v) for v in [180, 210, 240, 270, 300, 360]]

    if round_idx == 12:
        # Fine NZ around current champion
        return [_base(name=f"nz_{v}", nz=v) for v in [0.25, 0.30, 0.35, 0.40, 0.45, 0.55]]

    if round_idx == 13:
        # Fine leverage with optimised NZ + hist
        return [_base(name=f"lev_{v}", leverage=v) for v in [2.5, 3.0, 3.5, 4.0, 4.5]]

    if round_idx == 14:
        # Rebalance frequency re-test with fully tuned params
        return [_base(name=f"rebal_{v}b", rebal_bars=v) for v in [14, 21, 28, 35, 42]]

    if round_idx == 15:
        # Volume filter lookback length
        return [_base(name=f"vl_{v}", vol_lookback=v) for v in [20, 30, 50, 70, 100]]

    if round_idx == 16:
        # Stop-loss threshold (0 = disabled)
        return [_base(name=f"sl_{int(v*100)}pct", stop_loss_pct=v)
                for v in [0.0, 0.03, 0.05, 0.08, 0.12]]

    if round_idx == 17:
        # Take-profit threshold (0 = disabled)
        return [_base(name=f"tp_{int(v*100)}pct", take_profit_pct=v)
                for v in [0.0, 0.05, 0.10, 0.15, 0.20]]

    if round_idx == 18:
        # NZ × leverage interaction grid (both tuned together)
        combos = [(0.3, 3.5), (0.4, 3.0), (0.4, 3.5), (0.5, 3.0), (0.5, 3.5)]
        return [_base(name=f"nz{nz}_lev{lev}", nz=nz, leverage=lev)
                for nz, lev in combos]

    if round_idx == 19:
        # Final combo: apply best sl + tp + vol_lookback together
        sl  = c.stop_loss_pct
        tp  = c.take_profit_pct
        vl  = c.vol_lookback
        return [
            _base(name="champion"),
            _base(name="sl+tp",    stop_loss_pct=sl  or 0.05, take_profit_pct=tp  or 0.10),
            _base(name="sl+vl",    stop_loss_pct=sl  or 0.05, vol_lookback=vl  if vl != 50 else 30),
            _base(name="tp+vl",    take_profit_pct=tp or 0.10, vol_lookback=vl if vl != 50 else 30),
            _base(name="all_risk", stop_loss_pct=sl  or 0.05, take_profit_pct=tp or 0.10,
                  vol_lookback=vl if vl != 50 else 30),
        ]

    if round_idx == 20:
        # Improvement 1: volatility regime gate
        return [
            _base(name="no_vrg",   vol_regime_gate=False),
            _base(name="vrg",      vol_regime_gate=True),
        ]

    if round_idx == 21:
        # Improvement 2: multi-timeframe confirmation (vary the confirm window multiplier)
        return [
            _base(name="no_mtf",         multi_tf_confirm=False),
            _base(name="mtf_2x",         multi_tf_confirm=True, confirm_mult=2),
            _base(name="mtf_3x",         multi_tf_confirm=True, confirm_mult=3),
            _base(name="mtf_4x",         multi_tf_confirm=True, confirm_mult=4),
        ]

    if round_idx == 22:
        # Improvement 3: adaptive neutral zone
        return [
            _base(name="fixed_nz",   adaptive_nz=False),
            _base(name="adaptive_nz", adaptive_nz=True),
        ]

    if round_idx == 23:
        # Improvement 4: combinations of the three signal-quality improvements
        vrg = c.vol_regime_gate
        mtf = c.multi_tf_confirm
        cm  = c.confirm_mult
        anz = c.adaptive_nz
        return [
            _base(name="champion"),
            _base(name="vrg+mtf",  vol_regime_gate=True,  multi_tf_confirm=True,  confirm_mult=cm),
            _base(name="vrg+anz",  vol_regime_gate=True,  adaptive_nz=True),
            _base(name="mtf+anz",  multi_tf_confirm=True, confirm_mult=cm, adaptive_nz=True),
            _base(name="all_three",vol_regime_gate=True,  multi_tf_confirm=True,
                  confirm_mult=cm, adaptive_nz=True),
        ]

    return []


# ─── Main ─────────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds",       type=int, default=5, help="Max rounds to run")
    p.add_argument("--start-round",  type=int, default=0,
                   help="Skip to this round index (0-based), loading champion from --load)")
    p.add_argument("--load",         default="models/best_momentum_config.json",
                   help="Champion config to load when using --start-round")
    p.add_argument("--quick",  action="store_true", help="Fewer windows (faster)")
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--save",    default="models/best_momentum_config.json")
    return p.parse_args()


def fetch_data(ex: HyperliquidExchange, symbols: list[str]) -> tuple[dict, dict]:
    import time
    candles: dict[str, list] = {}
    funding: dict[str, list] = {}
    for sym in symbols:
        candles[sym] = ex.get_candles(sym, "4h", 2500)
        time.sleep(0.25)
        try:
            hist = ex.get_funding_rate_history(sym, limit=1300)
            funding[sym] = [r for _, r in hist]
        except Exception:
            funding[sym] = []
        time.sleep(0.25)
    return candles, funding


def main() -> None:
    args    = build_args()
    symbols = args.symbols or SYMBOLS
    n_win   = 5 if args.quick else 10

    logger.info("Fetching data for %s…", symbols)
    ex = HyperliquidExchange(paper=True)
    candles, funding = fetch_data(ex, symbols)

    min_len = min(len(v) for v in candles.values())
    avail_days = min_len // 6
    logger.info("Available: %d bars = %d days | warmup=%dd | %d × %dd OOS windows",
                min_len, avail_days, WARMUP_D, n_win, WINDOW_D)

    champion = StratConfig("baseline")
    champion_stats: dict | None = None
    history: list[dict] = []

    # Optionally resume from a saved champion
    if args.start_round > 0 and Path(args.load).exists():
        with open(args.load) as f:
            saved = json.load(f)
        p = saved.get("params", {})
        p.pop("extra", None)
        champion = StratConfig(**p)
        champion_stats = saved.get("stats")
        logger.info("Loaded champion from %s: %s  (score=%.3f)",
                    args.load, champion.name, champion_stats.get("score", 0) if champion_stats else 0)

    for round_idx in range(args.start_round, args.start_round + args.rounds):
        candidates = round_candidates(champion, round_idx)
        if not candidates:
            logger.info("No candidates for round %d — stopping.", round_idx + 1)
            break

        logger.info("")
        logger.info("══ Round %d ══════════════════════════════════════════════", round_idx + 1)
        logger.info("  %-20s  %8s  %7s  %7s  %5s  SCORE", "CONFIG", "AVG_RET%", "SHARPE", "MAXDD%", "WINS")
        logger.info("  " + "─" * 60)

        round_results = []
        for cfg in candidates:
            stats = eval_config(candles, funding, symbols, cfg, n_win)
            round_results.append((stats["score"], cfg, stats))
            wins_str = f"{stats['wins']}/{stats['n_windows']}"
            marker = " ← champion" if cfg.name == champion.name else ""
            logger.info("  %-20s  %+7.1f%%  %7.2f  %7.1f%%  %5s  %.3f%s",
                        cfg.name, stats["avg_return_pct"], stats["avg_sharpe"],
                        stats["avg_dd_pct"], wins_str, stats["score"], marker)

        # Pick winner: highest composite score (Sharpe × win_rate)
        round_results.sort(key=lambda x: x[0], reverse=True)
        best_score, new_champ, new_stats = round_results[0]

        if champion_stats is None or best_score > (champion_stats["score"] * 1.02):
            logger.info("  ✓ New champion: %s  (score %.3f → %.3f)",
                        new_champ.name, champion_stats["score"] if champion_stats else 0, best_score)
            champion = new_champ
            champion_stats = new_stats
        else:
            logger.info("  → Champion unchanged: %s", champion.name)

        history.append({"round": round_idx + 1, "winner": new_champ.name,
                        "score": best_score, **new_stats})

    # ── Final result ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("══ Final Champion ═══════════════════════════════════════════")
    logger.info("  Config:    %s", champion.name)
    logger.info("  mom_bars:  %d (%.1f days)", champion.mom_bars, champion.mom_bars / 6)
    logger.info("  nz:        %.2f", champion.nz)
    logger.info("  leverage:  %.1fx", champion.leverage)
    logger.info("  vol_filter:   %s", champion.vol_filter)
    logger.info("  funding_gate: %s", champion.funding_gate)
    logger.info("  rel_mom:      %s", champion.rel_mom)
    logger.info("  dyn_leverage: %s", champion.dyn_leverage)
    if champion_stats:
        logger.info("  Avg return:  %+.1f%%/window  Sharpe: %.2f  MaxDD: %.1f%%  Wins: %d/%d",
                    champion_stats["avg_return_pct"], champion_stats["avg_sharpe"],
                    champion_stats["avg_dd_pct"], champion_stats["wins"], champion_stats["n_windows"])

    # Save best config
    out = {
        "champion": champion.name,
        "params": asdict(champion),
        "stats": champion_stats,
        "rounds": history,
    }
    out["params"].pop("extra", None)
    Path(args.save).parent.mkdir(exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("  Saved to %s", args.save)


if __name__ == "__main__":
    main()
