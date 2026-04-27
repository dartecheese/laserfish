"""Laserfish — momentum + Transformer + HMM regime + dynamic leverage + RL on Hyperliquid.

Signal stack:
  1. 7d cross-sectional momentum (primary direction signal)
  2. HMM regime filter (3-state: trend/range/crisis — gates entries + scales leverage)
  3. Dynamic Kelly leverage (vol-targeting + regime multiplier + drawdown circuit breaker)
  4. Transformer entry filter (P(BUY)-P(SELL) must agree with momentum direction)
  5. RL agent (PPO) — continuous leverage control on BTC, 15m cadence

Dry-run (paper mode, default):
    python scripts/run.py

With RL agent on BTC (30% of equity, paper):
    python scripts/run.py --rl

Live trading:
    HL_PRIVATE_KEY=0x... HL_WALLET_ADDRESS=0x... \\
    python scripts/run.py --model models/transformer_5m.onnx --rl --live
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from trader.exchanges.hyperliquid import HyperliquidExchange
from trader.exchanges.base import BracketParams
from trader.strategies.momentum import MomentumConfig, MomentumStrategy
from trader.risk import RiskConfig, RiskManager
from trader.bars import make_volume_bars, auto_threshold
from trader.data import Kline
from trader.features import live_sequence
from trader.model import load_onnx, infer_alpha
from trader.regime import RegimeDetector, REGIME_LABEL, REGIME_COLORS
from trader.leverage import DynamicLeverage, LeverageConfig
from trader.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy, MRSignal
from trader.strategies.grid import GridConfig, GridStrategy

try:
    from src.live.executor import RLExecutor
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Minimum Transformer alpha to confirm a momentum entry.
TRANSFORMER_MIN_CONFIRM = 0.10

# Do not enter new positions in crisis regime — only manage existing ones.
BLOCK_ENTRIES_IN_CRISIS = True

# RL agent runs every 15 minutes (1 bar)
RL_STEP_INTERVAL = 900   # seconds


def _candles_to_klines(candles) -> list[Kline]:
    return [Kline(
        open_time=int(c.timestamp), open=float(c.open), high=float(c.high),
        low=float(c.low), close=float(c.close), volume=float(c.volume),
        close_time=int(c.timestamp) + 300_000, quote_volume=float(c.quote_volume), trades=0,
    ) for c in candles]


def transformer_confirms(session, exchange, symbol: str, side: str) -> bool:
    """Return True if the Transformer agrees with the momentum direction."""
    try:
        candles = exchange.get_candles(symbol, "5m", 500)
        klines = _candles_to_klines(candles)
        if not klines:
            return True
        threshold = auto_threshold(klines, 48.0, "volume")
        bars = make_volume_bars(klines, threshold)
        if len(bars) < 32:
            return True
        seq = live_sequence(bars)
        alpha = infer_alpha(session, seq)
        return (side == "buy" and alpha >= TRANSFORMER_MIN_CONFIRM) or \
               (side == "sell" and alpha <= -TRANSFORMER_MIN_CONFIRM)
    except Exception:
        return True


class RegimeTracker:
    """
    Maintains per-symbol price/funding/OI history for regime feature construction,
    fits the HMM on BTC anchor data during warm-up, and updates online each scan.
    """

    def __init__(self, exchange: HyperliquidExchange, anchor: str = "BTC"):
        self.ex = exchange
        self.anchor = anchor
        self.detector = RegimeDetector()

        # Rolling history for anchor symbol
        self._prices:   deque[float] = deque(maxlen=1000)
        self._funding:  deque[float] = deque(maxlen=200)
        self._oi:       deque[float] = deque(maxlen=200)

    def warm_up(self, n_candles: int = 500) -> None:
        """Fetch BTC anchor history and fit HMM."""
        log = logging.getLogger(__name__)
        try:
            candles = self.ex.get_candles(self.anchor, "5m", n_candles)
            for c in candles:
                self._prices.append(float(c.close))

            hist = self.ex.get_funding_rate_history(self.anchor, limit=100)
            for _, rate in hist:
                self._funding.append(rate)

            oi = self.ex.get_open_interest(self.anchor)
            for _ in range(50):
                self._oi.append(oi)  # seed with current value

            # Build obs matrix from price history
            obs_list = []
            prices = list(self._prices)
            funding = list(self._funding)
            oi_list = list(self._oi)
            for i in range(300, len(prices)):
                obs = RegimeDetector.make_obs(prices[:i+1], funding, oi_list)
                if obs is not None:
                    obs_list.append(obs)
                    self.detector.add_obs(obs)

            if obs_list:
                self.detector.fit()
                log.info("Regime HMM fitted | current state: %s %s",
                         REGIME_COLORS[self.detector.regime()],
                         self.detector.regime_label())
            else:
                log.warning("Regime HMM: insufficient obs for fit — defaulting to TREND")
        except Exception as e:
            log.warning("Regime warm-up failed: %s — defaulting to TREND state", e)

    def update(self) -> int:
        """Fetch latest BTC anchor data and return current regime state."""
        try:
            candles = self.ex.get_candles(self.anchor, "5m", 2)
            if candles:
                self._prices.append(float(candles[-1].close))
            fd = self.ex.get_funding_data(self.anchor)
            self._funding.append(fd.funding_rate_8h)
            oi = self.ex.get_open_interest(self.anchor)
            self._oi.append(oi)

            obs = RegimeDetector.make_obs(
                list(self._prices), list(self._funding), list(self._oi)
            )
            if obs is not None:
                return self.detector.regime(obs)
        except Exception:
            pass
        return self.detector.regime()


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(MomentumConfig().symbols))
    ap.add_argument("--model", default="models/transformer_5m.onnx",
                    help="ONNX model path. Omit or set to '' to run momentum-only.")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--max-leverage", type=float, default=5.0,
                    help="Hard cap on dynamic leverage (default 5x).")
    ap.add_argument("--target-vol", type=float, default=0.50,
                    help="Annualized vol target for Kelly sizing (default 0.50 = 50%%).")
    ap.add_argument("--max-position-pct", type=float, default=0.20)
    ap.add_argument("--z-entry", type=float, default=1.8)
    ap.add_argument("--poll-seconds", type=int, default=172800)  # 48h — tuned R7
    ap.add_argument("--live", action="store_true",
                    help="Place real orders. Default is paper/dry-run.")
    ap.add_argument("--rl", action="store_true",
                    help="Enable RL PPO agent on BTC (15m cadence, separate allocation).")
    ap.add_argument("--rl-model", default="models/rl_btc_perp.zip",
                    help="Path to trained PPO model zip.")
    ap.add_argument("--rl-allocation", type=float, default=0.30,
                    help="Fraction of equity allocated to RL agent (default 0.30 = 30%%).")
    args = ap.parse_args()

    dry_run = not args.live
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Load Transformer if available
    model_path = Path(args.model) if args.model else None
    session = None
    if model_path and model_path.exists():
        session = load_onnx(model_path)
        log.info("Transformer filter loaded: %s", model_path)
    else:
        log.info("No model found — running momentum-only mode.")

    exchange = HyperliquidExchange(paper=dry_run)
    risk = RiskManager(RiskConfig(max_position_pct=args.max_position_pct))

    cfg = MomentumConfig(
        symbols=symbols,
        top_n=args.top_n,
        max_leverage=args.max_leverage,
        z_entry=args.z_entry,
    )
    strategy = MomentumStrategy(exchange, cfg)

    # Dynamic leverage engine
    lev_cfg = LeverageConfig(
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
    )
    lev_engine = DynamicLeverage(lev_cfg)

    # HMM regime tracker (anchored on BTC)
    regime_tracker = RegimeTracker(exchange, anchor="BTC")

    # Mean-reversion + carry strategy (fires when momentum is absent)
    mr_cfg = MeanReversionConfig(symbols=symbols)
    mr_strategy = MeanReversionStrategy(exchange, mr_cfg)

    # ── Multi-asset grid configuration ────────────────────────────────────
    # Each symbol gets its own GridStrategy and per-symbol ranging detector.
    # Ranging gate = BTC market regime confirmed RANGE *and* per-symbol
    # choppiness index > CHOP_THRESHOLD (61.8 = Fibonacci level, standard).
    GRID_SYMBOLS        = ["BTC", "ETH", "SOL"]
    GRID_CHECK_INTERVAL = 60    # seconds between grid fill checks
    REGIME_CONFIRM_N    = 3     # consecutive identical HMM readings required

    # Per-symbol grid config — wider spacing for more volatile assets
    _grid_cfgs = {
        "BTC": GridConfig(symbol="BTC", spacing_pct=0.005, n_levels=10,
                          order_size_pct=0.03, max_leverage=2.0),
        "ETH": GridConfig(symbol="ETH", spacing_pct=0.006, n_levels=8,
                          order_size_pct=0.03, max_leverage=2.0),
        "SOL": GridConfig(symbol="SOL", spacing_pct=0.008, n_levels=8,
                          order_size_pct=0.02, max_leverage=2.0),
    }
    grids: dict[str, GridStrategy] = {
        sym: GridStrategy(exchange, cfg) for sym, cfg in _grid_cfgs.items()
    }
    # Stability buffers — BTC HMM is the shared market-level regime gate
    _regime_history: dict[str, deque] = {
        sym: deque(maxlen=REGIME_CONFIRM_N) for sym in GRID_SYMBOLS
    }
    _stable_regime: dict[str, int] = {sym: 0 for sym in GRID_SYMBOLS}
    _last_grid_check = 0.0

    # RL executor — PPO continuous leverage on BTC
    rl_executor = None
    _last_rl_step = 0.0
    if args.rl and RL_AVAILABLE:
        if Path(args.rl_model).exists():
            rl_executor = RLExecutor(
                exchange=exchange,
                symbol="BTC",
                paper_mode=dry_run,
                initial_equity=10_000.0,   # placeholder; updated on first step
            )
            log.info("RL executor loaded: %s | allocation=%.0f%%", args.rl_model,
                     args.rl_allocation * 100)
        else:
            log.warning("--rl flag set but no model at %s — RL disabled", args.rl_model)
    elif args.rl and not RL_AVAILABLE:
        log.warning("--rl flag set but src.live.executor not importable — RL disabled")

    log.info("Laserfish v2 starting | symbols=%d | transformer=%s | dynamic_leverage=True | rl=%s | dry_run=%s",
             len(symbols), session is not None, rl_executor is not None, dry_run)

    log.info("Warming up price + funding histories (shared across strategies)…")
    # Fetch once and share — avoids duplicate API calls and 429 rate limits
    needed = cfg.vol_window + cfg.momentum_window + 2
    batch = 500
    shared_prices: dict[str, list[float]] = {}
    shared_funding: dict[str, list[float]] = {}
    import time as _time
    for sym in symbols:
        try:
            all_closes: list[float] = []
            fetched = 0
            while fetched < needed:
                candles = exchange.get_candles(sym, "5m", min(batch, needed - fetched))
                if not candles:
                    break
                all_closes = [c.close for c in candles] + all_closes
                fetched += len(candles)
                if len(candles) < batch:
                    break
            shared_prices[sym] = all_closes[-(needed):]
            _time.sleep(0.3)
        except Exception as e:
            log.warning("Warm-up prices %s: %s", sym, e)
        try:
            hist = exchange.get_funding_rate_history(sym, limit=cfg.funding_window)
            shared_funding[sym] = [r for _, r in hist]
            _time.sleep(0.2)
        except Exception as e:
            log.warning("Warm-up funding %s: %s", sym, e)
    log.info("Shared warm-up complete (%d symbols).", len(shared_prices))

    strategy.warm_up(shared_prices=shared_prices, shared_funding=shared_funding)
    mr_strategy.warm_up(shared_prices=shared_prices, shared_funding=shared_funding)


    log.info("Warming up HMM regime detector…")
    regime_tracker.warm_up()

    log.info("Warm-up complete. Entering scan loop (poll=%ds).", args.poll_seconds)

    open_positions: dict[str, str] = {}       # symbol → side (momentum positions)
    mr_positions: dict[str, tuple[str, str]] = {}  # symbol → (side, signal_type)
    _last_grid_check = 0.0

    while True:
        try:
            balance = exchange.get_balance()
            equity = balance.total_usd
            positions = exchange.get_positions()
            pos_map = {p.symbol: p for p in positions}

            # Update high-water mark for drawdown tracking
            lev_engine.update_hwm(equity)
            drawdown = lev_engine.daily_drawdown(equity)

            # Update regime state
            regime_state = regime_tracker.update()
            regime_mult = regime_tracker.detector.leverage_multiplier()
            regime_name = regime_tracker.detector.regime_label()
            regime_icon = REGIME_COLORS.get(regime_state, "?")

            # BTC stability filter — shared market regime signal for all grid assets.
            # Only confirms a new regime when N consecutive readings agree.
            _regime_history["BTC"].append(regime_state)
            btc_hist = _regime_history["BTC"]
            if len(btc_hist) == REGIME_CONFIRM_N and len(set(btc_hist)) == 1:
                if _stable_regime["BTC"] != regime_state:
                    log.info("BTC market regime confirmed: %s → %s (%dx)",
                             REGIME_COLORS.get(_stable_regime["BTC"], "?"),
                             regime_icon, REGIME_CONFIRM_N)
                _stable_regime["BTC"] = regime_state
            # Mirror BTC confirmed regime to other grid symbols as market-level gate
            for sym in GRID_SYMBOLS:
                if sym != "BTC":
                    _stable_regime[sym] = _stable_regime["BTC"]

            log.info("Equity=%.0f  Drawdown=%.1f%%  Regime=%s %s  RegimeMult=%.1fx",
                     equity, drawdown * 100, regime_icon, regime_name, regime_mult)

            # Hard circuit breaker from risk manager
            if risk.check_drawdown(equity):
                log.warning("Max drawdown breached — skipping entries (equity=%.0f)", equity)
            else:
                # Exits: check momentum reversal regardless of regime
                for sym, entry_side in list(open_positions.items()):
                    if strategy.should_exit(sym, entry_side):
                        log.info("%s: momentum reversed → closing %s", sym, entry_side)
                        exchange.close_position(sym)
                        del open_positions[sym]

                # Block new entries in crisis regime
                if BLOCK_ENTRIES_IN_CRISIS and regime_state == 2:
                    log.info("Regime=CRISIS — blocking new entries, managing existing positions only")
                else:
                    signals = strategy.scan()
                    for sig in signals:
                        log.info(MomentumStrategy.format_signal(sig))

                        if sig.symbol in open_positions or sig.symbol in pos_map:
                            continue

                        if not risk.can_open_position(sig.symbol, positions, balance, equity):
                            log.debug("%s: risk limits prevent entry", sig.symbol)
                            continue

                        # Transformer confirmation gate
                        if session is not None:
                            if not transformer_confirms(session, exchange, sig.symbol, sig.side):
                                log.info("%s: Transformer veto — skipping %s", sig.symbol, sig.side)
                                continue
                            log.debug("%s: Transformer confirmed", sig.symbol)

                        # Dynamic leverage: vol-target + regime multiplier + circuit breaker
                        realized_vol = sig.vol_1w   # 7d realized vol from momentum signal
                        lev = lev_engine.compute(
                            realized_vol_24h=realized_vol,
                            regime_multiplier=regime_mult,
                            drawdown_pct=drawdown,
                            funding_z=abs(sig.funding_z),
                        )

                        qty = lev_engine.size_position(
                            equity=equity,
                            mark_price=sig.mark_price,
                            leverage=lev,
                            alpha=sig.alpha,
                            max_position_pct=args.max_position_pct,
                        )
                        if qty <= 0:
                            continue

                        log.info("%s: OPEN %s  qty=%.4f  mark=%.4f  lev=%.2fx  regime=%s",
                                 sig.symbol, sig.side.upper(), qty, sig.mark_price,
                                 lev, regime_name)

                        exchange.place_bracket_order(BracketParams(
                            symbol=sig.symbol,
                            side=sig.side,
                            quantity=qty,
                            price=None,
                            take_profit_pct=cfg.take_profit_pct,
                            stop_loss_pct=cfg.stop_loss_pct,
                            leverage=int(round(lev)),
                        ))

                        open_positions[sig.symbol] = sig.side

            # ── Mean-reversion + carry scan ───────────────────────────
            # MR fires in ranging/bear regimes; momentum fires in trending
            # Both can fire simultaneously on different symbols
            mr_poll_due = not hasattr(main, '_last_mr_poll') or \
                          (time.time() - getattr(main, '_last_mr_poll', 0)) >= 3600  # 1h cadence
            if mr_poll_due:
                main._last_mr_poll = time.time()
                try:
                    # Exit existing MR positions if signal reverted
                    for sym, (entry_side, sig_type) in list(mr_positions.items()):
                        if mr_strategy.should_exit(sym, entry_side, sig_type):
                            log.info("%s: MR/carry exit — %s signal normalized", sym, sig_type)
                            exchange.close_position(sym)
                            del mr_positions[sym]

                    mr_signals = mr_strategy.scan()
                    for sig in mr_signals:
                        log.info(MeanReversionStrategy.format_signal(sig))

                        all_open = set(open_positions) | set(mr_positions) | set(pos_map)
                        if sig.symbol in all_open:
                            continue

                        # Use lower leverage for MR (it's counter-trend)
                        mr_lev = lev_engine.compute(
                            realized_vol_24h=max(sig.mark_price * 0.0001, 0.40),
                            regime_multiplier=min(regime_mult, 1.0),  # cap at 1x for MR
                            drawdown_pct=drawdown,
                            funding_z=abs(sig.funding_z),
                        )
                        mr_lev = min(mr_lev, 2.0)  # hard cap MR at 2x

                        qty = lev_engine.size_position(
                            equity=equity,
                            mark_price=sig.mark_price,
                            leverage=mr_lev,
                            alpha=sig.alpha,
                            max_position_pct=0.10,  # smaller allocation for MR
                        )
                        if qty <= 0:
                            continue

                        log.info("%s: MR/CARRY OPEN %s  qty=%.4f  mark=%.4f  lev=%.2fx  type=%s",
                                 sig.symbol, sig.side.upper(), qty, sig.mark_price,
                                 mr_lev, sig.signal_type)

                        # MR uses tighter TP/SL than momentum
                        exchange.place_bracket_order(BracketParams(
                            symbol=sig.symbol,
                            side=sig.side,
                            quantity=qty,
                            price=None,
                            take_profit_pct=mr_cfg.take_profit_pct,
                            stop_loss_pct=mr_cfg.stop_loss_pct,
                            leverage=int(round(mr_lev)),
                        ))
                        mr_positions[sig.symbol] = (sig.side, sig.signal_type)

                except Exception as e:
                    log.error("MR scan error: %s", e)

            # ── Multi-asset grid — runs every 60s ─────────────────────────
            now = time.time()
            if now - _last_grid_check >= GRID_CHECK_INTERVAL:
                _last_grid_check = now
                for sym in GRID_SYMBOLS:
                    try:
                        fd = exchange.get_funding_data(sym)
                        price = fd.mark_price
                        grid = grids[sym]
                        sym_equity = equity / len(GRID_SYMBOLS)

                        if _stable_regime[sym] == 1:   # confirmed RANGE
                            if not grid.is_active:
                                log.info("%s grid OPEN (confirmed %dx) @ %.4f",
                                         sym, REGIME_CONFIRM_N, price)
                                grid.open(price, sym_equity)
                            else:
                                fills = grid.check(price, sym_equity)
                                for f in fills:
                                    log.info("%s grid fill | %s lvl=%+d  qty=%.4f  pnl=%+.2f",
                                             sym, f.side.upper(), f.level_idx, f.qty, f.pnl)
                                if fills:
                                    log.info(grid.status(price))
                        else:                           # confirmed TREND or CRISIS
                            if grid.is_active:
                                pnl = grid.close(price)
                                log.info("%s grid CLOSED (regime=%s, confirmed %dx) | pnl=%+.2f",
                                         sym, regime_name, REGIME_CONFIRM_N, pnl)

                    except Exception as e:
                        log.error("Grid error %s: %s", sym, e)

            # ── RL agent — PPO continuous leverage, 15m cadence ──────────
            if rl_executor is not None and (time.time() - _last_rl_step >= RL_STEP_INTERVAL):
                _last_rl_step = time.time()
                try:
                    btc_fd = exchange.get_funding_data("BTC")
                    btc_price = float(btc_fd.mark_price)
                    rl_equity = equity * args.rl_allocation
                    target_lev = rl_executor.step(btc_price, rl_equity)
                    log.info("RL step | BTC=%.2f | target_lev=%.2fx | equity_alloc=%.0f",
                             btc_price, target_lev, rl_equity)

                    # Daily reset at UTC midnight
                    import datetime as _dt
                    now_utc = _dt.datetime.utcnow()
                    if now_utc.hour == 0 and now_utc.minute < 20:
                        rl_executor.reset_daily(rl_equity)

                except Exception as e:
                    log.error("RL step error: %s", e)

        except Exception as e:
            log.error("Scan loop error: %s", e)

        time.sleep(min(args.poll_seconds, GRID_CHECK_INTERVAL))


if __name__ == "__main__":
    main()
