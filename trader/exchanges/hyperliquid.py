"""Hyperliquid perpetuals + spot exchange adapter.

Uses the official ccxt Hyperliquid integration.
Requires env vars: HL_PRIVATE_KEY, HL_WALLET_ADDRESS

Symbol convention: pass base asset only, e.g. "BTC", "ETH", "SOL".
Internally mapped to Hyperliquid perp format "BTC/USDC:USDC".
Spot format: "BTC/USDC" (see _SPOT_SYMBOL_MAP for exceptions).
"""
from __future__ import annotations

import os
import time

import ccxt

from .base import Balance, BracketParams, Candle, Exchange, FundingData, Order, Position
from trader.paper_portfolio import PaperPortfolio

# ccxt interval → Hyperliquid interval
_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}

# Assets available on Hyperliquid perps (verified active, top by OI as of 2026-04)
# Note: PEPE is listed as KPEPE (1000 PEPE contract) on Hyperliquid
SUPPORTED_SYMBOLS = [
    "BTC", "ETH", "SOL", "AVAX", "LINK", "ARB", "OP",
    "DOGE", "BNB", "XRP", "NEAR", "WIF", "KPEPE",
    "SUI", "HYPE", "INJ", "TIA", "SEI", "JUP",
    "MATIC", "ADA", "DOT", "LTC", "ATOM", "UNI",
]

# Perp symbols that have a liquid spot counterpart on Hyperliquid.
# Some use a different ticker (e.g. LINK0, BNB0) — handled by _SPOT_SYMBOL_MAP.
# Only symbols in this set can be fully delta-hedged.
SPOT_HEDGEABLE: frozenset[str] = frozenset(["BTC", "ETH", "SOL", "AVAX", "HYPE", "LINK", "BNB"])

# Maps perp base symbol → Hyperliquid spot ccxt symbol.
# Most are "<BASE>/USDC"; exceptions use a versioned ticker.
_SPOT_SYMBOL_MAP: dict[str, str] = {
    "BTC":  "BTC/USDC",
    "ETH":  "ETH/USDC",
    "SOL":  "SOL/USDC",
    "AVAX": "AVAX/USDC",
    "HYPE": "HYPE/USDC",
    "LINK": "LINK0/USDC",
    "BNB":  "BNB0/USDC",
}


def _hl_symbol(base: str) -> str:
    """Convert "BTC" → "BTC/USDC:USDC" (Hyperliquid perp format)."""
    return f"{base.upper()}/USDC:USDC"


def _spot_symbol(base: str) -> str:
    """Convert "BTC" → "BTC/USDC" (Hyperliquid spot format, with exceptions)."""
    return _SPOT_SYMBOL_MAP.get(base.upper(), f"{base.upper()}/USDC")


def _from_hl_symbol(hl_sym: str) -> str:
    """Convert "BTC/USDC:USDC" → "BTC"."""
    return hl_sym.split("/")[0]


class HyperliquidExchange(Exchange):
    """
    Hyperliquid perpetuals via ccxt.

    Paper-trading mode (no keys): set paper=True to skip order placement
    and just return simulated fills. Useful for dry-run testing.
    """

    @property
    def name(self) -> str:
        return "hyperliquid"

    def __init__(self, paper: bool = False, paper_equity: float = 10_000.0):
        self.paper = paper
        private_key = os.getenv("HL_PRIVATE_KEY", "")
        wallet = os.getenv("HL_WALLET_ADDRESS", "")

        self._client = ccxt.hyperliquid({
            "privateKey": private_key,
            "walletAddress": wallet,
            "options": {"defaultType": "swap"},
            "rateLimit": 200,        # 200ms between requests = 5 req/s
            "enableRateLimit": True,
        })

        # Separate spot client — Hyperliquid requires defaultType="spot" for spot orders
        self._spot_client = ccxt.hyperliquid({
            "privateKey": private_key,
            "walletAddress": wallet,
            "options": {"defaultType": "spot"},
            "rateLimit": 200,
            "enableRateLimit": True,
        })

        self._paper_portfolio: PaperPortfolio | None = None
        # Paper spot holdings: symbol → qty held (long only — we always buy spot to hedge)
        self._paper_spot: dict[str, float] = {}
        if paper and not wallet:
            self._paper_portfolio = PaperPortfolio(initial_equity=paper_equity)

        # Pre-load market metadata once so subsequent calls skip load_markets().
        # With backoff — if we're rate-limited at startup, wait before proceeding.
        for _attempt in range(4):
            try:
                self._client.load_markets()
                break
            except Exception as _e:
                if _attempt < 3:
                    time.sleep(2 ** _attempt * 5)   # 5, 10, 20, 40s

    def get_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        hl_sym = _hl_symbol(symbol)
        tf = _INTERVAL_MAP.get(interval, interval)
        rows = self._client.fetch_ohlcv(hl_sym, tf, limit=limit)
        return [
            Candle(
                timestamp=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
                quote_volume=float(r[5]) * float(r[4]),  # approx: volume * close
            )
            for r in rows
        ]

    def get_balance(self) -> Balance:
        if self._paper_portfolio is not None:
            prices = self._live_prices(list(self._paper_portfolio._positions.keys()))
            return self._paper_portfolio.get_balance(prices)
        bal = self._client.fetch_balance()
        total = float(bal.get("total", {}).get("USDC", 0))
        free = float(bal.get("free", {}).get("USDC", 0))
        return Balance(total_usd=total, available_usd=free)

    def get_positions(self) -> list[Position]:
        if self._paper_portfolio is not None:
            prices = self._live_prices(list(self._paper_portfolio._positions.keys()))
            return self._paper_portfolio.get_positions(prices)
        positions = self._client.fetch_positions()
        out: list[Position] = []
        for p in positions:
            if p.get("contracts", 0) == 0:
                continue
            side = "long" if p["side"] == "long" else "short"
            out.append(Position(
                symbol=_from_hl_symbol(p["symbol"]),
                side=side,
                quantity=float(p["contracts"]),
                entry_price=float(p.get("entryPrice") or 0),
                unrealized_pnl=float(p.get("unrealizedPnl") or 0),
                leverage=float(p.get("leverage") or 1),
            ))
        return out

    def place_bracket_order(self, params: BracketParams) -> Order:
        if self.paper:
            return self._paper_fill(params)

        hl_sym = _hl_symbol(params.symbol)
        order_type = "limit" if params.price else "market"
        price = params.price

        # Set leverage
        if params.leverage != 1:
            self._client.set_leverage(params.leverage, hl_sym)

        # Entry order
        entry = self._client.create_order(
            symbol=hl_sym,
            type=order_type,
            side=params.side,
            amount=params.quantity,
            price=price,
        )
        filled_price = float(entry.get("average") or entry.get("price") or price or 0)

        # TP order
        if params.take_profit_pct and filled_price:
            tp_side = "sell" if params.side == "buy" else "buy"
            tp_price = filled_price * (1 + params.take_profit_pct if params.side == "buy"
                                       else 1 - params.take_profit_pct)
            self._client.create_order(
                symbol=hl_sym,
                type="limit",
                side=tp_side,
                amount=params.quantity,
                price=tp_price,
                params={"reduceOnly": True},
            )

        # SL order (stop-market)
        if params.stop_loss_pct and filled_price:
            sl_side = "sell" if params.side == "buy" else "buy"
            sl_price = filled_price * (1 - params.stop_loss_pct if params.side == "buy"
                                        else 1 + params.stop_loss_pct)
            self._client.create_order(
                symbol=hl_sym,
                type="stop",
                side=sl_side,
                amount=params.quantity,
                price=sl_price,
                params={"reduceOnly": True, "stopPrice": sl_price},
            )

        return Order(
            order_id=entry["id"],
            symbol=params.symbol,
            side=params.side,
            quantity=params.quantity,
            price=filled_price,
            status="filled" if entry["status"] == "closed" else "open",
        )

    def close_position(self, symbol: str) -> None:
        if self._paper_portfolio is not None:
            prices = self._live_prices([symbol])
            self._paper_portfolio.close(symbol, prices.get(symbol, 0.0))
            return
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol:
                hl_sym = _hl_symbol(symbol)
                close_side = "sell" if p.side == "long" else "buy"
                self._client.create_order(
                    symbol=hl_sym,
                    type="market",
                    side=close_side,
                    amount=p.quantity,
                    params={"reduceOnly": True},
                )
                break

    def cancel_all_orders(self, symbol: str) -> None:
        self._client.cancel_all_orders(_hl_symbol(symbol))

    # ------------------------------------------------------------------ #
    # Funding rate methods (used by funding_arb strategy)                 #
    # ------------------------------------------------------------------ #

    def get_funding_data(self, symbol: str) -> FundingData:
        """Return current funding rate and mark/index prices for one symbol."""
        hl_sym = _hl_symbol(symbol)
        fr = self._client.fetch_funding_rate(hl_sym)
        rate_8h = float(fr.get("fundingRate") or fr.get("funding_rate") or 0.0)
        mark = float(fr.get("markPrice") or fr.get("mark_price") or 0.0)
        index = float(fr.get("indexPrice") or fr.get("index_price") or mark or 1.0)
        basis = (mark - index) / index * 100 if index else 0.0
        return FundingData(
            symbol=symbol,
            funding_rate_8h=rate_8h,
            funding_rate_annualized=rate_8h * 1095,  # 3 payments/day × 365
            mark_price=mark,
            index_price=index,
            basis_pct=basis,
        )

    def get_all_funding_data(self, symbols: list[str] | None = None) -> list[FundingData]:
        """Batch-fetch funding data; uses SUPPORTED_SYMBOLS if none given."""
        targets = symbols or SUPPORTED_SYMBOLS
        hl_syms = [_hl_symbol(s) for s in targets]
        try:
            batch = self._client.fetch_funding_rates(hl_syms)
        except Exception as _batch_exc:
            # Batch failed — check if it was a rate limit before spending time
            # on 25 individual calls.  If we're rate-limited, return [] immediately
            # so the caller's 600s sleep acts as a back-off window.
            _logger = __import__("logging").getLogger(__name__)
            _is_rl = (
                "429" in str(_batch_exc)
                or isinstance(_batch_exc, ccxt.RateLimitExceeded)
            )
            if _is_rl:
                _logger.warning(
                    "Batch funding fetch rate-limited — returning empty results. "
                    "Will retry after sleep interval."
                )
                return []

            # Non-rate-limit batch failure: fall back to per-symbol with single
            # attempt each.  Pre-load market metadata once to avoid per-symbol
            # load_markets() → fetch_currencies() calls.
            try:
                self._client.load_markets()
            except Exception as _e:
                _logger.warning("Could not pre-load markets: %s", _e)

            results = []
            for i, s in enumerate(targets):
                try:
                    results.append(self.get_funding_data(s))
                except Exception as e:
                    is_rate_limit = (
                        "429" in str(e) or isinstance(e, ccxt.RateLimitExceeded)
                    )
                    if is_rate_limit:
                        _logger.warning(
                            "Rate limited on %s during fallback — aborting scan.", s
                        )
                        return results   # return what we have; caller sleeps 600s
                    _logger.warning("Could not fetch %s: %s", s, e)
                if i < len(targets) - 1:
                    time.sleep(0.30)  # 300ms between symbols ~3 req/s
            return results

        results: list[FundingData] = []
        for sym, hl_sym in zip(targets, hl_syms):
            fr = batch.get(hl_sym, {})
            rate_8h = float(fr.get("fundingRate") or fr.get("funding_rate") or 0.0)
            mark = float(fr.get("markPrice") or fr.get("mark_price") or 0.0)
            index = float(fr.get("indexPrice") or fr.get("index_price") or mark or 1.0)
            basis = (mark - index) / index * 100 if index else 0.0
            results.append(FundingData(
                symbol=sym,
                funding_rate_8h=rate_8h,
                funding_rate_annualized=rate_8h * 1095,
                mark_price=mark,
                index_price=index,
                basis_pct=basis,
            ))
        return results

    def get_funding_rate_history(
        self, symbol: str, limit: int = 90
    ) -> list[tuple[int, float]]:
        """Return list of (timestamp_ms, rate_8h) for last `limit` 8-hour periods."""
        hl_sym = _hl_symbol(symbol)
        history = self._client.fetch_funding_rate_history(hl_sym, limit=limit)
        return [(int(r["timestamp"]), float(r["fundingRate"])) for r in history]

    def get_open_interest(self, symbol: str) -> float:
        """Return current open interest in base-asset contracts."""
        try:
            hl_sym = _hl_symbol(symbol)
            data = self._client.fetch_open_interest(hl_sym)
            return float(data.get("openInterestAmount") or data.get("openInterest") or 0.0)
        except Exception:
            return 0.0

    def get_predicted_funding(self, symbol: str) -> float:
        """Return Hyperliquid's predicted 1h funding rate (fraction)."""
        try:
            hl_sym = _hl_symbol(symbol)
            fr = self._client.fetch_funding_rate(hl_sym)
            # ccxt exposes 'nextFundingRate' for predicted funding when available
            predicted = fr.get("nextFundingRate") or fr.get("fundingRate") or 0.0
            return float(predicted)
        except Exception:
            return 0.0

    def get_l2_book(self, symbol: str, limit: int = 20) -> dict:
        """Return ccxt-style order book: {"bids": [[px, sz], ...], "asks": [...]}."""
        return self._client.fetch_order_book(_hl_symbol(symbol), limit=limit)

    def get_order_status(self, symbol: str, order_id: str) -> str:
        """Return order status string ('open', 'closed', 'canceled')."""
        try:
            o = self._client.fetch_order(order_id, _hl_symbol(symbol))
            return str(o.get("status", "open"))
        except Exception:
            return "open"

    def apply_paper_funding(self, symbol: str) -> float:
        """Accrue one funding payment for an open paper position. Returns amount paid/received."""
        if self._paper_portfolio is None:
            return 0.0
        try:
            fd = self.get_funding_data(symbol)
            return self._paper_portfolio.apply_funding(
                symbol, fd.funding_rate_8h, fd.mark_price
            )
        except Exception:
            return 0.0

    def paper_summary(self) -> str:
        """Return a formatted equity/position summary (paper mode only)."""
        if self._paper_portfolio is None:
            return "(live mode — no paper summary)"
        syms = list(self._paper_portfolio._positions.keys())
        prices = self._live_prices(syms)
        return self._paper_portfolio.summary(prices)

    def _live_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch current mid-price for each symbol. Falls back to last cached
        price on 429 so paper-portfolio equity doesn't flicker between
        marked-to-market and entry-price-fallback values."""
        if not hasattr(self, "_price_cache"):
            self._price_cache: dict[str, float] = {}
        prices: dict[str, float] = {}
        for sym in symbols:
            try:
                candles = self.get_candles(sym, "1m", 1)
                if candles:
                    prices[sym] = candles[-1].close
                    self._price_cache[sym] = candles[-1].close
            except Exception:
                # Use last known price rather than letting paper portfolio
                # fall back to entry_price (which causes equity to jump).
                if sym in self._price_cache:
                    prices[sym] = self._price_cache[sym]
        return prices

    # ------------------------------------------------------------------ #
    # Spot market methods (for delta-neutral hedging)                     #
    # ------------------------------------------------------------------ #

    def place_spot_order(self, symbol: str, side: str, quantity: float) -> Order:
        """
        Buy or sell `quantity` of `symbol` on Hyperliquid spot.

        side: "buy" (long spot hedge) | "sell" (close spot hedge)
        Returns a filled Order.

        In paper mode, tracks holdings in _paper_spot and simulates fill
        at the last known 1m close price.
        """
        if symbol not in SPOT_HEDGEABLE:
            raise ValueError(f"{symbol} has no liquid spot market on Hyperliquid")

        if self.paper:
            prices = self._live_prices([symbol])
            price = prices.get(symbol, 0.0)
            if side == "buy":
                self._paper_spot[symbol] = self._paper_spot.get(symbol, 0.0) + quantity
            else:
                self._paper_spot[symbol] = max(0.0, self._paper_spot.get(symbol, 0.0) - quantity)
            return Order(
                order_id=f"spot-paper-{int(time.time()*1000)}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                status="filled",
            )

        spot_sym = _spot_symbol(symbol)
        order = self._spot_client.create_order(
            symbol=spot_sym,
            type="market",
            side=side,
            amount=quantity,
        )
        filled_price = float(order.get("average") or order.get("price") or 0.0)
        return Order(
            order_id=order["id"],
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=filled_price,
            status="filled" if order.get("status") == "closed" else "open",
        )

    def close_spot_position(self, symbol: str) -> None:
        """Close (sell) the entire spot holding for `symbol`."""
        if self.paper:
            qty = self._paper_spot.pop(symbol, 0.0)
            if qty > 0:
                self.place_spot_order(symbol, "sell", qty)
            return

        spot_sym = _spot_symbol(symbol)
        try:
            bal = self._spot_client.fetch_balance()
            base = symbol if symbol not in ("LINK", "BNB") else symbol + "0"
            qty = float(bal.get("free", {}).get(base, 0))
            if qty > 0:
                self._spot_client.create_order(
                    symbol=spot_sym,
                    type="market",
                    side="sell",
                    amount=qty,
                )
        except Exception:
            pass

    def get_spot_holdings(self) -> dict[str, float]:
        """Return current spot holdings: {symbol: qty}. Paper mode reads _paper_spot."""
        if self.paper:
            return dict(self._paper_spot)
        try:
            bal = self._spot_client.fetch_balance()
            holdings: dict[str, float] = {}
            for sym in SPOT_HEDGEABLE:
                base = sym if sym not in ("LINK", "BNB") else sym + "0"
                qty = float(bal.get("free", {}).get(base, 0))
                if qty > 0:
                    holdings[sym] = qty
            return holdings
        except Exception:
            return {}

    def _paper_fill(self, params: BracketParams) -> Order:
        """Simulate an immediate fill via paper portfolio (tracks positions + equity)."""
        prices = self._live_prices([params.symbol])
        price = prices.get(params.symbol) or params.price or 0.0

        if self._paper_portfolio is not None:
            side_mapped = "long" if params.side == "buy" else "short"
            return self._paper_portfolio.open(
                symbol=params.symbol,
                side=side_mapped,
                qty=params.quantity,
                price=price,
                leverage=params.leverage,
            )

        return Order(
            order_id=f"paper-{int(time.time()*1000)}",
            symbol=params.symbol,
            side=params.side,
            quantity=params.quantity,
            price=price,
            status="filled",
        )
