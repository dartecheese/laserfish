# laserfish

5-minute perpetual scalper on Hyperliquid. Two strategies, one repo:

1. **Momentum** (`scripts/run.py`) — cross-sectional 1h momentum on 5m bars with funding-rate gating. Runs immediately, no training required.
2. **Transformer** (`scripts/trade.py`) — ONNX classifier trained with triple-barrier labels on volume bars. Needs `fetch.py` + `train.py` first.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env

# Paper trade (no keys needed)
python scripts/run.py

# Live trading
HL_PRIVATE_KEY=0x... HL_WALLET_ADDRESS=0x... python scripts/run.py --live
```

## Train the Transformer

```bash
# 1. Fetch 5m data (~3 years, ~500k bars per symbol)
python scripts/fetch.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,AVAXUSDT,LINKUSDT

# 2. Train + export ONNX
python scripts/train.py --symbols BTC,ETH,SOL,AVAX,LINK

# 3. Run Transformer agent
python scripts/trade.py --symbols BTC,ETH,SOL         # paper
HL_PRIVATE_KEY=0x... HL_WALLET_ADDRESS=0x... \
python scripts/trade.py --symbols BTC,ETH,SOL --live  # live
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--symbols` | `BTC,ETH,SOL,...` | Perp symbols to trade |
| `--leverage` | `2` | Position leverage |
| `--top-n` | `3` | Max simultaneous longs and shorts |
| `--z-entry` | `1.2` | Momentum z-score entry threshold |
| `--max-position-pct` | `0.20` | Max equity per position |
| `--poll-seconds` | `300` | Scan interval (one 5m candle) |
| `--live` | off | Place real orders |

## Strategy details

**Momentum**
- Signal: 1h return (12 × 5m) z-scored vs 6h vol window (72 × 5m)
- Funding gate: veto longs/shorts when funding is anomalously charged against the trade
- TP: 3% / SL: 2%

**Transformer**
- Features: 10-dim per bar (log-return, range, vwap-dev, vol z-scores, RSI, momentum)
- Sequence: 32 volume bars → triple-barrier label (+1 / 0 / -1)
- TP: 1.5% / SL: 1.5% / max hold: 12 bars (1h)

## Env

```
HL_PRIVATE_KEY=0x...
HL_WALLET_ADDRESS=0x...
```

## Symbol coverage

25 Hyperliquid perps: `BTC ETH SOL AVAX LINK ARB OP DOGE BNB XRP NEAR WIF KPEPE SUI HYPE INJ TIA SEI JUP MATIC ADA DOT LTC ATOM UNI`

> PEPE trades as `KPEPE` (1000 PEPE contract) on Hyperliquid.
