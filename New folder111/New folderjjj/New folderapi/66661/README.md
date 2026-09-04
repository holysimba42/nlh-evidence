# Freebuff FX: Free Data → Validated Edge → Auto-Trading (OANDA Demo)

A production-grade, cost-aware FX system that turns **free OANDA practice-API data**
(candles + pricing cost $0 on demo) into **walk-forward-validated trade signals**,
executed automatically on your OANDA demo account with hard risk limits.

📘 **The full method — the process this repo implements, teachable end-to-end — is
in [`docs/TEXTBOOK.md`](docs/TEXTBOOK.md).** The summary below is the short version.

```
free OANDA stream/REST ─▶ SQLite store ─▶ causal features ─▶ ensemble model
        ▲                                                      │ score in [0,1]
        │                                                      ▼
 close OCO trades ◀── market+OCO order ──── risk gate ──── threshold decision
 (ATR stop/TP)              (demo: free fills)
```

## The teaching core: how free data streams become an edge

**1. Data (free).** OANDA practice endpoints give unlimited M5 candles and live
price streams for $0. We store completed bars only in SQLite (`trader/store.py`).

**2. Features (causal, seen data).** `trader/features.py` builds 25 features from
data ≤ t: multi-horizon log returns, ATR, RSI, stochastic, MACD, Bollinger
position/width, trend efficiency (Kaufman), ADX-proxy, volume z-score, and UTC
calendar encodings. **The repo's test suite PROVES causality**: mutating future
bars cannot change past features (`test_features_are_causal`).

**3. Labels (the unseen).** Each bar is labeled with the *future* 12-bar return in
bps. The model's job: given only causal features, estimate P(price rises over the
next hour). That probability, net of costs, IS the edge.

**4. Ensemble (two uncorrelated learners).** Logistic regression (robust linear)
+ capped gradient boosting (non-linear). Averaging reduces variance → higher
Sharpe than either alone.

**5. Walk-forward validation (honest performance).** `trader/walkforward.py`
trains on 2000 bars, predicts the NEXT 500 bars it has never seen, steps forward,
repeats. Threshold is tuned per window on train data ONLY by maximizing net PnL
after costs — we optimize the objective directly. The concatenated OOS result is
the only Sharpe/profit number anyone should believe.

**6. Execution with honest frictions.** Signals on bar t's close fill at bar
t+1's open, pay spread + slippage in and out, and assume the stop hits before TP
when both fall inside one bar. If an edge survives that gauntlet, it's real.

**7. Risk envelope (the non-negotiables).** Fixed-fractional sizing
(0.4% equity per trade between entry and stop), 5x leverage cap, 1.5% daily loss
halt, 30-minute post-loss cooldown, spread filter, per-instrument caps, weekend
gap avoidance, and a file-based kill switch (`data/KILL` → flatten + stand down).

## Quick start (5 minutes, $0)

```bash
pip install -r requirements.txt
python -m trader.run doctor           # preflight checklist — needs your FREE practice credentials
cp .env.example .env          # paste your FREE practice token + account id
python -m trader.run doctor           # re-run until ALL CHECKS PASSED
python -m trader.run backtest --all   # walk-forward validation (API candles; artifacts land in ./artifacts/)
python -m trader.run live --dry-run   # simulated fills, persistent paper equity curve, zero orders
python -m trader.run report --dryrun  # artifacts for the paper session: fills + equity curve
python -m trader.run live --execute   # live demo orders (DRY_RUN=true default keeps paper mode)
```

**Account note:** candles, pricing, and order placement all use the OANDA practice
API, so credentials are required for `backtest` and `live` (they are free). The
doctor tells you exactly what to paste where before anything else runs.

## Go-live checklist (demo → confidence)

1. `python -m trader.run doctor` → ALL CHECKS PASSED
2. `backtest --all` OOS Sharpe > 0 and PF > 1.1 across instruments; **review the
   artifacts in `./artifacts/`** (per-window table, trades CSV, equity curve) —
   stdout alone is not an audit
3. `live --dry-run` for ≥ 3 trading days, then `python -m trader.run report --dryrun`:
   judge the **rendered fills + equity artifacts in `./artifacts/`** (same format
   as backtest artifacts), not the raw JSON and not vibes
4. `live` on demo for ≥ 2 weeks; equity curve steady, no > 1.5% day loss
5. Only then consider real funds — set `OANDA_ENVIRONMENT=live` + `ALLOW_LIVE=true`
   (two-key interlock) — and re-verify costs (live spreads > demo spreads)

## Files

| Path | Role |
|---|---|
| `trader/config.py` | env-driven typed config + environment interlock |
| `trader/oanda_client.py` | free REST client, orders with OCO, retries (fail-fast 4xx) |
| `trader/store.py` | SQLite candles + signal log |
| `trader/features.py` | causal feature engineering (+ leakage-proof test) |
| `trader/signal.py` | regime-aware logistic+GBM ensemble, decide() |
| `trader/backtest.py` | next-bar, cost-honest event backtester |
| `trader/walkforward.py` | rolling OOS validation + PnL-based threshold tuning |
| `trader/risk.py` | hard gates: sizing, daily halt, cooldown, correlated-exposure cap, kill switch |
| `trader/broker.py` | dry-run simulated broker: spread-adjusted fills, persistent paper equity |
| `trader/doctor.py` | preflight checklist (implicit before backtest/live) |
| `trader/artifacts.py` | per-window table, trades CSV, equity curve per run |
| `trader/live.py` | production loop: refresh→evaluate→gate→order, retrain weekly |
| `trader/run.py` | CLI: `doctor` / `backtest` / `live` / `report` |
| `docs/TEXTBOOK.md` | the full teaching document: process, cost math, runbook |
| `tests/` | 33 tests incl. the leakage proof, live-path regressions, hostile-input handling |

## Honest limitations (read this, seriously)

- **Demo ≠ live spreads.** We charge 1.6 pips round-trip (spread 1.0 + slippage
  0.3 per side, config-tunable); re-verify against real fills after go-live.
- **Regime shift risk.** Walk-forward mitigates but cannot eliminate. The daily
  halt + kill switch are your structural defense.
- **No system "guarantees" highest Sharpe.** This one maximizes the *probability*
  of a positive OOS Sharpe: causal features, cost-aware labels, ensemble variance
  reduction, direct PnL objective, hard risk envelope. That is the correct
  production posture.
- M5 horizon=12 (≈1h hold) is a sane default; re-run walk-forward per instrument
  before changing it.
