# The Textbook: Free Forex Data Streams → Validated Trade Signals → OANDA Demo Execution

This document transfers the exact process implemented in this repository. A reader
(human or model) with no access to any prior conversation should be able to
reproduce the method and operate the system from this file alone. Every principle
cites the code that embodies it; every command and key was verified against the
current implementation.

**What is guaranteed and what is not.** Guaranteed: *process integrity* — no
lookahead anywhere, every trade charged real costs, reported performance is
out-of-sample only, and hard risk gates run before every order (all enforced by
construction and by the test suite). Not guaranteed: *profitability*. No method
can promise a positive Sharpe; this process maximizes the probability that any
edge which exists will be found honestly and not blown up by costs, overfitting,
or unbounded risk.

The running example is M5 bars (5-minute candles), horizon 12 bars (≈1 hour),
on five majors: EUR_USD, GBP_USD, USD_JPY, AUD_USD, USD_CAD.

---

## 1. Free streams → completed-bar storage

OANDA's practice API provides candles and pricing for free. The pipeline uses
REST candle polling (`trader/oanda_client.py::OandaClient.get_candles`), keeping
only candles with `complete=True`.

**Principle: a half-formed bar is future information.** The close of the bar
in progress will change until the bar completes; any feature computed on it
peeks forward in time. The store therefore keeps completed bars only
(`trader/store.py`, primary key `(instrument, granularity, time)` with
`INSERT OR REPLACE` making ingestion idempotent), and the live loop fetches
only bars newer than the last cached one
(`trader/live.py::LiveEngine.refresh`, `from_time = last_bar + 1s`).

Storage is deliberately boring: local SQLite, no external services, zero cost.

## 2. Causal features — the leakage invariant

`trader/features.py::compute_features` builds 25 features from data at time ≤ t:
multi-horizon log returns (1/3/6/12/24), intra-bar structure (hl_range,
close_loc), ATR, RSI, stochastic, MACD, Bollinger position/width, Kaufman trend
efficiency, an ADX proxy, volume z-score, and UTC calendar encodings (known in
advance, hence causal). `FEATURES` is the single source of truth for the input
contract.

**The invariant:** for every row t, the feature vector must be a function of
bars 0..t only. This is *proved*, not asserted:
`tests/test_pipeline.py::test_features_are_causal` mutates the last 50 bars of a
price series and asserts the feature values of earlier rows are bit-identical.
If a new feature ever breaks this test, the whole backtest is invalid — that
test is the contract.

**One owner of the inference path.** `trader/signal.py::build_inference_frame`
produces model-ready rows (features + regime); `build_dataset` is exactly that
plus the label. Live and training therefore cannot skew: they call the same
constructor. A past defect (live code rebuilding features by hand, missing the
regime column, crashing the loop) is regression-tested in
`tests/test_live_path.py`.

## 3. Cost-adjusted labels — the hurdle an edge must clear

The label at bar t is the **sign-aligned future return over the next 12 bars,
in basis points** (`trader/features.py::future_return`):
`label(t) = (close(t+12)/close(t) − 1) × 10⁴`. The model's task: given only
causal features, estimate P(label > 0) — "does LONG pay from here?"

**The hurdle math.** The system charges every simulated trade, per side,
`(spread/2 + slippage)` pips (`trader/backtest.py::run_backtest`), with defaults
`DEFAULT_SPREAD_PIPS=1.0`, `SLIPPAGE_PIPS=0.3`:

- **EUR_USD** (pip = 0.0001): round trip = 2 × (0.5 + 0.3) = **1.6 pips** =
  0.00016 ≈ 1.45 bps at 1.10. A 1-hour expected move must beat ~1.5 bps before
  a signal earns anything.
- **USD_JPY** (pip = 0.01 — JPY pairs use a larger pip, `trader/backtest.py::pip_size`):
  1.6 pips = 0.016 ≈ 1.03 bps at 155.00. Same idea, different pip size — the
  per-pair conversion matters, and getting it wrong silently corrupts JPY risk
  accounting (a real defect class, covered by tests).

This is why the label lives in bps and why the decision threshold is tuned on
**net PnL after costs** (§5), not on accuracy: accuracy treats 1.6 pips of cost
as free, which is exactly the lie that makes naive backtests profitable.

Execution honesty (`trader/backtest.py`): the signal fires on bar t's close,
fills at bar t+1's open (one-bar delay — no lookahead), and when a bar's range
could hit both stop and take-profit, the **stop is assumed first**.

## 4. Ensemble → variance reduction → Sharpe logic

`trader/signal.py::make_models` defines two deliberately different learners:

- a **Pipeline**(StandardScaler → LogisticRegression(C=0.1)) — linear, low variance;
- a **GradientBoostingClassifier**(depth 2, 150 trees, subsample 0.8) — non-linear, capped capacity.

The ensemble score is the **mean of their predicted probabilities**. Averaging
two weakly-correlated predictors reduces variance; lower variance of returns at
similar mean return is, mechanically, a higher Sharpe. No single model is
trusted alone, and capacity is capped everywhere because financial data has a
poor signal-to-noise ratio — the failure mode is overfitting, not underfitting.

A regime veto (`classify_regime` + `EnsembleSignal.decide`) refuses marginal
signals in choppy, range-bound conditions — the regime features are themselves
causal, so the veto leaks nothing.

## 5. Walk-forward validation — only concatenated OOS numbers count

`trader/walkforward.py::walk_forward` is the only performance authority:

1. Train on 2,000 bars (features + labels from *seen* data).
2. Predict the next **500 bars the model has never seen**.
3. Step forward 500 bars, retrain, repeat. Concatenate all predictions.

The decision threshold is tuned per window, **inside the train slice only**
(THRESHOLDS = 0.50/0.55/0.60/0.65), by running the costed backtest on an inner
train split and picking the threshold with the best net PnL — the objective is
optimized directly, on data the test slice never influenced.

Anything computed on in-sample data (train accuracy is logged for telemetry
only) is not evidence. "OOS Sharpe" means: computed over the concatenated
never-seen slices, with costs charged, with the threshold the *past* chose. A
per-window table of these numbers is written to `./artifacts/` on every run
(`trader/artifacts.py::write_artifacts`): windows CSV (train/test bounds,
threshold, OOS trades, OOS net PnL, OOS Sharpe), all-trades CSV, and the equity
curve (CSV; PNG when matplotlib is installed).

## 6. The risk envelope — what makes the target maintainable

High accuracy and a good backtest mean nothing if one day can end the account.
`trader/risk.py::RiskManager.check_entry` gates every order, in order:

| Gate | Default (config key) | Purpose |
|---|---|---|
| Kill switch file | `data/KILL` exists → flatten + stand down | instant human override |
| Daily loss halt | 1.5% of day-start equity (`MAX_DAILY_LOSS_PCT`) | no recovery trading |
| Profit lock (optional) | `DAILY_PROFIT_LOCK_PCT` | bank a good day |
| Per-instrument cap | 1 trade (`MAX_TRADES_PER_INSTRUMENT`) | no doubling down |
| Concurrent trades | 3 (`MAX_OPEN_TRADES`) | breadth limit |
| Loss cooldown | 30 min per instrument (`COOLDOWN_AFTER_LOSS_MINUTES`) | break tilt/revenge loops |
| Spread filter | 2.0 pips (`MAX_SPREAD_PIPS`) | don't pay 3× the modeled cost |
| Position sizing | 0.4% equity / stop distance (`RISK_PER_TRADE_PCT`) | fixed-fractional, per-trade |
| Leverage cap | 5× equity notional (`MAX_LEVERAGE`) | sizing never exceeds the cap |
| **Correlated exposure** | 6.0× equity per currency (`MAX_CCY_EXPOSURE_X`) | no hidden dollar bets |

The correlated-exposure cap (`trader/instruments.py::legs_usd`, summed per
`trader/risk.py::_exposure_map`) accumulates
signed per-currency exposure across all open trades. Three long-USD pairs are a
3× dollar bet, not "3 positions". JPY correctness is explicit: USD_JPY legs are
normalized to USD (100k units = 100k USD of exposure, not 15.5M JPY), and the
cap must exceed the leverage cap so a single full-size position can still open —
the guard blocks *stacking*, not *trading*.

Closures are reconciled each cycle (`trader/live.py::reconcile_closures`): OCO
fills detected by diffing open trades, realized PnL fed back into the daily-halt
and cooldown state. Fill records that cannot yet be priced are **retained and
retried**, never dropped (tested in `tests/test_live_path.py`).

In dry-run the same gates run against the simulated portfolio
(`trader/broker.py::DryRunBroker.sync_risk`), so the paper account obeys the
identical risk law as the demo account.

## 7. Go-live runbook

All commands verified. Practice ("demo") endpoints are free; credentials come
from the OANDA practice console → Manage API Access.

```bash
pip install -r requirements.txt

# 1. Preflight. Exit code 2 on any FAIL; prints exactly what to paste where.
python -m trader.run doctor            # also runs implicitly before backtest/live

# 2. Credentials: cp .env.example .env, then edit:
#    OANDA_TOKEN=<32+ hex token>   OANDA_ACCOUNT_ID=<999-999-9999999-999>
python -m trader.run doctor            # re-run until ALL CHECKS PASSED

# 3. Honest performance on unseen data; writes ./artifacts/ per instrument.
python -m trader.run backtest --all            # or --instrument EUR_USD
#    Judge: windows_*.csv OOS Sharpe/PnL per window; trades_*.csv; equity_*.csv.

# 4. Paper trading. DRY_RUN defaults to true; simulated fills are spread-adjusted,
#    OCO-marked per bar, and the equity accumulates across sessions in
#    DRYRUN_STATE_PATH (default data/dryrun_state.json).
python -m trader.run live --dry-run            # run ≥ 3 trading days
python -m trader.run report --dryrun           # artifacts for the paper session: fills + equity curve

# 5. Demo orders (free practice account; OCO attached server-side).
python -m trader.run live --execute
```

**Judging each stage, in order — do not skip ahead:**

1. **Doctor:** every check PASS. A traceback here is a bug; a FAIL is an instruction.
2. **Backtest artifacts:** OOS Sharpe > 0 and profit factor > 1.1 across
   instruments, with no single window carrying everything. Files, not stdout,
   are the audit trail.
3. **Dry-run:** `python -m trader.run report --dryrun` renders the accumulated
   simulated fills and equity curve into `./artifacts/` (same format as backtest
   artifacts) — judge those files, which apply the identical risk gates and cost
   model, not the count of signals logged.
4. **Demo live:** ≥ 2 weeks, steady equity, no day worse than the halt threshold.
5. Only then consider real funds — and re-verify costs first: live spreads are
   wider than demo spreads. `OANDA_ENVIRONMENT=live` additionally requires
   `ALLOW_LIVE=true` (two-key interlock, `trader/config.py::Config.api_base`).

---

### File map (principle → code)

| Principle | Code |
|---|---|
| Completed bars only | `trader/oanda_client.py::get_candles`, `trader/store.py`, `trader/live.py::refresh` |
| Causal features + proof | `trader/features.py::compute_features`, `tests/test_pipeline.py::test_features_are_causal` |
| Cost-adjusted labels & hurdle | `trader/features.py::future_return`, `trader/backtest.py::run_backtest` |
| Ensemble & regime veto | `trader/signal.py::make_models`, `EnsembleSignal.decide` |
| Walk-forward, OOS-only | `trader/walkforward.py::walk_forward` (validation), `run_instrument_validation` (orchestration) |
| Risk envelope | `trader/risk.py::check_entry`, `legs_usd`, `trader/live.py::reconcile_closures` |
| Dry-run simulation | `trader/broker.py::DryRunBroker` |
| Instrument conventions (pip size, exposure legs) | `trader/instruments.py` |
| Preflight | `trader/doctor.py::run_doctor` |
| Artifacts | `trader/artifacts.py::write_artifacts` |

Test suite: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` — 33 tests,
including the leakage proof and regression tests for every defect the
adversarial passes ever found in the live path.
