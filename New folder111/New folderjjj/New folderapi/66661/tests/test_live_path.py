"""Regression tests for live-path defects found by adversarial probing:
bar dedup, per-instrument models, decide() contract, 4xx fail-fast,
closure reconciliation (incl. retry-on-unpriceable ordering)."""
import numpy as np
import pandas as pd
import pytest

from trader.config import Config
from trader.live import LiveEngine
from trader.oanda_client import OandaClient, OandaClientError
from trader.signal import build_inference_frame, build_dataset, EnsembleSignal


def synth(n, seed=3, base=1.10):
    rng = np.random.default_rng(seed)
    raw = pd.date_range("2026-08-03", periods=n * 2, freq="5min")
    t = raw[np.asarray(raw.dayofweek < 5)][:n]
    c = base + np.cumsum(rng.normal(0, 0.00008, n))
    return pd.DataFrame({
        "time": t, "open": np.r_[c[0], c[:-1]],
        "high": c + np.abs(rng.normal(0, 0.00025, n)),
        "low": c - np.abs(rng.normal(0, 0.00025, n)),
        "close": c, "volume": rng.integers(50, 400, n)})


class FakeClient:
    def open_trades(self):
        return []

    def get_pricing(self, inst):
        return [{"bids": [{"price": "1.1000"}], "asks": [{"price": "1.1002"}]}]


@pytest.fixture
def engine(tmp_path):
    cfg = Config(db_path=str(tmp_path / "probe.db"),
                 telemetry_path=str(tmp_path / "tel.jsonl"),
                 wf_train_bars=400, signal_threshold=0.55)
    eng = LiveEngine(cfg)
    eng.client = FakeClient()
    return eng


def test_retrain_partial_bootstrap_and_per_instrument_models(engine):
    """Instruments missing from cache must not crash retrain; models must be
    per-instrument (JPY scale never leaks into a EUR model)."""
    engine.candles = {"EUR_USD": synth(700), "USD_JPY": synth(700, seed=99, base=155.0)}
    engine.retrain()
    assert set(engine.sigs) == {"EUR_USD", "USD_JPY"}
    assert engine.sigs["EUR_USD"] is not engine.sigs["USD_JPY"]


def test_same_bar_evaluated_once(engine):
    """Two polls on the same completed bar -> exactly one signals_log row."""
    df = synth(700)
    engine.candles = {"EUR_USD": df}
    engine.evaluate("EUR_USD")
    engine.evaluate("EUR_USD")
    n = engine.store.conn.execute("SELECT COUNT(*) FROM signals_log").fetchone()[0]
    assert n == 1


def test_decide_contract(engine):
    """decide() takes exactly one inference row; rejects other shapes loudly."""
    frame = build_inference_frame(synth(700))
    sig = EnsembleSignal(threshold=0.55).fit(build_dataset(synth(700), horizon=12))
    action, score, regime = sig.decide(frame.iloc[[-1]])
    assert action in {"LONG", "SHORT", "FLAT"} and 0.0 <= score <= 1.0
    with pytest.raises(ValueError):
        sig.decide(frame.iloc[[-1, -2]])


def test_4xx_fails_fast_no_retry():
    """A 400 must raise on the first attempt — no retry, no backoff sleep."""
    class FakeResp:
        status_code, text, content, headers = 400, "x", b"x", {}

    class FakeSession:
        calls = 0

        def request(self, *a, **k):
            FakeSession.calls += 1
            return FakeResp()

    c = OandaClient("tok", "acc", "https://x.invalid")
    c.session = FakeSession()
    with pytest.raises(OandaClientError):
        c._request("GET", "https://x.invalid/v3/x")
    assert FakeSession.calls == 1


def test_reconcile_closures_updates_risk_state(engine):
    """A closed trade must feed realized PnL into the daily-halt accounting."""
    kt = engine.risk.state.known_trades
    kt["777"] = {"instrument": "EUR_USD", "entry": 1.1000, "units": 10_000, "direction": 1}
    engine.reconcile_closures()
    assert engine.risk.state.realized_today == pytest.approx(1.00)  # 10k units * 1 pip
    assert kt == {}


def test_unpriceable_closure_retries_not_dropped(engine):
    """If pricing is unavailable, the closure record must be RETAINED and
    reconciled later — not silently dropped from risk accounting."""
    kt = engine.risk.state.known_trades
    kt["888"] = {"instrument": "ZZZ_XXX", "entry": 1.0, "units": 1, "direction": 1}

    class NoPriceClient(FakeClient):
        def get_pricing(self, inst):
            raise RuntimeError("pricing unavailable")

    engine.client = NoPriceClient()
    engine.reconcile_closures()
    assert "888" in kt, "unpriceable closure was dropped — PnL lost"
    engine.client = FakeClient()  # pricing recovers
    engine.reconcile_closures()
    assert kt == {}
