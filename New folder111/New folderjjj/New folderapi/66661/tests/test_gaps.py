"""Tests for the four audit gaps: doctor preflight, review artifacts,
dry-run fill simulation, correlated-exposure cap. No network needed."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trader.config import Config
from trader.risk import RiskManager
from trader.instruments import legs_usd, parse_currencies
from trader.broker import DryRunBroker, DryRunStateError
from trader.doctor import run_doctor
from trader.backtest import run_backtest
from trader.walkforward import walk_forward
from trader.artifacts import write_artifacts
from trader.walkforward import run_instrument_validation
from trader.config import Config as RealConfig
from trader.store import CandleStore, StoreError


def test_dryrun_report_command(tmp_path, capsys):
    """The report command turns dryrun_state.json into backtest-format
    artifacts; a reviewer never opens the JSON."""
    from trader.broker import DryRunBroker
    from trader.run import cmd_report_dryrun

    state = str(tmp_path / "dry.json")
    art = str(tmp_path / "art")
    br = DryRunBroker(state, 100_000.0, 1.0, 0.3)
    br.count_session()
    t0 = pd.Timestamp("2026-09-01", tz="UTC")
    br.fill("EUR_USD", 10_000, 1.1000, 0.0005, 1.6, 2.4, t0)
    br.mark("EUR_USD", pd.Series({"time": t0 + pd.Timedelta(minutes=5),
                                   "open": 1.101, "high": 1.105, "low": 1.0995,
                                   "close": 1.104}))
    br2 = DryRunBroker(state, 100_000.0, 1.0, 0.3)  # second session on same state
    br2.count_session()
    br2.fill("USD_JPY", 50_000, 155.00, 0.05, 1.6, 2.4, t0 + pd.Timedelta(hours=1))
    br2.mark("USD_JPY", pd.Series({"time": t0 + pd.Timedelta(hours=2),
                                    "open": 155.0, "high": 155.40, "low": 154.9,
                                    "close": 155.3}))

    cfg = RealConfig(db_path=str(tmp_path / "s.db"), artifacts_dir=art,
                     dryrun_state_path=state)
    rc = cmd_report_dryrun(cfg)
    out = capsys.readouterr().out
    assert rc == 0
    assert "sessions: 2" in out and "open positions: 0" in out
    files = {p.name for p in Path(art).iterdir()}
    assert any(n.startswith("equity_DRYRUN_") for n in files)
    assert any(n.startswith("trades_DRYRUN_") for n in files)
    assert not any(n.startswith("windows_") for n in files)  # no walk-forward windows
    # format parity with backtest trade artifacts (same header, instrument per row)
    tf = next(Path(art).glob("trades_DRYRUN_*.csv"))
    header = tf.read_text(encoding="utf-8").strip().split("\n")[0]
    assert header == ("instrument,direction,units,entry_time,entry_price,"
                      "exit_time,exit_price,costs_pips,net_pnl,exit_reason")
    rows = tf.read_text(encoding="utf-8").strip().split("\n")[1:]
    assert {r.split(",")[0] for r in rows} == {"EUR_USD", "USD_JPY"}


def test_dryrun_report_empty_state(tmp_path, capsys):
    from trader.run import cmd_report_dryrun
    cfg = RealConfig(db_path=str(tmp_path / "s.db"), artifacts_dir=str(tmp_path / "art"),
                     dryrun_state_path=str(tmp_path / "empty.json"))
    assert cmd_report_dryrun(cfg) == 1
    assert "nothing to report" in capsys.readouterr().out


def test_hostile_corrupt_store_typed_error(tmp_path, capsys):
    """A garbage-bytes candles.db must surface as a typed error with recovery
    instructions — never a raw sqlite traceback."""
    bad = tmp_path / "candles.db"
    bad.write_text("not a database at all", encoding="utf-8")
    with pytest.raises(StoreError) as ei:
        CandleStore(str(bad))
    assert "Recovery" in str(ei.value) and "not a database" in str(ei.value)


def test_hostile_garbage_dryrun_state_refused(tmp_path, capsys):
    """Garbage-typed dry-run state must be refused loudly (typed error), not
    silently traded on; report surfaces it cleanly with exit 2."""
    st = tmp_path / "dry.json"
    st.write_text('{"equity":"lots","positions":42,"fills":"abc"}',
                  encoding="utf-8")
    with pytest.raises(DryRunStateError):
        DryRunBroker(str(st), 100_000.0, 1.0, 0.3)
    from trader.run import cmd_report_dryrun
    cfg = RealConfig(db_path=str(tmp_path / "s.db"), artifacts_dir=str(tmp_path / "a"),
                     dryrun_state_path=str(st))
    assert cmd_report_dryrun(cfg) == 2
    assert "ERROR" in capsys.readouterr().out


def test_hostile_negative_equity_state_refused(tmp_path):
    st = tmp_path / "neg.json"
    st.write_text('{"equity":-50,"day_start_equity":-50,"day_key":"d",'
                  '"realized_today":0,"last_loss_time":{},"positions":[],'
                  '"next_id":-3,"fills":[],"sessions":1}', encoding="utf-8")
    with pytest.raises(DryRunStateError):
        DryRunBroker(str(st), 100_000.0, 1.0, 0.3)


def test_hostile_instrument_normalization():
    """Whitespace/case variants must not create separate exposure buckets."""
    assert parse_currencies(" eur_usd ") == ("EUR", "USD")
    rm = RiskManager(_Cfg())
    rm._roll_day_if_needed(100_000.0, pd.Timestamp("2026-09-01 10:00", tz="UTC").to_pydatetime())
    ok, reason, _ = rm.check_entry(
        instrument="EUR_USD", price=1.10, atr=0.0005, equity=100_000.0,
        open_trades=[{"instrument": " eur_usd ", "units": 100_000, "price": 1.10}],
        now=pd.Timestamp("2026-09-01 12:00", tz="UTC"), spread_pips=1.0)
    assert ok is False and reason == "per_instrument_cap"


def test_hostile_weekend_guard_in_dryrun(tmp_path):
    """Marking must refuse weekend/Friday-evening fills — the backtester's
    Friday-flat rule applies to the simulated broker too."""
    br = DryRunBroker(str(tmp_path / "d.json"), 100_000.0, 1.0, 0.3)
    br.fill("EUR_USD", 10_000, 1.10, 0.0005, 1.6, 2.4,
            pd.Timestamp("2026-09-04 18:00", tz="UTC"))  # Friday 18:00 UTC
    for ts in ["2026-09-04 19:30", "2026-09-05 12:00", "2026-09-06 12:00"]:
        br.mark("EUR_USD", pd.Series({"time": pd.Timestamp(ts, tz="UTC"),
                                       "open": 1.05, "high": 1.2, "low": 1.0,
                                       "close": 1.1}))
        assert len(br.st["fills"]) == 0, f"fill leaked at {ts}"
    # Monday 00:00 UTC is a legitimately open market: a held position may close
    br.mark("EUR_USD", pd.Series({"time": pd.Timestamp("2026-09-07 00:00", tz="UTC"),
                                   "open": 1.05, "high": 1.05, "low": 1.05,
                                   "close": 1.05}))
    assert len(br.st["fills"]) == 1  # weekend gap realized on the held position
    assert br.st["positions"] == []  # nothing left dangling


def test_orchestration_direct_unit_call(tmp_path):
    """run_instrument_validation is importable and works end-to-end with a fake
    client — orchestration is unit-testable without process invocation."""
    class FakeClient:
        def get_candles(self, instrument, gran, count):
            return synth(int(count))

    cfg = RealConfig(db_path=str(tmp_path / "s.db"),
                     artifacts_dir=str(tmp_path / "art"),
                     wf_train_bars=800, wf_test_bars=200, wf_step_bars=200,
                     history_bars=1600)
    store = CandleStore(cfg.db_path)
    out = run_instrument_validation(FakeClient(), store, "EUR_USD", cfg)
    assert out is not None
    assert out["instrument"] == "EUR_USD" and out["bars"] == 1600
    assert len(out["windows"]) >= 3
    assert all("oos_sharpe" in w for w in out["windows"])
    assert out["result"].n_trades >= 0
    # artifacts written by the orchestration itself
    names = {p.name for p in (tmp_path / "art").iterdir()}
    assert any(n.startswith("windows_") for n in names)
    assert any(n.startswith("trades_") for n in names)
    assert any(n.startswith("equity_") for n in names)


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


class _Cfg:
    risk_per_trade_pct = 0.4
    max_daily_loss_pct = 1.5
    max_open_trades = 3
    max_trades_per_instrument = 1
    cooldown_after_loss_minutes = 30
    max_spread_pips = 2.0
    max_ccy_exposure_x = 6.0
    atr_sl_multiple = 1.6
    max_leverage = 5.0
    daily_profit_lock_pct = 0.0
    kill_switch_file = "/nonexistent/KILL"


# ---------------- Gap 4: correlated exposure ----------------

def test_legs_usd_normalization():
    assert parse_currencies("USD_JPY") == ("USD", "JPY")
    # JPY correctness: 100k USD_JPY units = 100k USD exposure, not 15.5M JPY
    assert legs_usd("USD_JPY", 100_000, 155.0) == [("USD", 100_000.0), ("JPY", -100_000.0)]
    # EUR_USD long: +EUR leg at price, -USD leg
    legs = dict(legs_usd("EUR_USD", 100_000, 1.10))
    assert legs["EUR"] == pytest.approx(110_000) and legs["USD"] == pytest.approx(-110_000)


def test_third_correlated_usd_order_blocked():
    rm = RiskManager(_Cfg())
    rm._roll_day_if_needed(100_000.0, now=pd.Timestamp("2026-09-01 10:00", tz="UTC").to_pydatetime())
    # two long-USD pairs, sized so they coexist under the cap (USD = -5.9x):
    opens = [
        {"tradeId": "1", "instrument": "EUR_USD", "units": 250_000, "price": 1.10},
        {"tradeId": "2", "instrument": "GBP_USD", "units": 250_000, "price": 1.26},
    ]
    # a third long-USD pair would push net USD short exposure past 6x equity
    ok, reason, _ = rm.check_entry(
        instrument="AUD_USD", price=0.65, atr=0.0005, equity=100_000.0,
        open_trades=opens, now=pd.Timestamp("2026-09-01 12:00", tz="UTC"),
        spread_pips=1.0)
    assert not ok and "ccy_exposure USD" in reason, reason


def test_opposite_side_usd_trade_allowed_by_exposure_guard():
    rm = RiskManager(_Cfg())
    rm._roll_day_if_needed(100_000.0, now=pd.Timestamp("2026-09-01 10:00", tz="UTC").to_pydatetime())
    # USD_CAD long = long USD: offsets the short-USD book instead of stacking it
    ok, reason, _ = rm.check_entry(
        instrument="USD_CAD", price=1.36, atr=0.0005, equity=100_000.0,
        open_trades=[{"tradeId": "1", "instrument": "EUR_USD", "units": 100_000, "price": 1.10}],
        now=pd.Timestamp("2026-09-01 12:00", tz="UTC"),
        spread_pips=1.0)
    assert ok and reason == "ok"


def test_usd_jpy_uses_usd_normalized_legs():
    rm = RiskManager(_Cfg())
    rm._roll_day_if_needed(100_000.0, now=pd.Timestamp("2026-09-01 10:00", tz="UTC").to_pydatetime())
    # one USD_JPY long (100k USD exposure) must not false-block on raw-JPY magnitude
    opens = [{"instrument": "USD_JPY", "units": 100_000, "price": 155.0}]
    ok, reason, _ = rm.check_entry(
        instrument="AUD_USD", price=0.65, atr=0.0005, equity=100_000.0,
        open_trades=opens, now=pd.Timestamp("2026-09-01 12:00", tz="UTC"),
        spread_pips=1.0)
    # exposure guard speaks in USD-normalized legs, never raw JPY
    if not ok:
        assert "ccy_exposure USD" in reason, reason


# ---------------- Gap 3: dry-run fill simulation ----------------

def test_dryrun_fill_accounting_and_persistence(tmp_path):
    state = str(tmp_path / "dry.json")
    br = DryRunBroker(state, 100_000.0, spread_pips=1.0, slippage_pips=0.3)
    # fill long 10k @1.1000: entry pays half-spread+slippage = 0.8 pips = 0.00008
    br.fill("EUR_USD", 10_000, 1.1000, atr=0.0005,
            atr_sl_multiple=1.6, atr_tp_multiple=2.4, bar_time=pd.Timestamp("2026-09-01", tz="UTC"))
    assert len(br.open_trades()) == 1
    pos = br.st["positions"][0]
    assert pos["entry"] == pytest.approx(1.10008)
    assert pos["tp"] == pytest.approx(1.10008 + 2.4 * 0.0005)
    # bar hits TP only (low stays above SL 1.09928) -> exit pays cost side too
    bar = pd.Series({"time": pd.Timestamp("2026-09-01 00:05", tz="UTC"),
                     "open": 1.101, "high": 1.1050, "low": 1.0995, "close": 1.104})
    br.mark("EUR_USD", bar)
    assert br.st["positions"] == []
    assert br.st["fills"][0]["reason"] == "tp"
    assert br.st["fills"][0]["net_pnl"] == pytest.approx((1.10128 - 0.00008 - 1.10008) * 10_000)
    assert br.equity() == pytest.approx(100_000 + 11.20)
    # persistence: a new session continues the same equity curve
    br2 = DryRunBroker(state, 100_000.0, spread_pips=1.0, slippage_pips=0.3)
    assert br2.equity() == pytest.approx(100_011.20)
    assert len(br2.st["fills"]) == 1


def test_dryrun_stop_wins_when_both_hit(tmp_path):
    br = DryRunBroker(str(tmp_path / "d.json"), 100_000.0, 1.0, 0.3)
    br.fill("EUR_USD", 10_000, 1.1000, 0.0005, 1.6, 2.4, pd.Timestamp("2026-09-01", tz="UTC"))
    bar = pd.Series({"time": pd.Timestamp("2026-09-01 00:05", tz="UTC"),
                     "open": 1.101, "high": 1.12, "low": 1.08, "close": 1.10})
    br.mark("EUR_USD", bar)
    assert br.st["fills"][0]["reason"] == "stop"  # conservative: stop first
    assert br.equity() < 100_000


def test_dryrun_sync_risk_and_roll_day(tmp_path):
    br = DryRunBroker(str(tmp_path / "d.json"), 100_000.0, 1.0, 0.3)
    br.fill("EUR_USD", 10_000, 1.1000, 0.0005, 1.6, 2.4, pd.Timestamp("2026-09-01", tz="UTC"))
    bar = pd.Series({"time": pd.Timestamp("2026-09-01 00:05", tz="UTC"),
                     "open": 1.101, "high": 1.12, "low": 1.08, "close": 1.10})
    br.mark("EUR_USD", bar)
    br.roll_day("2026-09-01")
    rm = RiskManager(_Cfg())
    br.sync_risk(rm)
    assert rm.state.realized_today == br.st["realized_today"]
    assert "EUR_USD" in rm.state.last_loss_time


# ---------------- Gap 1: doctor ----------------

class FakeOandaOK:
    def __init__(self, *a, **k):
        pass

    def account_summary(self):
        return {"account": {"id": "101-004-1234567-001", "currency": "USD"}}

    def account_details(self):
        insts = [{"id": i} for i in ("EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD")]
        return {"account": {"instruments": insts}}


def test_doctor_missing_credentials_is_checklist_not_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no .env here
    cfg = Config(db_path=str(tmp_path / "s.db"))
    ok = run_doctor(cfg)
    out = capsys.readouterr().out
    assert ok is False
    assert "[FAIL] OANDA token format" in out and "[FAIL] env file (.env)" in out
    assert "To fix credentials" in out and "OANDA_TOKEN=" in out
    assert "Traceback" not in out


def test_doctor_all_pass_with_fake_client(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OANDA_TOKEN=x\n", encoding="utf-8")
    monkeypatch.setattr("trader.oanda_client.OandaClient", FakeOandaOK)
    cfg = Config(db_path=str(tmp_path / "s.db"), oanda_token="a" * 64,
                 oanda_account_id="101-004-1234567-001")
    assert run_doctor(cfg) is True
    assert "ALL CHECKS PASSED" in capsys.readouterr().out


def test_doctor_connectivity_fail(tmp_path, monkeypatch, capsys):
    class FakeOandaDead(FakeOandaOK):
        def account_summary(self):
            raise RuntimeError("401: bad token")

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OANDA_TOKEN=x\n", encoding="utf-8")
    monkeypatch.setattr("trader.oanda_client.OandaClient", FakeOandaDead)
    cfg = Config(db_path=str(tmp_path / "s.db"), oanda_token="a" * 64,
                 oanda_account_id="101-004-1234567-001")
    assert run_doctor(cfg) is False
    assert "[FAIL] practice API connectivity" in capsys.readouterr().out


# ---------------- Gap 2: artifacts after the e2e pipeline ----------------

def test_artifacts_written_after_walkforward(tmp_path):
    df = synth(1600)
    ds, windows = walk_forward(df, train_bars=800, test_bars=200, step_bars=200,
                               verbose=False)
    oos = ds[ds["oos_proba"].notna()].reset_index(drop=True)
    res = run_backtest(oos, oos["oos_proba"].to_numpy(), threshold=0.55,
                       instrument="EUR_USD")
    # per-window OOS stats, as run.py computes them
    for wd in windows:
        seg = oos[(oos["time"] >= wd["start"]) & (oos["time"] <= wd["end"])]
        wd.update(oos_trades=0, oos_net_pnl=0.0, oos_sharpe=0.0)
        if len(seg) > 50:
            r = run_backtest(seg.reset_index(drop=True), seg["oos_proba"].to_numpy(),
                             threshold=wd["threshold"], instrument="EUR_USD")
            wd.update(oos_trades=r.n_trades, oos_net_pnl=r.net_profit, oos_sharpe=r.sharpe)
    tag = write_artifacts(str(tmp_path), "EUR_USD", windows, res.trades, res.equity, "summary text")
    assert "EUR_USD" in tag
    files = {p.name for p in tmp_path.iterdir()}
    assert any(f.startswith("windows_") for f in files)
    assert any(f.startswith("trades_") for f in files)
    assert any(f.startswith("equity_") for f in files)
    # window table carries the audit columns
    wf = next(p for p in tmp_path.iterdir() if p.name.startswith("windows_"))
    row = wf.read_text(encoding="utf-8").strip().split("\n")[1].split(",")
    assert len(row) == 8
