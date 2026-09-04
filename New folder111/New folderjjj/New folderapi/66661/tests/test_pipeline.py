"""Unit tests for the trading pipeline. These encode the safety invariants:
no lookahead, correct label alignment, hard risk gates, honest cost math."""
import numpy as np
import pandas as pd

from trader.features import FEATURES, compute_features, future_return
from trader.instruments import cost_per_side, pip_size, price_to_pips
from trader.risk import RiskManager


def _synth_candles(n=800, seed=3) -> pd.DataFrame:
    """Synthetic M5 candles with mild trend + noise (no OANDA needed)."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-08-03", tz="UTC")  # a Monday
    raw = pd.date_range(t0, periods=n * 2, freq="5min")
    times = raw[np.asarray(raw.dayofweek < 5)][:n]  # drop weekend bars, FX-like
    n2 = len(times)
    steps = rng.normal(0, 0.00008, n2) + 0.000003 * np.sin(np.arange(n2) / 200)
    close = 1.10 + np.cumsum(steps)
    high = close + np.abs(rng.normal(0, 0.00025, n2))
    low = close - np.abs(rng.normal(0, 0.00025, n2))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"time": times, "open": open_, "high": high,
                         "low": low, "close": close,
                         "volume": rng.integers(50, 400, n2)})


def test_features_are_causal():
    """Changing the FUTURE must not change past feature values (leakage proof)."""
    df = _synth_candles(400)
    base = compute_features(df).iloc[150:200]
    df2 = df.copy()
    df2.loc[df2.index[-50:], "close"] *= 1.05   # mutate only the future
    df2.loc[df2.index[-50:], "high"] *= 1.05
    fut = compute_features(df2).iloc[150:200]
    pd.testing.assert_frame_equal(base, fut)


def test_label_alignment():
    df = _synth_candles(200)
    fr = future_return(df, horizon=12)
    assert abs(fr.iloc[100] - (df['close'].iloc[112] / df['close'].iloc[100] - 1) * 1e4) < 1e-6
    assert fr.iloc[-12:].isna().all()      # last horizon bars have no label
    assert fr.iloc[:-12].notna().all()


def test_all_features_present_and_finite():
    df = compute_features(_synth_candles(600)).dropna()
    missing = [c for c in FEATURES if c not in df.columns]
    assert not missing, f"missing features: {missing}"
    assert np.isfinite(df[FEATURES].to_numpy()).all()


def test_pip_math():
    assert pip_size("EUR_USD") == 0.0001
    assert pip_size("USD_JPY") == 0.01
    assert price_to_pips(0.0010, "EUR_USD") == 10.0


def test_cost_per_side_math():
    """One home for the per-side cost; both money paths consume it.
    Majors: (1.0/2 + 0.3) * 0.0001 = 8e-05. JPY: same pips * 0.01 = 0.008."""
    assert cost_per_side(1.0, 0.3, "EUR_USD") == 8e-05
    assert cost_per_side(1.0, 0.3, "USD_JPY") == 0.008
    assert cost_per_side(0.0, 0.0, "EUR_USD") == 0.0  # free fills cost nothing


class _Cfg:
    risk_per_trade_pct = 0.4
    max_daily_loss_pct = 1.5
    max_open_trades = 3
    max_trades_per_instrument = 1
    cooldown_after_loss_minutes = 30
    max_spread_pips = 2.0
    max_ccy_exposure_x = 2.0
    atr_sl_multiple = 1.6
    daily_profit_lock_pct = 0.0
    kill_switch_file = "/nonexistent/KILL"


def test_risk_daily_loss_halt():
    rm = RiskManager(_Cfg())
    rm._roll_day_if_needed(100_000.0, now=pd.Timestamp("2026-09-01 10:00", tz="UTC").to_pydatetime())
    rm.on_trade_closed(-1_600.0, "EUR_USD", pd.Timestamp("2026-09-01 11:00", tz="UTC"))
    ok, reason, _ = rm.check_entry(
        instrument="EUR_USD", price=1.1, atr=0.0005, equity=98_400.0,
        open_trades=[], now=pd.Timestamp("2026-09-01 11:05", tz="UTC"),
        spread_pips=1.0)
    assert not ok and "daily_loss" in reason


def test_risk_cooldown_and_spread():
    rm = RiskManager(_Cfg())
    rm._roll_day_if_needed(100_000.0, now=pd.Timestamp("2026-09-01 10:00", tz="UTC").to_pydatetime())
    rm.on_trade_closed(-50.0, "EUR_USD", pd.Timestamp("2026-09-01 11:00", tz="UTC"))
    ok, reason, _ = rm.check_entry(
        instrument="EUR_USD", price=1.1, atr=0.0005, equity=99_950.0,
        open_trades=[], now=pd.Timestamp("2026-09-01 11:10", tz="UTC"),
        spread_pips=1.0)
    assert not ok and "cooldown" in reason
    rm2 = RiskManager(_Cfg())
    rm2._roll_day_if_needed(100_000.0, now=pd.Timestamp("2026-09-01 10:00", tz="UTC").to_pydatetime())
    ok, reason, _ = rm2.check_entry(
        instrument="EUR_USD", price=1.1, atr=0.0005, equity=100_000.0,
        open_trades=[], now=pd.Timestamp("2026-09-01 10:30", tz="UTC"),
        spread_pips=5.0)
    assert not ok and "spread" in reason
