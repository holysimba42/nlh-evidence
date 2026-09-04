"""End-to-end: synthetic FX data -> walk-forward OOS predictions -> costed
backtest. Proves the whole machine runs and produces sane (non-exploding)
out-of-sample behavior WITHOUT touching OANDA."""
import numpy as np
import pandas as pd

from trader.backtest import run_backtest
from trader.walkforward import walk_forward


def test_full_pipeline_e2e():
    rng = np.random.default_rng(11)
    n = 4200
    raw = pd.date_range("2026-01-05", periods=int(n * 1.5), freq="5min")
    times = raw[np.asarray(raw.dayofweek < 5)][:n]
    # trend + mean-reversion mix + noise: learnable structure
    t = np.arange(n)
    drift = 0.000004 * np.sin(t / 300) + 0.000002 * np.sign(np.sin(t / 900))
    steps = rng.normal(0, 0.00008, n) + drift
    close = 1.10 + np.cumsum(steps)
    spread_noise = np.abs(rng.normal(0, 0.0003, n))
    df = pd.DataFrame({
        "time": times,
        "open": np.r_[close[0], close[:-1]],
        "high": close + spread_noise,
        "low": close - np.abs(rng.normal(0, 0.0003, n)),
        "close": close,
        "volume": rng.integers(50, 400, n),
    })

    ds, windows = walk_forward(df, train_bars=2000, test_bars=500, step_bars=500,
                               verbose=False)
    oos = ds[ds["oos_proba"].notna()].reset_index(drop=True)
    assert len(windows) >= 3
    assert len(oos) >= 1500

    res = run_backtest(
        oos, oos["oos_proba"].to_numpy(), threshold=0.55, instrument="EUR_USD",
        spread_pips=1.0, slippage_pips=0.3,
    )
    # sanity bounds: pipeline must not explode equity or produce absurd stats
    assert res.max_drawdown_pct < 50.0
    assert abs(res.sharpe) < 100.0
    assert (res.trades["units"] > 0).all()
    # every trade paid round-trip costs
    assert (res.trades["net_pips"] <= res.trades["gross_pips"]).all()
    print(res.summary())
