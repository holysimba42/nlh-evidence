"""Event-driven backtester with honest execution:

  - features on bar t use data <= t; the signal fires on bar t's CLOSE
  - execution happens at bar t+1's OPEN (one-bar delay, no lookahead)
  - costs: spread + slippage applied on entry AND exit, converted from pips
  - ATR stop / TP evaluated against bar extremes; if both are hit inside one
    bar we conservatively assume the STOP hit first (adverse assumption)
  - every trade pays its costs; net profit = gross - costs

If an edge survives THIS gauntlet out-of-sample, it has a chance in live.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .instruments import cost_per_side, pip_size, price_to_pips


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    net_profit: float
    n_trades: int
    avg_trade_pips: float
    cagr: float

    def summary(self) -> str:
        return (
            f"trades={self.n_trades}  win={self.win_rate:.1%}  PF={self.profit_factor:.2f}  "
            f"Sharpe={self.sharpe:.2f}  Sortino={self.sortino:.2f}  "
            f"maxDD={self.max_drawdown_pct:.1%}  net=${self.net_profit:,.0f}  "
            f"avg={self.avg_trade_pips:+.1f} pips/trade"
        )


@dataclass
class _Open:
    direction: int          # +1 long / -1 short
    entry_price: float
    units: int
    sl: float
    tp: float
    entry_time: pd.Timestamp


def run_backtest(
    ds: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    instrument: str,
    *,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.3,
    atr_mult_sl: float = 1.6,
    atr_mult_tp: float = 2.4,
    risk_per_trade_pct: float = 0.004,
    start_equity: float = 100_000.0,
    max_leverage: float = 5.0,
    skip_hours: set[int] | None = None,
    skip_friday_after: int = 19,
    contract_size: float = 100_000.0,   # units per 1.0 lot (FX standard)
    veto_range_edge: float = 0.18,
) -> BacktestResult:
    """proba[i] = P(long wins) for ds row i (already causal — produced by
    time-ordered fit/predict). Pure numpy loop over bars: fast & exact."""
    skip_hours = skip_hours or set()
    ps = pip_size(instrument)
    cost = cost_per_side(spread_pips, slippage_pips, instrument)  # price units per side
    round_cost = 2.0 * cost

    open_, high, low, close = (ds[c].to_numpy() for c in ("open", "high", "low", "close"))
    atr = ds["atr"].to_numpy()
    regime = ds["regime"].to_numpy() if "regime" in ds else np.array(["range"] * len(ds))
    hours = ds["time"].dt.hour.to_numpy()
    dows = ds["time"].dt.dayofweek.to_numpy()
    times = ds["time"].to_numpy()

    n = len(ds)
    equity = start_equity
    equity_curve: list[float] = [equity]
    trades: list[dict] = []
    pos: _Open | None = None

    def desired_units(direction: int, entry: float, stop: float, eq: float) -> int:
        risk_amount = eq * risk_per_trade_pct
        stop_dist = max(abs(entry - stop), 2 * ps)  # never size to absurdity
        units = risk_amount / stop_dist
        # leverage cap: notional <= eq * max_leverage
        units = min(units, eq * max_leverage / entry)
        return int(max(units, 0))

    for i in range(n - 1):
        # ---------- manage open position on bar i (t+1 = bar i) ----------
        if pos is not None:
            exit_price: float | None = None
            exit_reason = ""
            if pos.direction == 1:
                if low[i] <= pos.sl:                      # stop first (conservative)
                    exit_price, exit_reason = pos.sl, "stop"
                elif high[i] >= pos.tp:
                    exit_price, exit_reason = pos.tp, "tp"
            else:
                if high[i] >= pos.sl:
                    exit_price, exit_reason = pos.sl, "stop"
                elif low[i] <= pos.tp:
                    exit_price, exit_reason = pos.tp, "tp"
            # signal flip -> exit at open of the flip bar
            p = proba[i - 1] if i > 0 else 0.5
            want = 1 if p >= 0.5 + threshold / 2 else (-1 if p <= 0.5 - threshold / 2 else 0)
            if exit_price is None and want != 0 and want != pos.direction:
                exit_price, exit_reason = open_[i], "flip"
            if exit_price is None and dows[i] == 4 and hours[i] >= skip_friday_after:
                exit_price, exit_reason = close[i], "friday_flat"
            if exit_price is not None:
                gross = (exit_price - pos.entry_price) * pos.direction * pos.units
                net = gross - round_cost * pos.units
                equity += net
                trades.append({
                    "entry_time": pos.entry_time, "exit_time": times[i],
                    "direction": "long" if pos.direction == 1 else "short",
                    "entry": pos.entry_price, "exit": exit_price,
                    "units": pos.units,
                    "gross_pips": price_to_pips((exit_price - pos.entry_price) * pos.direction, instrument),
                    "net_pips": price_to_pips((exit_price - pos.entry_price) * pos.direction - round_cost, instrument),
                    "net_pnl": net, "reason": exit_reason,
                    "equity_after": equity,
                })
                pos = None

        # ---------- decide on bar i's close, execute at bar i+1 open ----------
        if pos is None:
            p = proba[i]
            want = 1 if p >= 0.5 + threshold / 2 else (-1 if p <= 0.5 - threshold / 2 else 0)
            if want != 0:
                r = regime[i]
                if r == "range" and abs(p - 0.5) < veto_range_edge:
                    want = 0
            if want != 0 and hours[i] not in skip_hours and hours[i + 1] not in skip_hours:
                entry = open_[i + 1]
                a = atr[i]
                sl = entry - want * atr_mult_sl * a
                tp = entry + want * atr_mult_tp * a
                u = desired_units(want, entry, sl, equity)
                if u > 0:
                    pos = _Open(want, entry, u, sl, tp, times[i + 1])

        equity_curve.append(equity)

    # force-close anything left at the final close
    if pos is not None:
        exit_price = close[-1]
        gross = (exit_price - pos.entry_price) * pos.direction * pos.units
        net = gross - round_cost * pos.units
        equity += net
        trades.append({
            "entry_time": pos.entry_time, "exit_time": times[-1],
            "direction": "long" if pos.direction == 1 else "short",
            "entry": pos.entry_price, "exit": exit_price,
            "units": pos.units,
            "gross_pips": price_to_pips((exit_price - pos.entry_price) * pos.direction, instrument),
            "net_pips": price_to_pips((exit_price - pos.entry_price) * pos.direction - round_cost, instrument),
            "net_pnl": net, "reason": "eod",
            "equity_after": equity,
        })

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity_curve, index=pd.DatetimeIndex(times), name="equity")

    return _metrics(tdf, eq, start_equity)


def _metrics(trades: pd.DataFrame, equity: pd.Series, start_equity: float) -> BacktestResult:
    if trades.empty:
        return BacktestResult(trades, equity, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    rets = equity.pct_change().dropna()
    # per-M5-bar returns -> annualize: 288 bars/day * 252 trading days (24/5 FX)
    ann = np.sqrt(288 * 252)
    sd = rets.std()
    sharpe = float(rets.mean() / sd * ann) if sd > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = float(rets.mean() / downside * ann) if downside and downside > 0 else 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1.0).min()
    wins = trades[trades["net_pnl"] > 0]["net_pnl"].sum()
    losses = -trades[trades["net_pnl"] <= 0]["net_pnl"].sum()
    pf = float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0
    net = float(trades["net_pnl"].sum())
    years = max((equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    cagr = (equity.iloc[-1] / start_equity) ** (1 / years) - 1.0
    return BacktestResult(
        trades=trades, equity=equity, sharpe=sharpe, sortino=sortino,
        max_drawdown_pct=float(-dd), win_rate=float((trades["net_pnl"] > 0).mean()),
        profit_factor=pf, net_profit=net, n_trades=int(len(trades)),
        avg_trade_pips=float(trades["net_pips"].mean()), cagr=float(cagr),
    )
