"""Walk-forward (rolling out-of-sample) validation — the only honest Sharpe.

For each window:
  train on bars   [t0,              t0+train)
  predict bars   [t0+train,        t0+train+test)   <- model has NEVER seen these
  step forward by `step`, retrain, repeat.
Concatenated OOS predictions = what live would have actually looked like.
Parameters (threshold) are tuned INSIDE the train slice only, per window.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .artifacts import write_artifacts
from .backtest import run_backtest
from .signal import EnsembleSignal, build_dataset

log = logging.getLogger(__name__)

THRESHOLDS = (0.50, 0.55, 0.60, 0.65)


def run_instrument_validation(client, store, instrument: str, cfg) -> dict | None:
    """Full validation orchestration for one instrument: history (store-first,
    then client), walk-forward, per-window OOS scoring, overall OOS backtest,
    directional accuracy, artifacts. Returns a result bundle, or None when the
    API returns no candles. Raises ValueError on insufficient history.
    Pure strategy-layer logic — the CLI only parses args and prints."""
    gran = cfg.signal_granularity
    df = store.load(instrument, gran)
    if len(df) >= cfg.history_bars:
        df = df.tail(cfg.history_bars).reset_index(drop=True)
    else:
        fresh = client.get_candles(instrument, gran, count=cfg.history_bars)
        if fresh.empty:
            return None
        store.upsert(fresh, instrument, gran)
        df = fresh

    ds, windows = walk_forward(
        df, instrument=instrument, train_bars=cfg.wf_train_bars,
        test_bars=cfg.wf_test_bars, step_bars=cfg.wf_step_bars,
        spread_pips=cfg.default_spread_pips, slippage_pips=cfg.slippage_pips)
    oos = ds[ds["oos_proba"].notna()].reset_index(drop=True)

    # per-window OOS performance (unseen data, window's own tuned threshold)
    for wd in windows:
        seg = oos[(oos["time"] >= wd["start"]) & (oos["time"] <= wd["end"])].reset_index(drop=True)
        wd.update(oos_trades=0, oos_net_pnl=0.0, oos_sharpe=0.0)
        if len(seg) > 50:
            r = run_backtest(
                seg, seg["oos_proba"].to_numpy(), threshold=wd["threshold"],
                instrument=instrument, spread_pips=cfg.default_spread_pips,
                slippage_pips=cfg.slippage_pips, skip_hours=cfg.skip_hours,
                skip_friday_after=cfg.skip_friday_after)
            wd.update(oos_trades=r.n_trades, oos_net_pnl=r.net_profit,
                      oos_sharpe=r.sharpe)

    res = run_backtest(
        oos, oos["oos_proba"].to_numpy(), threshold=0.55, instrument=instrument,
        spread_pips=cfg.default_spread_pips, slippage_pips=cfg.slippage_pips,
        atr_mult_sl=cfg.atr_sl_multiple, atr_mult_tp=cfg.atr_tp_multiple,
        risk_per_trade_pct=cfg.risk_per_trade_pct, skip_hours=cfg.skip_hours,
        skip_friday_after=cfg.skip_friday_after)

    # directional accuracy on tradable OOS bars (vs the 50% coin)
    y = (oos["label"] > 0).astype(int)
    p = oos["oos_proba"].to_numpy()
    tradable = (p >= 0.5 + 0.55 / 2) | (p <= 0.5 - 0.55 / 2)
    acc_txt = ""
    if tradable.sum() > 0:
        acc = float(((p[tradable] > 0.5).astype(int) == y[tradable]).mean())
        acc_txt = (f"OOS directional accuracy on tradable bars: {acc:.1%} "
                   f"({int(tradable.sum())} bars, base rate {y.mean():.1%})")

    write_artifacts(cfg.artifacts_dir, instrument, windows, res.trades, res.equity,
                    f"OOS (unseen data): {res.summary()}\n{acc_txt}")
    return {
        "instrument": instrument, "bars": len(df),
        "span": f"{df['time'].iloc[0]} .. {df['time'].iloc[-1]}",
        "oos_bars": len(oos), "windows": windows, "result": res,
        "accuracy_text": acc_txt,
    }


def walk_forward(
    df: pd.DataFrame,
    *,
    instrument: str = "EUR_USD",
    horizon: int = 12,
    train_bars: int = 2000,
    test_bars: int = 500,
    step_bars: int = 500,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.3,
    seed: int = 7,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (dataset-with-OOS-proba, per-window metadata).
    Only rows inside test slices carry meaningful `oos_proba` (NaN elsewhere)."""
    ds = build_dataset(df, horizon=horizon)
    n = len(ds)
    proba = np.full(n, np.nan)
    windows: list[dict] = []

    starts = range(train_bars, n - test_bars + 1, step_bars)
    if len(list(starts)) == 0:
        raise ValueError(
            f"not enough data: have {n} labelled rows, need > {train_bars + test_bars}")

    for s in starts:
        tr = ds.iloc[s - train_bars:s]
        te = ds.iloc[s:s + test_bars]
        # --- tune threshold on TRAIN slice only (mini inner split to avoid
        #     tuning on the same rows the model fit) ---
        inner_cut = int(len(tr) * 0.8)
        try:
            sig = EnsembleSignal(threshold=0.5, horizon=horizon, seed=seed).fit(tr.iloc[:inner_cut])
        except ValueError as e:
            log.warning("window @%d skipped (fit): %s", s, e)
            continue
        inner_tr = tr.iloc[inner_cut:]
        p_inner = sig.predict_proba_frame(inner_tr)
        # Tune threshold by NET PnL AFTER COSTS on the inner train slice —
        # the objective is the target itself (net profit / Sharpe), not a proxy.
        best_th, best_score = THRESHOLDS[1], -np.inf
        for th in THRESHOLDS:
            try:
                res = run_backtest(
                    inner_tr.reset_index(drop=True), p_inner, threshold=th,
                    instrument=instrument,
                    spread_pips=spread_pips, slippage_pips=slippage_pips,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("threshold probe failed (th=%.2f): %s", th, e)
                continue
            # require a minimum number of trades to avoid degenerate picks
            if res.n_trades < 5:
                score = -np.inf
            else:
                score = res.net_profit if res.net_profit > 0 else res.net_profit * 2.0
            if score > best_score:
                best_th, best_score = th, score
        # --- final fit on the FULL train slice with the chosen threshold ---
        sig = EnsembleSignal(threshold=best_th, horizon=horizon, seed=seed).fit(tr)
        proba[s:s + test_bars] = sig.predict_proba_frame(te)
        windows.append({
            "start": ds["time"].iloc[s], "end": ds["time"].iloc[s + test_bars - 1],
            "train_end": ds["time"].iloc[s - 1], "threshold": best_th,
            "tune_score": float(best_score),
            "train_acc": getattr(sig, "train_acc_", float("nan")),
            "train_n": getattr(sig, "train_n_", int(len(tr))),
        })
        if verbose:
            log.info("wf window %s -> %s th=%.2f train_acc=%.3f",
                     windows[-1]["start"].date(), windows[-1]["end"].date(),
                     best_th, windows[-1]["train_acc"])

    ds["oos_proba"] = proba
    return ds, windows
