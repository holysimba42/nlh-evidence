"""Artifact writer: every walk-forward/backtest run leaves auditable files in
./artifacts/ — a per-window table, an all-trades CSV, and an equity curve
(PNG when matplotlib is available, else the curve as CSV)."""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def write_artifacts(artifacts_dir: str, instrument: str, windows: list[dict] | None,
                    trades: pd.DataFrame, equity: pd.Series,
                    summary: str) -> str:
    """windows=None skips the per-window table (e.g. dry-run reports have no
    walk-forward windows). Trade rows may carry their own 'instrument' column
    (dry-run fills span instruments); the argument is the fallback/tag name."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(artifacts_dir)
    base.mkdir(parents=True, exist_ok=True)
    tag = f"{instrument}_{stamp}"
    paths: list[str] = []

    # --- per-window table (optional) ---
    if windows is not None:
        p = base / f"windows_{tag}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start", "end", "train_end", "threshold", "train_acc",
                        "oos_trades", "oos_net_pnl", "oos_sharpe"])
            for wd in windows:
                w.writerow([
                    wd.get("start"), wd.get("end"), wd.get("train_end"),
                    wd.get("threshold"), wd.get("train_acc"),
                    wd.get("oos_trades"), wd.get("oos_net_pnl"), wd.get("oos_sharpe")])
        paths.append(str(p))

    # --- all-trades CSV ---
    p = base / f"trades_{tag}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instrument", "direction", "units", "entry_time", "entry_price",
                    "exit_time", "exit_price", "costs_pips", "net_pnl", "exit_reason"])
        for _, t in trades.iterrows():
            w.writerow([t.get("instrument", instrument), t["direction"], t["units"],
                        t["entry_time"], t["entry"], t["exit_time"], t["exit"],
                        t["net_pips"], t["net_pnl"], t["reason"]])
    paths.append(str(p))

    # --- equity curve ---
    p = base / f"equity_{tag}.csv"
    equity.rename("equity").to_csv(p, index_label="time", header=True)
    paths.append(str(p))
    png = base / f"equity_{tag}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(equity.index, equity.to_numpy())
        ax.set_title(f"{instrument} — OOS equity (walk-forward)")
        ax.set_ylabel("equity")
        fig.autofmt_xdate()
        fig.savefig(png, dpi=110, bbox_inches="tight")
        paths.append(str(png))
    except ImportError:
        pass  # matplotlib absent -> CSV is the artifact

    print(f"\nartifacts -> {', '.join(paths)}\n{summary}")
    return tag
