"""Risk manager: the module that keeps you in the game.

Constraints (all hard, checked BEFORE every order):
  - per-trade risk  = equity * risk_per_trade_pct / stop-distance  (position sizing)
  - daily loss halt: if day realized PnL <= -max_daily_loss_pct of day-start equity,
    flat + no new trades until next UTC day
  - profit lock: optional stop-above target (locks a good day)
  - max concurrent trades, max trades per instrument
  - leverage cap on notional
  - cooldown after a loss on that instrument
  - spread filter at entry time
  - kill switch: create file `data/KILL` -> flatten & stand down instantly
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .instruments import legs_usd

log = logging.getLogger(__name__)


@dataclass
class RiskState:
    day_start_equity: float = 0.0
    day_key: str = ""
    realized_today: float = 0.0
    trade_count_today: int = 0
    last_loss_time: dict[str, pd.Timestamp] = field(default_factory=dict)
    # tradeId -> {instrument, entry, units, direction}; maintained by live reconciliation
    known_trades: dict[str, dict] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = RiskState()

    # ---------- state upkeep ----------
    def _roll_day_if_needed(self, equity: float, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        key = now.strftime("%Y-%m-%d")
        if key != self.state.day_key:
            self.state.day_key = key
            self.state.day_start_equity = equity
            self.state.realized_today = 0.0
            self.state.trade_count_today = 0
            self.state.halted = False
            self.state.halt_reason = ""
            log.info("new trading day %s, equity anchor %.2f", key, equity)

    def on_trade_closed(self, pnl: float, instrument: str, closed_at: pd.Timestamp) -> None:
        self.state.realized_today += pnl
        self.state.trade_count_today += 1
        if pnl < 0:
            self.state.last_loss_time[instrument] = closed_at

    def kill_switch_active(self) -> bool:
        return Path(self.cfg.kill_switch_file).exists()

    # ---------- the gate every order must pass ----------
    def check_entry(self, *, instrument: str, price: float, atr: float,
                    equity: float, open_trades: list[dict],
                    now: pd.Timestamp, spread_pips: float) -> tuple[bool, str, float]:
        """Returns (allowed, reason, units). units=0 when not allowed."""
        c = self.cfg
        self._roll_day_if_needed(equity, now.to_pydatetime() if hasattr(now, "to_pydatetime") else now)

        if self.kill_switch_active():
            return False, "kill_switch", 0.0
        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}", 0.0

        # daily loss halt (on realized day PnL)
        day_floor = -c.max_daily_loss_pct / 100.0 * self.state.day_start_equity
        if self.state.realized_today <= day_floor:
            self.state.halted = True
            self.state.halt_reason = "daily loss limit"
            return False, "daily_loss_halt", 0.0

        # optional profit lock
        if c.daily_profit_lock_pct > 0:
            lock = c.daily_profit_lock_pct / 100.0 * self.state.day_start_equity
            if self.state.realized_today >= lock:
                return False, "daily_profit_lock", 0.0

        # kill switch file checked above; instrument-level guards next
        mine = [t for t in open_trades
                if str(t.get("instrument", "")).strip().upper() == instrument]
        if len(mine) >= c.max_trades_per_instrument:
            return False, "per_instrument_cap", 0.0
        if len(open_trades) >= c.max_open_trades:
            return False, "max_open_trades", 0.0

        # cooldown after a loss
        last_loss = self.state.last_loss_time.get(instrument)
        if last_loss is not None:
            mins = (now - last_loss).total_seconds() / 60.0
            if mins < c.cooldown_after_loss_minutes:
                return False, f"cooldown {mins:.0f}m", 0.0

        # spread filter
        if spread_pips > c.max_spread_pips:
            return False, f"spread {spread_pips:.1f}p > {c.max_spread_pips}p", 0.0

        # --- sizing: risk-based, then leverage-capped ---
        stop_dist = max(c.atr_sl_multiple * atr, 2e-5)
        risk_amount = equity * c.risk_per_trade_pct / 100.0
        units = risk_amount / stop_dist
        max_units = equity * c.max_leverage / price
        units = min(units, max_units)
        units = int(units)
        if units <= 0:
            return False, "size_too_small", 0.0

        # correlated-exposure guard (needs final units): net exposure per currency
        # across ALL opens. 3 long-USD pairs is a 3x dollar bet, not '3 positions'.
        exposure = self._exposure_map(open_trades)
        limit = c.max_ccy_exposure_x * equity
        for ccy, leg in legs_usd(instrument, units, price):
            projected = exposure.get(ccy, 0.0) + leg
            if abs(projected) > limit:
                return False, (
                    f"ccy_exposure {ccy} {projected / equity:.2f}x > "
                    f"{c.max_ccy_exposure_x:.1f}x"), 0.0
        return True, "ok", units

    def _exposure_map(self, open_trades: list[dict]) -> dict[str, float]:
        """Net signed exposure per currency (USD-equivalent via legs_usd).
        Accepts OANDA trade dicts ('price') and dry-run positions ('entry')."""
        expo: dict[str, float] = {}
        for t in open_trades:
            units = float(t.get("units", 0.0) or 0.0)
            price = float(t.get("price", 0.0) or 0.0) or float(t.get("entry", 0.0) or 0.0)
            if units == 0 or price <= 0:
                continue
            for ccy, leg in legs_usd(t.get("instrument", ""), units, price):
                expo[ccy] = expo.get(ccy, 0.0) + leg
        return expo

    def register_fill(self, instrument: str, trade_id: str, entry_price: float,
                      units: int, direction: int) -> None:
        """Record a filled order so reconcile_closures() can price its PnL."""
        self.state.known_trades[str(trade_id)] = {
            "instrument": instrument, "entry": float(entry_price),
            "units": int(units), "direction": int(direction)}

    def flatten_all(self, client) -> None:
        """Emergency: close everything immediately."""
        try:
            client.close_all()
            log.warning("KILL SWITCH: all positions flattened")
        except Exception as e:  # noqa: BLE001
            log.error("kill switch close failed: %s", e)
