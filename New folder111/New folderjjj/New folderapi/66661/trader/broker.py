"""Dry-run simulated broker: fills at order-time price, spread-adjusted exactly
like the backtest (pay half-spread + slippage per side), OCO stop/TP checked
against each new bar's extremes (stop-first, conservative). State persists as
JSON so a simulated equity curve accumulates across dry-run sessions.

Live-mode behavior is untouched: LiveEngine only routes to this when dry_run.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .instruments import cost_per_side, price_to_pips

log = logging.getLogger(__name__)


class DryRunStateError(RuntimeError):
    """dryrun_state.json is corrupt or holds hostile values; the broker refuses
    to trade on it rather than silently trading garbage."""
    RECOVERY = ("recovery: delete or move the state file "
                "(a fresh simulated account starts on the next run).")


class DryRunBroker:
    def __init__(self, state_path: str, start_equity: float,
                 spread_pips: float, slippage_pips: float,
                 skip_hours: set[int] | None = None, skip_friday_after: int = 19):
        self.state_path = Path(state_path)
        self.spread_pips = float(spread_pips)
        self.slippage_pips = float(slippage_pips)
        self.skip_hours = skip_hours or set()
        self.skip_friday_after = skip_friday_after
        self.st = self._load(start_equity)

    def _load(self, start_equity: float) -> dict:
        if self.state_path.exists():
            try:
                st = json.loads(self.state_path.read_text(encoding="utf-8"))
                self._validate(st)
                return st
            except (OSError, json.JSONDecodeError) as e:
                log.warning("dry-run state unreadable (%s); starting fresh", e)
            except DryRunStateError:
                raise
        return {"equity": float(start_equity), "day_start_equity": float(start_equity),
                "day_key": "", "realized_today": 0.0, "last_loss_time": {},
                "positions": [], "next_id": 1, "fills": [], "sessions": 0}

    @staticmethod
    def _validate(st: dict) -> None:
        """Structural + value checks. Fail loudly BEFORE any order can trade on
        hostile state — silent garbage is the one unacceptable outcome."""
        if not isinstance(st, dict):
            raise DryRunStateError(f"state is {type(st).__name__}, not an object")
        try:
            eq = float(st["equity"])
            day0 = float(st["day_start_equity"])
            nid = int(st["next_id"])
            sessions = int(st.get("sessions", 0))
        except (KeyError, TypeError, ValueError) as e:
        # hostile values: garbage numbers break every downstream decision
            raise DryRunStateError(f"state has non-numeric core fields: {e}\n"
                                   f"{DryRunStateError.RECOVERY}")
        if eq <= 0 or day0 <= 0:
            raise DryRunStateError(f"equity {eq} must be positive\n{DryRunStateError.RECOVERY}")
        if nid < 1:
            raise DryRunStateError(f"next_id {nid} must be >= 1\n{DryRunStateError.RECOVERY}")
        if sessions < 0:
            raise DryRunStateError(f"sessions {sessions} must be >= 0\n{DryRunStateError.RECOVERY}")
        if not isinstance(st.get("positions"), list):
            raise DryRunStateError("'positions' must be a list\n" + DryRunStateError.RECOVERY)
        if not isinstance(st.get("fills"), list):
            raise DryRunStateError("'fills' must be a list\n" + DryRunStateError.RECOVERY)
        if not isinstance(st.get("last_loss_time", {}), dict):
            raise DryRunStateError("'last_loss_time' must be an object\n" + DryRunStateError.RECOVERY)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.st, indent=1), encoding="utf-8")

    # ------------- portfolio snapshot for the risk gate -------------
    def open_trades(self) -> list[dict]:
        return [
            {"tradeId": str(p["id"]), "instrument": p["instrument"],
             "units": p["direction"] * p["units"], "price": p["entry"]}
            for p in self.st["positions"]
        ]

    def equity(self) -> float:
        return float(self.st["equity"])

    # ------------- order-time fill -------------
    def fill(self, instrument: str, units: int, price: float, atr: float,
             atr_sl_multiple: float, atr_tp_multiple: float, bar_time) -> int:
        sign = 1 if units > 0 else -1
        n = abs(int(units))
        adj = cost_per_side(self.spread_pips, self.slippage_pips, instrument)
        entry = price + sign * adj          # pay the spread side you cross
        tid = str(self.st["next_id"])
        self.st["next_id"] += 1
        self.st["positions"].append({
            "id": tid, "instrument": instrument, "direction": sign, "units": n,
            "entry": entry, "sl": entry - sign * atr_sl_multiple * atr,
            "tp": entry + sign * atr_tp_multiple * atr,
            "entry_time": str(bar_time), "cost_per_side": adj})
        self.save()
        return tid

    # ------------- per-bar marking: OCO + realized PnL -------------
    def mark(self, instrument: str, bar) -> None:
        """bar: row with time/open/high/low/close of a completed candle."""
        ts = pd.Timestamp(bar["time"])
        # weekend guard, identical to the backtester: no fills from Friday
        # SKIP_FRI_AFTER_HOUR_UTC to the Monday open (weekend gap noise)
        if ts.dayofweek >= 5 or (ts.dayofweek == 4 and ts.hour >= self.skip_friday_after):
            return
        hi, lo = float(bar["high"]), float(bar["low"])
        for p in [p for p in self.st["positions"] if p["instrument"] == instrument]:
            exit_price = None
            reason = ""
            if p["direction"] == 1:
                if lo <= p["sl"]:
                    exit_price, reason = p["sl"], "stop"
                elif hi >= p["tp"]:
                    exit_price, reason = p["tp"], "tp"
            else:
                if hi >= p["sl"]:
                    exit_price, reason = p["sl"], "stop"
                elif lo <= p["tp"]:
                    exit_price, reason = p["tp"], "tp"
            if exit_price is None:
                continue
            exit_price -= p["direction"] * p["cost_per_side"]  # pay the exit side
            gross = (exit_price - p["entry"]) * p["direction"] * p["units"]
            self._close(p, exit_price, reason, str(bar["time"]), gross)

    def _close(self, p: dict, exit_price: float, reason: str, at_time: str,
               gross: float) -> None:
        self.st["positions"].remove(p)
        self.st["equity"] += gross
        self.st["fills"].append({
            "tradeId": p["id"], "instrument": p["instrument"],
            "direction": "long" if p["direction"] == 1 else "short",
            "units": p["units"], "entry": p["entry"], "exit": exit_price,
            "entry_time": p["entry_time"], "exit_time": at_time,
            "net_pnl": gross, "reason": reason})
        self.st["realized_today"] += gross
        if gross < 0:
            self.st["last_loss_time"][p["instrument"]] = at_time
        log.info("[DRY-RUN] closed %s %s: %+.2f (%s)",
                 p["instrument"], p["id"], gross, reason)
        self.save()

    # ------------- risk sync (daily halt + cooldowns work in dry-run) -------------
    def roll_day(self, day_key: str) -> None:
        if self.st.get("day_key") != day_key:
            self.st["day_key"] = day_key
            self.st["day_start_equity"] = self.st["equity"]
            self.st["realized_today"] = 0.0
            self.save()

    def count_session(self) -> None:
        """Record one engine run — the session count tells a reviewer how much
        simulated history a report covers."""
        self.st["sessions"] = self.st.get("sessions", 0) + 1
        self.save()

    # ------------- reviewability: state -> backtest-format frames -------------
    def build_report(self) -> tuple[pd.DataFrame, pd.Series]:
        """(trades, equity) in the exact column format the backtest artifacts
        use, so a reviewer reads one format for both simulated and backtested
        trades. Trades carry their own 'instrument' column (fills span
        instruments); equity is cumulative realized equity at each fill time."""
        fills = self.st.get("fills", [])
        trades = pd.DataFrame([{
            "instrument": f["instrument"],
            "direction": f["direction"],
            "units": f["units"],
            "entry_time": f["entry_time"],
            "entry": f["entry"],
            "exit_time": f["exit_time"],
            "exit": f["exit"],
            "net_pnl": f["net_pnl"],
            "reason": f["reason"],
        } for f in fills])
        if not trades.empty:
            trades["net_pips"] = [
                price_to_pips((f["exit"] - f["entry"]) * (1 if f["direction"] == "long" else -1),
                              f["instrument"])
                for f in fills]
        eq = self.st["equity"]
        curve: list[tuple[str, float]] = []
        for f in reversed(fills):          # walk backwards: end equity minus later fills
            curve.append((f["exit_time"], eq))
            eq -= f["net_pnl"]
        curve.reverse()
        equity = pd.Series(
            [e for _, e in curve],
            index=pd.Index([t for t, _ in curve], name="time"), name="equity")
        return trades, equity

    def sync_risk(self, risk) -> None:
        s = risk.state
        s.day_key = self.st.get("day_key") or s.day_key
        s.day_start_equity = self.st.get("day_start_equity", s.day_start_equity)
        s.realized_today = self.st.get("realized_today", 0.0)
        lt: dict[str, pd.Timestamp] = {}
        for inst, ts in self.st.get("last_loss_time", {}).items():
            try:
                lt[inst] = pd.Timestamp(ts)
            except Exception:  # noqa: BLE001
                continue
        s.last_loss_time = lt
