"""Live trading loop: poll completed bars -> inference frame -> ensemble -> risk
gate -> OANDA demo order with OCO. Retrains on a rolling window on a schedule.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from .instruments import pip_size
from .config import Config
from .oanda_client import make_client
from .risk import RiskManager
from .signal import EnsembleSignal, build_dataset, build_inference_frame
from .store import CandleStore

log = logging.getLogger(__name__)


class LiveEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = make_client(cfg)
        self.store = CandleStore(cfg.db_path)
        self.risk = RiskManager(cfg)
        self.sigs: dict[str, EnsembleSignal] = {}   # one model set per instrument
        self.frames: dict[str, pd.DataFrame] = {}   # cached inference frame per instrument
        self.candles: dict[str, pd.DataFrame] = {}
        self.last_bar_time: dict[str, pd.Timestamp] = {}
        self._blocked_emitted: set[tuple[str, str, str]] = set()
        self.broker = None
        if cfg.dry_run:
            from .broker import DryRunBroker
            self.broker = DryRunBroker(
                cfg.dryrun_state_path, cfg.dryrun_start_equity,
                cfg.default_spread_pips, cfg.slippage_pips,
                cfg.skip_hours, cfg.skip_friday_after)
        self.last_retrain = datetime.now(timezone.utc)

    # ---------------- bootstrap ----------------
    def bootstrap(self) -> None:
        for inst in self.cfg.instrument_list:
            df = self.store.load(inst, self.cfg.signal_granularity)
            if len(df) < self.cfg.wf_train_bars + 200:
                log.info("bootstrapping %s from OANDA (free)...", inst)
                fetched = self.client.get_candles(
                    inst, self.cfg.signal_granularity, count=self.cfg.history_bars)
                self.store.upsert(fetched, inst, self.cfg.signal_granularity)
                df = fetched
            self.candles[inst] = df
            self.last_bar_time[inst] = df["time"].iloc[-1] if not df.empty else None
            log.info("%s: %d bars cached (latest %s)", inst, len(df),
                     self.last_bar_time[inst])

    # ---------------- model upkeep ----------------
    def retrain(self) -> None:
        """One fitted model per instrument — never shared across price scales."""
        now = datetime.now(timezone.utc)
        for inst, df in self.candles.items():
            if len(df) < self.cfg.wf_train_bars:
                log.warning("%s: insufficient history to train (%d bars)", inst, len(df))
                continue
            try:
                ds = build_dataset(df.iloc[-self.cfg.wf_train_bars:], horizon=12)
                sig = EnsembleSignal(threshold=self.cfg.signal_threshold).fit(ds)
                self.sigs[inst] = sig
                log.info("%s: model retrained on %d rows (train_acc=%.3f)",
                         inst, sig.train_n_, sig.train_acc_)
            except ValueError as e:
                log.error("%s retrain failed: %s", inst, e)
        self.last_retrain = now

    def _retrain_due(self) -> bool:
        return (datetime.now(timezone.utc) - self.last_retrain).total_seconds() > 7 * 86400

    # ---------------- data upkeep ----------------
    def refresh(self, inst: str) -> None:
        """Fetch completed candles newer than the last cached bar."""
        latest = self.last_bar_time.get(inst)
        try:
            fresh = self.client.get_candles(
                inst, self.cfg.signal_granularity, count=min(100, self.cfg.history_bars),
                from_time=(latest + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                if latest is not None else None)
        except Exception as e:  # noqa: BLE001
            log.error("%s: candle fetch failed: %s", inst, e)
            return
        if fresh.empty:
            return
        self.store.upsert(fresh, inst, self.cfg.signal_granularity)
        df = pd.concat([self.candles.get(inst, fresh), fresh]).drop_duplicates(
            subset="time").sort_values("time").reset_index(drop=True)
        self.candles[inst] = df.tail(self.cfg.history_bars).reset_index(drop=True)
        self.last_bar_time[inst] = df["time"].iloc[-1]
        self.frames.pop(inst, None)  # cached frame is stale after new bars
        if self.broker is not None:  # simulate OCO fills on each new bar (dry-run)
            for _, row in fresh.iterrows():
                self.broker.mark(inst, row)

    # ---------------- decision pipeline ----------------
    def evaluate(self, inst: str) -> None:
        df = self.candles.get(inst)
        if df is None or len(df) < 300:
            return
        frame = self.frames.get(inst)
        if frame is None:  # computed once per batch of new bars, not once per poll
            frame = build_inference_frame(df)
            self.frames[inst] = frame
            self._blocked_emitted.clear()  # per-bar-cycle spam guard, no unbounded growth
        if frame.empty:
            return
        last = frame.iloc[[-1]]
        bar_time = last["time"].iloc[0]
        if self.store.conn.execute(
            "SELECT 1 FROM signals_log WHERE instrument=? AND note LIKE ? LIMIT 1",
            (inst, f"%bar={bar_time}%")).fetchone():
            return  # this completed bar was already evaluated

        sig = self.sigs.get(inst)
        action, score, regime = (
            sig.decide(last) if sig else ("FLAT", 0.5, str(last["regime"].iloc[0])))
        price = float(last["close"].iloc[0])
        atr = float(last["atr"].iloc[0])
        note = f"score={score:.3f} regime={regime} bar={bar_time}"

        if action == "FLAT":
            self._emit(ts=_utc(), instrument=inst, action="FLAT", score=score,
                       regime=regime, price=price, note=note)
            return

        spread_pips = self._current_spread_pips(inst)
        if self.broker is not None:
            self.broker.roll_day(_utc()[:10])
            self.broker.sync_risk(self.risk)  # daily halt + cooldowns work in dry-run
            equity, open_trades = self.broker.equity(), self.broker.open_trades()
        else:
            equity = float(self.client.account_details()["account"]["NAV"])
            open_trades = self.client.open_trades()
        ok, reason, units = self.risk.check_entry(
            instrument=inst, price=price, atr=atr, equity=equity,
            open_trades=open_trades, now=bar_time, spread_pips=spread_pips)
        if not ok:
            # blocked bars are NOT recorded as evaluated: risk state is time-based
            # (cooldown/halt), so a later poll on the same bar may legitimately pass.
            # Only the first block per (bar, reason) is emitted to avoid poll spam.
            key = (inst, str(bar_time), reason)
            if key not in self._blocked_emitted:
                self._blocked_emitted.add(key)
                self._telemetry({"ts": _utc(), "event": "blocked", "instrument": inst,
                                 "reason": reason, "score": score, "bar": str(bar_time)})
            return

        sign = 1 if action == "LONG" else -1
        sl = price - sign * self.cfg.atr_sl_multiple * atr
        tp = price + sign * self.cfg.atr_tp_multiple * atr
        self._emit(ts=_utc(), instrument=inst, action=action, score=score,
                   regime=regime, price=price, units=sign * units, sl=sl, tp=tp, note=note)
        if self.cfg.dry_run:
            tid = self.broker.fill(inst, sign * units, price, atr,
                                   self.cfg.atr_sl_multiple, self.cfg.atr_tp_multiple,
                                   bar_time)
            log.info("[DRY-RUN] %s %s %d units @~%.5f sl=%.5f tp=%.5f (%s)",
                     inst, action, units, price, sl, tp, note)
            self._telemetry({"ts": _utc(), "event": "dry_fill", "instrument": inst,
                             "action": action, "units": sign * units, "tid": tid,
                             "equity": self.broker.equity()})
        else:
            resp = self.client.market_order(
                inst, sign * units, sl, tp,
                client_tag=f"glm-{inst}-{bar_time.strftime('%Y%m%d%H%M')}")
            log.info("ORDER %s %s %d units, resp=%s", inst, action, sign * units,
                     json.dumps(resp)[:200])
            fill = resp.get("orderFillTransaction") or {}
            tid = (fill.get("tradeOpened") or {}).get("tradeId")
            if tid:
                self.risk.register_fill(inst, str(tid), float(fill.get("price", price)),
                                        units, sign)
            self._telemetry({"ts": _utc(), "event": "order_sent", "instrument": inst,
                             "action": action, "units": sign * units, "resp": str(resp)[:300]})

    def reconcile_closures(self) -> None:
        """Detect OCO fills by comparing open trades to locally known ones, and
        feed realized PnL into the risk manager (daily halt / cooldowns)."""
        try:
            now_open = {str(t["tradeId"]) for t in self.client.open_trades()}
        except Exception as e:  # noqa: BLE001
            log.error("closure reconciliation failed: %s", e)
            return
        known = self.risk.state.known_trades
        for tid in [t for t in known if t not in now_open]:
            info = known[tid]
            exit_price = self._mid_price(info["instrument"])
            if exit_price is None:
                # transient pricing failure: keep the record, retry next cycle —
                # dropping it here would permanently lose the PnL from risk state
                log.warning("trade %s closed but no price yet; will retry", tid)
                continue
            pnl = (exit_price - info["entry"]) * info["direction"] * info["units"]
            del known[tid]
            self.risk.on_trade_closed(pnl, info["instrument"], pd.Timestamp(_utc()))
            log.info("trade %s closed, realized %.2f on %s", tid, pnl, info["instrument"])

    # ---------------- helpers ----------------
    def _emit(self, **kw) -> None:
        self.store.log_signal(**kw)
        self._telemetry({"ts": kw.get("ts"), "event": "signal",
                         "instrument": kw.get("instrument"), "action": kw.get("action"),
                         "score": kw.get("score"), "regime": kw.get("regime"),
                         "bar": str(kw.get("note", ""))[-40:]})

    def _mid_price(self, inst: str) -> float | None:
        try:
            p = self.client.get_pricing([inst])[0]
            bids, asks = p["bids"], p["asks"]
            return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2.0
        except Exception:  # noqa: BLE001
            return None

    def _current_spread_pips(self, inst: str) -> float:
        try:
            p = self.client.get_pricing([inst])[0]
            spread = float(p["asks"][0]["price"]) - float(p["bids"][0]["price"])
            return spread / pip_size(inst)
        except Exception as e:  # noqa: BLE001
            log.warning("spread fetch failed for %s: %s (using default)", inst, e)
            return self.cfg.default_spread_pips

    def _telemetry(self, rec: dict) -> None:
        try:
            with open(self.cfg.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    # ---------------- main loop ----------------
    def run(self) -> None:  # pragma: no cover - long-running
        log.info("live engine starting (env=%s, dry_run=%s, instruments=%s)",
                 self.cfg.oanda_environment, self.cfg.dry_run,
                 self.cfg.instrument_list)
        self.bootstrap()
        if self.broker is not None:
            self.broker.count_session()
        self.retrain()
        while True:
            try:
                if self.risk.kill_switch_active():
                    log.warning("kill switch detected -> flattening & standing down")
                    if not self.cfg.dry_run:
                        self.risk.flatten_all(self.client)
                    time.sleep(10)
                    continue
                if self.broker is None:  # dry-run closures come from broker.mark()
                    self.reconcile_closures()
                for inst in self.cfg.instrument_list:
                    if inst not in self.candles:
                        continue  # bootstrap failure on one pair must not kill the rest
                    self.refresh(inst)
                    self.evaluate(inst)
                if self._retrain_due():
                    self.retrain()
            except KeyboardInterrupt:
                log.info("shutting down on user request")
                break
            except Exception as e:  # noqa: BLE001 - loop must survive anything
                log.exception("loop error: %s", e)
            time.sleep(self.cfg.poll_seconds)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:  # pragma: no cover
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from .config import load_config
    LiveEngine(load_config()).run()


if __name__ == "__main__":
    main()
