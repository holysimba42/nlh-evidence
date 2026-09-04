#!/usr/bin/env python
"""Thin CLI: argument parsing and output only. All orchestration lives in the
strategy layer (`trader.walkforward.run_instrument_validation`), where it is
unit-testable without process invocation.

Examples:
  python -m trader.run doctor                     # preflight checklist
  python -m trader.run backtest --instrument EUR_USD
  python -m trader.run backtest --all
  python -m trader.run live --dry-run
  python -m trader.run live          # real demo orders
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config

log = logging.getLogger("trader.run")


def cmd_backtest(cfg, instrument: str | None, all_inst: bool) -> int:
    if instrument is not None and instrument != instrument.strip().upper():
        print(f"note: normalizing instrument {instrument!r} -> {instrument.strip().upper()!r}")
        instrument = instrument.strip().upper()
    from .doctor import run_doctor
    if not run_doctor(cfg):
        return 2  # first run without credentials = helpful checklist, exit nonzero

    from .oanda_client import make_client
    from .store import CandleStore, StoreError
    from .walkforward import run_instrument_validation

    client = make_client(cfg)
    try:
        store = CandleStore(cfg.db_path)
    except StoreError as e:
        print(f"ERROR: {e}")
        return 2
    targets = cfg.instrument_list if all_inst else (
        [instrument] if instrument else cfg.instrument_list)
    for inst in targets:
        print(f"\n=== {inst} ===")
        try:
            out = run_instrument_validation(client, store, inst, cfg)
        except ValueError as e:
            print(f"skip: {e}")
            continue
        if out is None:
            print(f"no candles returned for {inst} (check token/account)")
            continue
        print(f"bars: {out['bars']}  ({out['span']})")
        print(f"OOS bars: {out['oos_bars']} across {len(out['windows'])} walk-forward windows")
    return 0


def cmd_live(cfg, dry_run_override: bool | None) -> int:
    from .doctor import run_doctor
    if not run_doctor(cfg):
        return 2
    if dry_run_override is not None:
        cfg.dry_run = dry_run_override
    from .live import LiveEngine
    LiveEngine(cfg).run()
    return 0


def cmd_report_dryrun(cfg) -> int:
    """Thin IO: read dry-run state, write artifacts, print a summary."""
    from .broker import DryRunBroker, DryRunStateError
    from .artifacts import write_artifacts
    try:
        br = DryRunBroker(cfg.dryrun_state_path, cfg.dryrun_start_equity,
                          cfg.default_spread_pips, cfg.slippage_pips)
    except DryRunStateError as e:
        print(f"ERROR: {e}")
        return 2
    fills = br.st.get("fills", [])
    print(f"dry-run sessions: {br.st.get('sessions', 0)}")
    print(f"simulated equity: {br.equity():.2f} "
          f"(start {cfg.dryrun_start_equity:.2f}, "
          f"net {br.equity() - cfg.dryrun_start_equity:+.2f})")
    print(f"open positions: {len(br.st['positions'])}")
    print(f"closed fills: {len(fills)}")
    if not fills:
        print("nothing to report yet — run 'python -m trader.run live --dry-run' first")
        return 1
    trades, equity = br.build_report()
    write_artifacts(cfg.artifacts_dir, "DRYRUN", None, trades, equity,
                    f"dry-run report: {len(fills)} fills, {len(br.st['positions'])} open, "
                    f"equity {br.equity():.2f}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="trader")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="preflight checklist (credentials, API, store)")

    bt = sub.add_parser("backtest", help="walk-forward backtest on free OANDA history")
    bt.add_argument("--instrument", default=None)
    bt.add_argument("--all", action="store_true")

    lv = sub.add_parser("live", help="run live loop (demo by default)")
    lv.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
    lv.add_argument("--execute", dest="dry_run", action="store_false", default=None)

    rp = sub.add_parser("report", help="write review artifacts from accumulated runs")
    rp.add_argument("--dryrun", action="store_true",
                    help="report the dry-run state (equity curve + fills table)")

    args = ap.parse_args(argv)
    cfg = load_config()
    if args.cmd == "doctor":
        from .doctor import run_doctor
        return 0 if run_doctor(cfg) else 2
    if args.cmd == "backtest":
        return cmd_backtest(cfg, args.instrument, args.all)
    if args.cmd == "live":
        return cmd_live(cfg, args.dry_run)
    if args.cmd == "report":
        if args.dryrun:
            return cmd_report_dryrun(cfg)
        print("nothing to report: use --dryrun")
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
