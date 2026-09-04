"""Preflight doctor — the no-credential first run must be a checklist, never a
traceback. Also run implicitly before backtest/live startups."""
from __future__ import annotations

import re
from pathlib import Path

from .config import Config

_TOKEN_RE = re.compile(r"^[0-9a-f]{32,128}$")            # OANDA tokens are hex
_ACCOUNT_RE = re.compile(r"^\d{3}-\d{3}-\d{6,7}-\d{3}$")  # e.g. 101-004-1234567-001
_FAILS: list[str] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _FAILS.append(name)


def run_doctor(cfg: Config) -> bool:
    """Print PASS/FAIL per item; True iff all pass."""
    del _FAILS[:]
    print("\n=== doctor ===")

    env_ok = Path(".env").exists()
    _check("env file (.env)", env_ok,
           "found" if env_ok else "missing — run:  cp .env.example .env")

    token_ok = bool(_TOKEN_RE.match(cfg.oanda_token or ""))
    acct_ok = bool(_ACCOUNT_RE.match(cfg.oanda_account_id or ""))
    _check("OANDA token format", token_ok,
           "set" if token_ok else
           "paste your practice token as OANDA_TOKEN=... in .env (OANDA -> Manage API Access)")
    _check("OANDA account ID format", acct_ok,
           "set" if acct_ok else
           "paste your account id as OANDA_ACCOUNT_ID=... in .env (practice console)")

    if token_ok and acct_ok:
        try:
            from .oanda_client import OandaClient
            client = OandaClient(cfg.oanda_token, cfg.oanda_account_id, cfg.api_base)
            summary = client.account_summary()
            acct = summary.get("account", {})
            _check("practice API connectivity", True,
                   str(acct.get("id", "ok")) + f"  currency={acct.get('currency', '?')}")
            try:
                details = client.account_details()["account"]
                avail = {i["id"] for i in details.get("instruments", [])}
                missing = [i for i in cfg.instrument_list if i not in avail]
                _check("instruments available on account", not missing,
                       "all present" if not missing else f"missing: {missing}")
            except Exception as e:  # noqa: BLE001
                _check("instruments available on account", False, str(e)[:120])
        except Exception as e:  # noqa: BLE001
            _check("practice API connectivity", False, str(e)[:120])
    else:
        _check("practice API connectivity", False, "skipped — credentials missing/invalid")

    try:
        from .store import CandleStore
        st = CandleStore(cfg.db_path)
        st.conn.execute("CREATE TABLE IF NOT EXISTS _doctor_probe (x INTEGER)")
        st.conn.execute("INSERT INTO _doctor_probe VALUES (1)")
        st.conn.commit()
        st.conn.execute("DROP TABLE _doctor_probe")
        st.conn.commit()
        st.close()
        _check("candle store readable/writable", True, cfg.db_path)
    except Exception as e:  # noqa: BLE001
        _check("candle store readable/writable", False, str(e)[:160])

    try:
        probe = Path(cfg.artifacts_dir) / "_doctor_probe"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        _check("artifacts dir writable", True, cfg.artifacts_dir)
    except Exception as e:  # noqa: BLE001
        _check("artifacts dir writable", False,
               f"{cfg.artifacts_dir}: {str(e)[:100]} — fix permissions or set ARTIFACTS_DIR")

    if _FAILS:
        print("\nFAILURES: " + ", ".join(_FAILS))
        print(
            "\nTo fix credentials:\n"
            "  1. Log in at https://www.oanda.com/demo (practice account, free)\n"
            "  2. My Account -> Manage API Access -> Generate token\n"
            "  3. If needed: cp .env.example .env\n"
            "  4. Edit .env:  OANDA_TOKEN=<token>   OANDA_ACCOUNT_ID=<id>\n"
            "  5. Re-run:  python -m trader.run doctor\n")
        return False
    print("\nALL CHECKS PASSED")
    return True
