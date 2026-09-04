"""Central configuration. All knobs are env-driven so ops never touch code."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OANDA (free practice/demo endpoints) ---
    oanda_token: str = ""
    oanda_account_id: str = ""
    oanda_environment: str = "demo"
    allow_live: bool = False  # hard safety: live endpoints refused unless True

    # --- Universe & timeframe ---
    instruments: str = "EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD"
    signal_granularity: str = "M5"
    history_bars: int = 3000          # bootstrap candles per instrument (max 5000)

    # --- Risk (the non-negotiables) ---
    risk_per_trade_pct: float = 0.4   # % equity risked between entry and stop
    max_daily_loss_pct: float = 1.5   # day PnL floor -> halt trading until next UTC day
    max_open_trades: int = 3
    max_trades_per_instrument: int = 1
    max_leverage: float = 5.0         # notional / equity cap
    daily_profit_lock_pct: float = 0.0  # optional lock: stop trading above this day gain
    cooldown_after_loss_minutes: int = 30
    max_spread_pips: float = 2.0      # refuse entries when spread wider than this
    # max |net exposure| per currency, in equity multiples. Must exceed
    # max_leverage so a single full-size position can open; correlated stacking
    # (2nd, 3rd USD pair in the same direction) then breaches and is blocked.
    max_ccy_exposure_x: float = 6.0

    # --- Strategy defaults (walk-forward optimizer may override) ---
    signal_threshold: float = 0.55
    atr_sl_multiple: float = 1.6
    atr_tp_multiple: float = 2.4
    skip_hours_utc: str = "21,22,23"  # rollover/thin liquidity
    skip_friday_after_hour_utc: int = 19  # avoid weekend gap exposure

    # --- Costs (demo-conservative) ---
    default_spread_pips: float = 1.0
    slippage_pips: float = 0.3

    # --- Runtime ---
    db_path: str = "data/candles.db"
    poll_seconds: int = 5
    dry_run: bool = True              # True = compute + log signals, no orders
    kill_switch_file: str = "data/KILL"
    telemetry_path: str = "data/telemetry.jsonl"
    artifacts_dir: str = "artifacts"
    dryrun_state_path: str = "data/dryrun_state.json"
    dryrun_start_equity: float = 100_000.0
    log_level: str = "INFO"

    # --- Walk-forward validation ---
    wf_train_bars: int = 2000
    wf_test_bars: int = 500
    wf_step_bars: int = 500

    @property
    def instrument_list(self) -> list[str]:
        return [x.strip().upper() for x in self.instruments.split(",") if x.strip()]

    @property
    def skip_hours(self) -> set[int]:
        return {int(h) for h in self.skip_hours_utc.split(",") if h.strip()}

    @property
    def skip_friday_after(self) -> int:
        return self.skip_friday_after_hour_utc

    @property
    def api_base(self) -> str:
        if self.oanda_environment == "live":
            if not self.allow_live:
                raise RuntimeError("oanda_environment=live but allow_live=false (safety interlock)")
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"


def load_config() -> Config:
    return Config()
