"""Feature engineering. STRICTLY CAUSAL: every feature at row t uses data <= t only.
This is the #1 place backtests lie — we prove causality in tests/test_pipeline.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature contract: any column listed here is a model input. Keep it explicit so
# train/live skew is impossible — live uses the exact same function.
FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "hl_range", "close_loc",       # intra-bar structure (uses bar t's own OHLC = known at close)
    "atr_pct",
    "rsi_14",
    "stoch_k",
    "ema_fast_gap", "ema_slow_gap", "ema_cross",
    "macd", "macd_signal", "macd_hist",
    "bb_pos", "bb_width",
    "adx_proxy", "trend_eff",
    "vol_ratio", "vol_z",
    "hour_sin", "hour_cos", "dow",
]


def _wilder_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = _wilder_ema(d.clip(lower=0), n)
    dn = _wilder_ema((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append all feature columns. Input: completed candles only, sorted by time."""
    out = df.copy()
    close, high, low, vol = (out[c].astype(float) for c in ("close", "high", "low", "volume"))

    # --- returns (fully past-looking) ---
    logret = np.log(close / close.shift(1))
    for n in (1, 3, 6, 12, 24):
        out[f"ret_{n}"] = logret.rolling(n).sum()

    # --- intra-bar structure: known the instant the bar closes ---
    out["hl_range"] = (high - low) / close
    out["close_loc"] = np.where((high - low) > 0, (close - low) / (high - low), 0.5)

    # --- volatility ---
    tr = true_range(out)
    atr = _wilder_ema(tr, 14)
    out["atr_pct"] = atr / close

    # --- momentum oscillators ---
    out["rsi_14"] = rsi(close, 14)
    ll = low.rolling(14).min()
    hh = high.rolling(14).max()
    out["stoch_k"] = np.where((hh - ll) > 0, 100 * (close - ll) / (hh - ll), 50.0)

    # --- trend ---
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=48, adjust=False).mean()
    out["ema_fast_gap"] = (close - ema_fast) / close
    out["ema_slow_gap"] = (close - ema_slow) / close
    out["ema_cross"] = (ema_fast - ema_slow) / close
    macd_line = ema_fast - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    out["macd"] = macd_line / close
    out["macd_signal"] = macd_sig / close
    out["macd_hist"] = (macd_line - macd_sig) / close
    bb_mid = close.rolling(20).mean()
    bb_sd = close.rolling(20).std()
    out["bb_pos"] = (close - bb_mid) / (2 * bb_sd + 1e-12)
    out["bb_width"] = (4 * bb_sd) / (bb_mid + 1e-12)

    # --- regime features ---
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = _wilder_ema(pd.Series(
        np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=out.index), 14)
    minus_dm = _wilder_ema(pd.Series(
        np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=out.index), 14)
    adx_proxy = 100 * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm + 1e-12)
    out["adx_proxy"] = adx_proxy
    # trend efficiency: net move / path length (Kaufman)
    change = (close - close.shift(20)).abs()
    path = close.diff().abs().rolling(20).sum()
    out["trend_eff"] = (change / (path + 1e-12)).clip(0, 1)

    # --- volume / microstructure ---
    v_mean = vol.rolling(48).mean()
    out["vol_ratio"] = vol / (v_mean + 1e-9)
    out["vol_z"] = (vol - v_mean) / (vol.rolling(48).std() + 1e-9)

    # --- calendar (UTC; known in advance, hence causal) ---
    hrs = out["time"].dt.hour + out["time"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hrs / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hrs / 24)
    out["dow"] = out["time"].dt.dayofweek

    out["atr"] = atr  # kept for sizing; not a model input
    return out


FEATURE_COLS = [f for f in FEATURES]


def future_return(df: pd.DataFrame, horizon: int = 12) -> pd.Series:
    """SUPERVISED TARGET — uses FUTURE data. Never feed into features.
    Sign-aligned label: +1 if price rises over next `horizon` bars (long wins)."""
    return (df["close"].shift(-horizon) / df["close"] - 1.0) * 1e4  # in bps
