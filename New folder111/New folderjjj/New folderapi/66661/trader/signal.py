"""Regime-aware ensemble signal engine.

The teaching core — the exact process for turning free data streams into edges:

  1. FEATURES   — causal indicators from features.py (identical code live & backtest,
                  so there is zero train/live skew).
  2. LABELS     — sign-aligned future return over `horizon` bars in bps, i.e. the
                  model learns "does LONG make money here after costs?".
  3. MODELS     — two weakly-correlated learners:
       * logistic regression (linear, low variance, hard to overfit)
       * gradient boosting    (non-linear interactions, capped depth)
     Ensemble score = mean of the two, in [0, 1] = P(long wins).
  4. DECISION   — score > 0.5 + th/2 -> LONG; < 0.5 - th/2 -> SHORT; else FLAT.
     An asymmetric threshold (from walk-forward) is the anti-overfit dial: it only
     trades when the ensemble is confident enough to survive costs.
  5. REGIME     — trend_eff + atr_pct + adx_proxy buckets: range / trend / volatile.
     Optional veto: in "range" regime, require higher confidence to trade.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .features import FEATURES, compute_features, future_return

log = logging.getLogger(__name__)


def build_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    """SINGLE OWNER of the inference path: causal features + regime, rows ready
    for the models. build_dataset() is this plus the (future-looking) label, so
    train and live feature frames cannot skew."""
    feat = compute_features(df)
    feat["regime"] = classify_regime(feat)
    return feat.dropna(subset=FEATURES).reset_index(drop=True)


def build_dataset(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Inference frame + label. Label is LAST and must never leak into X
    (guaranteed by explicit column selection everywhere)."""
    out = build_inference_frame(df)
    out["label"] = future_return(out, horizon=horizon)
    return out


def classify_regime(feat: pd.DataFrame) -> pd.Series:
    """Simple, robust regime bucket from causal features only."""
    te = feat["trend_eff"]
    vol = feat["atr_pct"]
    vol_hi = vol > vol.rolling(288, min_periods=50).quantile(0.8)
    trending = te > 0.35
    regime = np.where(vol_hi, "volatile", np.where(trending, "trend", "range"))
    return pd.Series(regime, index=feat.index)


def make_models(seed: int = 7) -> list:
    """Two diverse learners. Diversity = lower ensemble variance = higher Sharpe."""
    lin = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=500, random_state=seed)),
    ])
    gbm = GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=seed)
    return [lin, gbm]


class EnsembleSignal:
    """Fit once per walk-forward window; predict causally on unseen bars."""

    def __init__(self, threshold: float = 0.55, veto_in_range: bool = True,
                 horizon: int = 12, seed: int = 7):
        self.threshold = threshold
        self.veto_in_range = veto_in_range
        self.horizon = horizon
        self.seed = seed
        self.models: list = []
        self.feature_names: list[str] = list(FEATURES)

    # ---------- training ----------
    def fit(self, ds: pd.DataFrame) -> "EnsembleSignal":
        """ds: output of build_dataset. Rows without a usable label are dropped
        (the most recent `horizon` bars of any live snapshot legitimately have
        no label yet — that's expected, not data loss)."""
        d = ds.dropna(subset=["label"]).copy()
        y = (d["label"] > 0).astype(int).to_numpy()
        if len(np.unique(y)) < 2 or len(d) < 200:
            raise ValueError(f"not enough labelled data to fit: {len(d)} rows")
        X = d[self.feature_names].to_numpy(dtype=float)
        self.models = make_models(self.seed)
        for m in self.models:
            m.fit(X, y)
        # calibration sanity on train (for telemetry only, never a go/no-go)
        p = self.predict_proba_frame(d)
        self.train_acc_ = float(((p > 0.5).astype(int) == y).mean())
        self.train_n_ = int(len(d))
        return self

    # ---------- inference ----------
    def predict_proba_frame(self, ds: pd.DataFrame) -> np.ndarray:
        X = ds[self.feature_names].to_numpy(dtype=float)
        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        return np.mean(probs, axis=0)

    def decide(self, ds_row: pd.DataFrame) -> tuple[str, float, str]:
        """Return (action, score, regime) for exactly ONE inference row
        (a 1-row frame from build_inference_frame)."""
        if len(ds_row) != 1:
            raise ValueError(f"decide() takes exactly one row, got {len(ds_row)}")
        regime = str(ds_row["regime"].iloc[0])
        if not self.models:
            return "FLAT", 0.5, regime
        score = float(self.predict_proba_frame(ds_row)[0])
        th = self.threshold
        action = "FLAT"
        if score >= 0.5 + th / 2:
            action = "LONG"
        elif score <= 0.5 - th / 2:
            action = "SHORT"
        if action != "FLAT" and self.veto_in_range and regime == "range" and abs(score - 0.5) < 0.18:
            action = "FLAT"  # chop + weak conviction = the account killer; veto it
        return action, score, regime
