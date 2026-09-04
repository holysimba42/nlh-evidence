"""Instrument-name conventions — the ONE home.

pip sizes, currency parsing, and exposure-leg mapping are system-wide FX
conventions, not concerns of any single consumer. A convention change lands
here and nowhere else. (Moved verbatim from backtest.py and risk.py.)
"""
from __future__ import annotations

PIP_SIZE = {"JPY": 0.01}  # JPY quote pairs: pip = 0.01; everything else 0.0001


def pip_size(instrument: str) -> float:
    quote = instrument.split("_")[-1]
    return PIP_SIZE.get(quote, 0.0001)


def price_to_pips(price_diff: float, instrument: str) -> float:
    return price_diff / pip_size(instrument)


def cost_per_side(spread_pips: float, slippage_pips: float, instrument: str) -> float:
    """Per-side transaction cost in PRICE units (one home for both money paths):
    pay half the spread plus slippage on entry AND on exit. Same function drives
    the backtest's round_cost and the dry-run broker's fill adjustment."""
    return (spread_pips / 2.0 + slippage_pips) * pip_size(instrument)


def parse_currencies(instrument: str) -> tuple[str, str]:
    """EUR_USD -> ('EUR', 'USD'); USD_JPY -> ('USD', 'JPY'). Normalizes case
    and whitespace: ' eur_usd ' must not create separate exposure buckets."""
    base, _, quote = instrument.strip().upper().partition("_")
    return base, quote


def legs_usd(instrument: str, units: float, price: float) -> list[tuple[str, float]]:
    """Signed per-currency exposure legs, USD-equivalent where determinable.
    - XXX_USD: quote leg is already USD.
    - USD_XXX: units*price of quote ccy == `units` USD worth, so the quote leg
      normalizes to -units USD (JPY correctness: 12_500 USD_JPY units =
      12_500 USD exposure, not 1.9M raw JPY).
    - crosses (no USD leg): stay in raw quote-currency units (documented limit;
      not in the default universe)."""
    base, quote = parse_currencies(instrument)
    if quote == "USD":
        return [(base, units * price), ("USD", -units * price)]
    if base == "USD":
        return [("USD", float(units)), (quote, -float(units))]
    return [(base, units * price), (quote, -units * price)]
