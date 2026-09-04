"""OANDA v20 REST client. Practice ("demo") endpoints give free candles and
free realistic fills — the whole data pipeline costs $0."""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)


class OandaError(RuntimeError):
    pass


class OandaClientError(OandaError):
    """Non-retryable 4xx: bad token, bad params, rejected order."""
    pass


class OandaClient:
    def __init__(self, token: str, account_id: str, api_base: str,
                 timeout: int = 15, max_retries: int = 4):
        self.account_id = account_id
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        })

    # ---------------- core request helper ----------------
    def _request(self, method: str, url: str, **kwargs) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", "5"))
                    log.warning("rate limited, sleeping %ss", wait)
                    time.sleep(wait)
                    continue
                if r.status_code >= 500:
                    raise OandaError(f"{r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    # 4xx = our request is wrong (auth, params). Retrying cannot
                    # succeed — surface it immediately.
                    raise OandaClientError(f"{r.status_code}: {r.text[:400]}")
                return r.json() if r.content else {}
            except OandaClientError:
                raise  # fail fast — retries cannot fix a 4xx
            except (requests.ConnectionError, requests.Timeout, OandaError) as e:
                last_exc = e
                back = min(30, 2 ** attempt)
                log.warning("request failed (%s), retry in %ss", e, back)
                time.sleep(back)
        raise OandaError(f"exhausted retries: {last_exc}")

    # ---------------- market data (free) ----------------
    def get_candles(self, instrument: str, granularity: str = "M5",
                    count: int = 500, price: str = "M", from_time: str | None = None) -> pd.DataFrame:
        """Completed mid candles -> DataFrame[time, open, high, low, close, volume]."""
        params: dict = {"granularity": granularity, "count": min(count, 5000), "price": price}
        if from_time:
            params["from"] = from_time
        url = f"{self.api_base}/v3/instruments/{instrument}/candles"
        data = self._request("GET", url, params=params)
        rows = []
        for c in data.get("candles", []):
            if not c.get("complete", False):
                continue
            mid = c.get("mid", {})
            rows.append({
                "time": pd.to_datetime(c["time"], utc=True),
                "open": float(mid["o"]), "high": float(mid["h"]),
                "low": float(mid["l"]), "close": float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        return df

    # ---------------- account ----------------
    def account_summary(self) -> dict:
        return self._request("GET", f"{self.api_base}/v3/accounts/{self.account_id}/summary")

    def account_details(self) -> dict:
        return self._request("GET", f"{self.api_base}/v3/accounts/{self.account_id}")

    def open_trades(self) -> list[dict]:
        data = self._request("GET", f"{self.api_base}/v3/accounts/{self.account_id}/openTrades")
        return data.get("trades", [])

    def positions(self) -> list[dict]:
        data = self._request("GET", f"{self.api_base}/v3/accounts/{self.account_id}/positions")
        return data.get("positions", [])

    # ---------------- execution ----------------
    def get_pricing(self, instruments: list[str]) -> list[dict]:
        """Current bid/ask (free) — used for the live spread filter."""
        data = self._request(
            "GET",
            f"{self.api_base}/v3/accounts/{self.account_id}/pricing",
            params={"instruments": ",".join(instruments)},
        )
        return data.get("prices", [])

    def market_order(self, instrument: str, units: int, stop_loss: float, take_profit: float,
                     client_tag: str = "") -> dict:
        """Market order with server-side OCO (stop + take). units signed: +long / -short."""
        body = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(int(units)),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": f"{stop_loss:.5f}", "timeInForce": "GTC"},
                "takeProfitOnFill": {"price": f"{take_profit:.5f}", "timeInForce": "GTC"},
            }
        }
        if client_tag:
            body["order"]["clientExtensions"] = {
                "id": client_tag[:128], "tag": "glm-ensemble-v1", "comment": "auto"
            }
        return self._request("POST", f"{self.api_base}/v3/accounts/{self.account_id}/orders",
                             json=body)

    def close_trade(self, trade_id: str) -> dict:
        return self._request(
            "PUT",
            f"{self.api_base}/v3/accounts/{self.account_id}/trades/{trade_id}/close",
            json={},
        )

    def close_all(self, instrument: str | None = None) -> None:
        for t in self.open_trades():
            if instrument and t.get("instrument") != instrument:
                continue
            try:
                self.close_trade(t["tradeId"])
            except OandaError as e:
                log.error("failed to close trade %s: %s", t.get("tradeId"), e)


def make_client(cfg) -> OandaClient:
    return OandaClient(cfg.oanda_token, cfg.oanda_account_id, cfg.api_base)
