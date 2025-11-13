"""Paper trading integration for Alpaca's paper API."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable

import requests

logger = logging.getLogger("paper_trading")

DEFAULT_BASE_URL = "https://paper-api.alpaca.markets/v2"


class AlpacaPaperBroker:
    """Small wrapper around the Alpaca paper trading REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        key_id: str | None = None,
        secret_key: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("ALPACA_PAPER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.key_id = key_id or os.getenv("ALPACA_PAPER_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
        self.secret_key = secret_key or os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
        self._session = session or requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        logger.debug("AlpacaPaperBroker configured: base=%s enabled=%s", self.base_url, self.enabled)

    @property
    def enabled(self) -> bool:
        return bool(self.key_id and self.secret_key)

    def _headers(self) -> Dict[str, str]:
        if not self.enabled:
            raise RuntimeError("Alpaca paper trading credentials are not configured")
        return {
            "APCA-API-KEY-ID": self.key_id or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not path.startswith("/"):
            raise ValueError("API path must start with '/'")
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        req_headers = {**self._headers(), **headers}
        logger.debug("Alpaca %s %s", method, url)
        response = self._session.request(method, url, headers=req_headers, timeout=15, **kwargs)
        if response.status_code == 204:
            return {}
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - unexpected HTML/plain response
            logger.error("Non-JSON response from Alpaca: %s", response.text[:200])
            raise RuntimeError("Unexpected response from Alpaca") from exc
        if response.status_code >= 400:
            message = data.get("message") if isinstance(data, dict) else str(data)
            raise RuntimeError(f"Alpaca error {response.status_code}: {message}")
        return data

    def get_account(self) -> Dict[str, Any]:
        return self._request("GET", "/account")

    def get_positions(self) -> Iterable[Dict[str, Any]]:
        data = self._request("GET", "/positions")
        return data if isinstance(data, list) else []

    def list_orders(self, status: str = "open", limit: int = 50, nested: bool = True) -> Iterable[Dict[str, Any]]:
        params = {"status": status, "limit": limit, "nested": str(nested).lower()}
        data = self._request("GET", "/orders", params=params)
        return data if isinstance(data, list) else []

    def submit_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Submitting paper order: %s", payload)
        return self._request("POST", "/orders", json=payload)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/orders/{order_id}")

    def close_position(self, symbol: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/positions/{symbol}")

