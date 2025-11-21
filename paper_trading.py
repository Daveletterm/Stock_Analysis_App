"""Paper trading integration for Alpaca's paper API."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable


class AlpacaAPIError(RuntimeError):
    """Raised when Alpaca returns a non-success response."""

    def __init__(self, status_code: int, message: str, *, payload: Any | None = None):
        self.status_code = status_code
        self.api_message = message or ""
        self.payload = payload
        super().__init__(f"Alpaca error {status_code}: {message}")


class NoAvailableBidError(AlpacaAPIError):
    """Raised when Alpaca rejects option exits because no bid is available."""

    code = "no_available_bid"


class OptionCloseRejectedError(AlpacaAPIError):
    """Raised when Alpaca rejects a closing option trade as uncovered."""

    code = "option_close_rejected"

import requests

logger = logging.getLogger("paper_trading")

DEFAULT_BASE_URL = "https://paper-api.alpaca.markets/v2"


class AlpacaPaperBroker:
    """Small wrapper around the Alpaca paper trading REST API.

    This broker is intended for paper trading only and includes helpers
    specific to paper account cleanup that must not be used for live
    trading.
    """

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
            raise AlpacaAPIError(response.status_code, message, payload=data)
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

    def list_open_orders_for_symbol(
        self,
        symbol: str,
        *,
        asset_class: str | None = None,
        side: str | None = None,
        orders: Iterable[Dict[str, Any]] | None = None,
    ) -> list[Dict[str, Any]]:
        """Return open orders for the requested symbol, filtered by asset class/side."""

        symbol_cmp = symbol.replace(" ", "").upper()
        active_status = {"new", "accepted", "open", "pending_new", "partially_filled"}
        order_source = orders if orders is not None else self.list_orders(status="open", limit=200)
        matches: list[Dict[str, Any]] = []
        for order in order_source:
            try:
                if str(order.get("symbol", "")).replace(" ", "").upper() != symbol_cmp:
                    continue
                if str(order.get("status", "")).lower() not in active_status:
                    continue
                if asset_class and str(order.get("asset_class", "")).lower() != asset_class.lower():
                    continue
                if side and str(order.get("side", "")).lower() != side.lower():
                    continue
                matches.append(order)
            except Exception:
                continue
        return matches

    def submit_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Submitting paper order: %s", payload)
        try:
            return self._request("POST", "/orders", json=payload)
        except AlpacaAPIError as exc:
            side = str(payload.get("side", "")).lower()
            asset_class = str(payload.get("asset_class", "")).lower()
            message = exc.api_message.lower()
            is_close = str(payload.get("position_effect", "")).lower() == "close"
            if (
                side == "sell"
                and asset_class == "option"
                and exc.status_code == 403
                and "no available bid for symbol" in message
            ):
                raise NoAvailableBidError(exc.status_code, exc.api_message, payload=exc.payload) from exc
            if (
                side == "sell"
                and asset_class == "option"
                and exc.status_code == 403
                and "uncovered option" in message
            ):
                logger.warning(
                    "Option close rejected for %s: %s",
                    payload.get("symbol", "unknown"),
                    exc.api_message,
                )
                if is_close:
                    return {
                        "status": "rejected_uncovered",
                        "symbol": payload.get("symbol"),
                        "side": side,
                        "asset_class": asset_class,
                        "message": exc.api_message,
                    }
                raise OptionCloseRejectedError(
                    exc.status_code, exc.api_message, payload=exc.payload
                ) from exc
            raise

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/orders/{order_id}")

    def close_position(self, symbol: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/positions/{symbol}")

    def delete_position(self, symbol: str) -> dict:
        """Hard-delete a position from the paper account via Alpaca API.

        This is a last-resort cleanup for zombie positions that cannot be
        closed with normal orders (no bid, uncovered, etc.). It MUST NOT
        be used with live trading accounts.
        """
        if not symbol:
            raise ValueError("symbol is required for delete_position")

        path = f"/positions/{symbol}"
        logger.warning("Attempting hard delete for zombie position %s", symbol)
        # This uses the underlying _request helper which already wraps errors
        return self._request("DELETE", path)

