"""Paper trading integration for Alpaca's paper API."""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd

import requests

try:  # Python 3.9+; allow backport on older runtimes
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for Python <3.9
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore
    except Exception:
        ZoneInfo = None  # type: ignore


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

    def get_portfolio_history(
        self,
        period: str = "1D",
        timeframe: str = "15Min",
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """Return account equity history for the requested window."""

        params = {
            "period": period,
            "timeframe": timeframe,
            "extended_hours": str(bool(extended_hours)).lower(),
        }
        data = self._request("GET", "/account/portfolio/history", params=params)
        return data if isinstance(data, dict) else {}

    def list_trade_activities(
        self,
        activity_types: str = "FILL",
        date: _dt.date | None = None,
        page_size: int = 100,
    ) -> Iterable[Dict[str, Any]]:
        """Return trade activities (fills) for the given date from Alpaca.

        This is used to compute realized P/L per fill.
        """
        params: Dict[str, Any] = {"activity_types": activity_types, "page_size": page_size}
        if date is not None:
            # Alpaca expects YYYY-MM-DD in local account timezone
            params["date"] = date.isoformat()

        data = self._request("GET", "/account/activities", params=params)
        return data if isinstance(data, list) else []

    def get_activities(
        self,
        activity_types: str | Iterable[str] | None = None,
        *,
        date: _dt.date | str | None = None,
        page_size: int | None = None,
    ) -> Iterable[Dict[str, Any]]:
        """Return account activities such as fills for the requested date."""

        params: Dict[str, Any] = {}
        if activity_types:
            if isinstance(activity_types, str):
                params["activity_types"] = activity_types
            else:
                params["activity_types"] = ",".join(activity_types)
        if date:
            params["date"] = date.isoformat() if isinstance(date, _dt.date) else str(date)
        if page_size:
            params["page_size"] = int(page_size)

        data = self._request("GET", "/account/activities", params=params)
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


def _resolve_timezone(name: str | None = None) -> _dt.tzinfo:
    """Resolve the app timezone from env or local settings."""

    tz_name = name or os.getenv("APP_TIMEZONE")
    if tz_name and ZoneInfo:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            logger.warning("Falling back to local timezone; invalid APP_TIMEZONE=%s", tz_name)
    try:
        local_tz = _dt.datetime.now().astimezone().tzinfo
        if local_tz:
            return local_tz
    except Exception:
        pass
    return _dt.timezone.utc


def _parse_alpaca_timestamp(value: Any, tz: _dt.tzinfo) -> Optional[_dt.datetime]:
    """Parse Alpaca timestamp strings into timezone-aware datetimes."""

    if not value:
        return None
    if isinstance(value, _dt.datetime):
        dt_val = value
    elif isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned.replace("Z", "+00:00")
        try:
            dt_val = _dt.datetime.fromisoformat(cleaned)
        except ValueError:
            return None
    else:
        return None
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=_dt.timezone.utc)
    return dt_val.astimezone(tz)


CSV_COLUMNS = [
    "row_type",
    "date",
    "timestamp",
    "equity",
    "cash",
    "buying_power",
    "portfolio_value",
    "symbol",
    "asset_class",
    "qty",
    "side",
    "avg_entry_price",
    "current_price",
    "market_value",
    "unrealized_pl",
    "unrealized_plpc",
    "order_type",
    "time_in_force",
    "submitted_at",
    "filled_at",
    "filled_avg_price",
    "status",
    "order_id",
    "mode_or_strategy",
    "strategy_name",
    "underlying_symbol",
    "notes",
    "realized_pl",
    "realized_plpc",
]

# Backwards compatibility alias
PAPER_TRADES_COLUMNS = CSV_COLUMNS


def _build_realized_pnl_map(
    broker: AlpacaPaperBroker,
    snapshot_date: _dt.date,
    tzinfo: _dt.tzinfo,
) -> dict[str, dict[str, Any]]:
    """Return a mapping from Alpaca activity id or order id to realized P/L info."""

    activities = broker.list_trade_activities(date=snapshot_date)
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(activities, list):
        return result

    for act in activities:
        if not isinstance(act, dict):
            continue
        activity_type = str(act.get("activity_type") or "").lower()
        if activity_type not in ("fill", "trade"):
            continue

        raw_pl = act.get("profit_loss")
        raw_plpc = act.get("profit_loss_pct")

        realized_pl = None
        if raw_pl is not None:
            try:
                realized_pl = float(raw_pl)
            except Exception:
                realized_pl = None

        if realized_pl is None:
            net_amount = act.get("net_amount")
            side = str(act.get("side") or "").lower()
            qty = act.get("qty") or act.get("quantity")
            price = act.get("price")
            try:
                qty_f = float(qty) if qty is not None else None
                price_f = float(price) if price is not None else None
                if qty_f and price_f:
                    realized_pl = float(net_amount) if net_amount is not None else None
            except Exception:
                realized_pl = None

        realized_plpc = None
        if raw_plpc is not None:
            try:
                realized_plpc = float(raw_plpc)
            except Exception:
                realized_plpc = None

        if realized_pl is not None and realized_plpc is None:
            qty = act.get("qty") or act.get("quantity")
            price = act.get("price")
            try:
                qty_f = float(qty) if qty is not None else None
                price_f = float(price) if price is not None else None
                notional = (qty_f or 0.0) * (price_f or 0.0)
                if notional:
                    realized_plpc = realized_pl / notional
            except Exception:
                realized_plpc = None

        key = str(act.get("order_id") or act.get("id") or "").strip()
        if not key:
            continue

        result[key] = {
            "realized_pl": realized_pl,
            "realized_plpc": realized_plpc,
            "symbol": act.get("symbol"),
        }

    return result


def get_daily_account_snapshot(
    broker: AlpacaPaperBroker,
    target_date: _dt.date | None = None,
    *,
    tz: _dt.tzinfo | None = None,
) -> dict[str, Any]:
    """Collect account, positions, and orders for the target date."""

    if broker is None:
        raise ValueError("broker is required for snapshot")
    if not broker.enabled:
        raise RuntimeError("Paper trading credentials are not configured")

    tzinfo = tz or _resolve_timezone()
    snapshot_date = target_date or _dt.datetime.now(tzinfo).date()

    account = broker.get_account()
    positions = list(broker.get_positions())
    orders = list(broker.list_orders(status="all", limit=500))
    order_lookup = {o.get("id"): o for o in orders if o.get("id")}

    realized_pnl_map = _build_realized_pnl_map(broker, snapshot_date, tzinfo)

    orders_for_day: list[dict[str, Any]] = []
    for order in orders:
        submitted_dt = _parse_alpaca_timestamp(order.get("submitted_at"), tzinfo)
        filled_dt = _parse_alpaca_timestamp(order.get("filled_at"), tzinfo)
        reference_dt = filled_dt or submitted_dt
        if reference_dt and reference_dt.date() == snapshot_date:
            cloned = dict(order)
            if submitted_dt:
                cloned["_submitted_at_local"] = submitted_dt
            if filled_dt:
                cloned["_filled_at_local"] = filled_dt
            order_id = str(order.get("id") or "").strip()
            if order_id and order_id in realized_pnl_map:
                rp = realized_pnl_map[order_id]
                cloned["realized_pl"] = rp.get("realized_pl")
                cloned["realized_plpc"] = rp.get("realized_plpc")
            orders_for_day.append(cloned)

    orders_for_day.sort(
        key=lambda o: o.get("_filled_at_local") or o.get("_submitted_at_local") or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc),
        reverse=True,
    )

    fills_for_day: list[dict[str, Any]] = []
    try:
        activities = broker.get_activities(activity_types="FILL", date=snapshot_date)
    except Exception:
        logger.exception("Failed to fetch fill activities for %s", snapshot_date)
        activities = []

    for activity in activities or []:
        tx_time = _parse_alpaca_timestamp(activity.get("transaction_time"), tzinfo)
        activity_date = tx_time.date() if tx_time else None
        if activity_date and activity_date != snapshot_date:
            continue

        merged: dict[str, Any] = dict(activity)
        related_order = order_lookup.get(activity.get("order_id"))

        if related_order:
            merged.setdefault("symbol", related_order.get("symbol"))
            merged.setdefault("asset_class", related_order.get("asset_class"))
            merged.setdefault("type", related_order.get("type"))
            merged.setdefault("time_in_force", related_order.get("time_in_force"))
            merged.setdefault("submitted_at", related_order.get("submitted_at"))
            merged.setdefault("side", related_order.get("side"))
            merged.setdefault("status", related_order.get("status"))
            merged.setdefault("filled_avg_price", related_order.get("filled_avg_price"))
            merged.setdefault("qty", related_order.get("qty"))
            merged.setdefault("filled_qty", related_order.get("filled_qty"))
            merged.setdefault("order_class", related_order.get("order_class"))
            merged.setdefault("mode_or_strategy", related_order.get("mode_or_strategy"))
            merged.setdefault("strategy_name", related_order.get("strategy_name"))
            merged.setdefault("underlying_symbol", related_order.get("underlying_symbol"))
            submitted_dt = _parse_alpaca_timestamp(related_order.get("submitted_at"), tzinfo)
            if submitted_dt:
                merged["_submitted_at_local"] = submitted_dt
        merged["_filled_at_local"] = tx_time or _parse_alpaca_timestamp(activity.get("processed_at"), tzinfo)

        merged.setdefault("symbol", activity.get("symbol"))
        merged.setdefault("qty", activity.get("quantity") or activity.get("cum_qty") or activity.get("qty"))
        merged.setdefault("side", activity.get("side"))
        merged.setdefault("filled_avg_price", activity.get("price") or activity.get("price_per_share"))

        if "profit_loss" in activity:
            merged.setdefault("realized_pl", activity.get("profit_loss"))
        if "profit_loss_pct" in activity:
            merged.setdefault("realized_plpc", activity.get("profit_loss_pct"))
        order_id_key = str(activity.get("order_id") or "").strip()
        if order_id_key and order_id_key in realized_pnl_map:
            rp = realized_pnl_map[order_id_key]
            merged.setdefault("realized_pl", rp.get("realized_pl"))
            merged.setdefault("realized_plpc", rp.get("realized_plpc"))

        fills_for_day.append(merged)

    fills_for_day.sort(
        key=lambda o: o.get("_filled_at_local") or o.get("_submitted_at_local") or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc),
        reverse=True,
    )

    logger.info(
        "Daily snapshot %s tz=%s: positions=%d orders=%d fills=%d",
        snapshot_date,
        tzinfo,
        len(positions),
        len(orders_for_day),
        len(fills_for_day),
    )

    return {
        "date": snapshot_date,
        "timezone": tzinfo,
        "as_of": _dt.datetime.now(tzinfo),
        "account": account or {},
        "positions": positions,
        "orders": fills_for_day or orders_for_day,
        "fills": fills_for_day,
        "realized_pnl_map": realized_pnl_map,
    }


def _safe_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# Safety helper to keep option exits anchored to premium-based losses.
def option_pnl_percent(avg_entry_price: float | None, current_price: float | None) -> float | None:
    """Return percent change for an option premium for safety exits."""

    if avg_entry_price is None or avg_entry_price <= 0 or current_price is None:
        return None
    return ((current_price - avg_entry_price) / avg_entry_price) * 100.0


def _normalize_timestamp(value: Any, tzinfo: _dt.tzinfo) -> tuple[str | None, _dt.datetime | None]:
    """Return ISO string timestamp and datetime in the configured timezone."""

    if value is None:
        return None, None
    if isinstance(value, _dt.datetime):
        dt_val = value
    else:
        dt_val = _parse_alpaca_timestamp(value, tzinfo)
    if dt_val is None:
        return str(value), None
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=tzinfo)
    else:
        dt_val = dt_val.astimezone(tzinfo)
    return dt_val.isoformat(), dt_val


def _date_string(value: Any, tzinfo: _dt.tzinfo) -> str:
    if isinstance(value, _dt.datetime):
        return value.astimezone(tzinfo).date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    try:
        parsed = _parse_alpaca_timestamp(value, tzinfo)
        if parsed:
            return parsed.date().isoformat()
    except Exception:
        pass
    return _dt.datetime.now(tzinfo).date().isoformat()


def build_paper_trades_export(
    snapshot: dict,
    *,
    mode_or_strategy: str | None = None,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    """Construct a normalized export DataFrame for paper trades."""

    tzinfo = snapshot.get("timezone") or _resolve_timezone()
    as_of = snapshot.get("as_of") or _dt.datetime.now(tzinfo)
    as_of_ts_iso, as_of_dt = _normalize_timestamp(as_of, tzinfo)
    snapshot_date = snapshot.get("date") or (as_of_dt.date() if as_of_dt else _dt.datetime.now(tzinfo).date())
    snapshot_date_str = _date_string(snapshot_date, tzinfo)

    rows: list[dict[str, Any]] = []
    account = snapshot.get("account") or {}
    account_ts_iso, _ = _normalize_timestamp(account.get("updated_at") or as_of_dt, tzinfo)
    account_strategy_name = (
        strategy_name or snapshot.get("strategy_name") or snapshot.get("profile")
    )
    rows.append(
        {
            "row_type": "account_summary",
            "date": snapshot_date_str,
            "timestamp": account_ts_iso,
            "equity": _safe_float(account.get("equity")),
            "cash": _safe_float(account.get("cash")),
            "buying_power": _safe_float(account.get("buying_power")),
            "portfolio_value": _safe_float(account.get("portfolio_value")),
            "symbol": None,
            "asset_class": None,
            "qty": None,
            "side": None,
            "avg_entry_price": None,
            "current_price": None,
            "market_value": None,
            "unrealized_pl": None,
            "unrealized_plpc": None,
            "order_type": None,
            "time_in_force": None,
            "submitted_at": None,
            "filled_at": None,
            "filled_avg_price": None,
            "status": None,
            "order_id": None,
            "mode_or_strategy": mode_or_strategy,
            "strategy_name": account_strategy_name,
            "underlying_symbol": None,
            "notes": None,
            "realized_pl": None,
            "realized_plpc": None,
        }
    )

    for pos in snapshot.get("positions") or []:
        rows.append(
            {
                "row_type": "position",
                "date": snapshot_date_str,
                "timestamp": account_ts_iso,
                "equity": None,
                "cash": None,
                "buying_power": None,
                "portfolio_value": None,
                "symbol": pos.get("symbol"),
                "asset_class": pos.get("asset_class"),
                "qty": _safe_float(pos.get("qty")),
                "side": None,
                "avg_entry_price": _safe_float(pos.get("avg_entry_price")),
                "current_price": _safe_float(pos.get("current_price")),
                "market_value": _safe_float(pos.get("market_value")),
                "unrealized_pl": _safe_float(pos.get("unrealized_pl")),
                "unrealized_plpc": _safe_float(pos.get("unrealized_plpc")),
                "order_type": None,
                "time_in_force": None,
                "submitted_at": None,
                "filled_at": None,
                "filled_avg_price": None,
                "status": None,
                "order_id": None,
                "mode_or_strategy": mode_or_strategy,
                "strategy_name": pos.get("strategy_name") or account_strategy_name,
                "underlying_symbol": pos.get("underlying_symbol") if str(pos.get("asset_class", "")).lower() == "option" else None,
                "notes": None,
                "realized_pl": None,
                "realized_plpc": None,
            }
        )

    trades_source = []
    fills_list = snapshot.get("fills")
    orders_list = snapshot.get("orders")
    if fills_list:
        trades_source.extend(fills_list)
    if orders_list and orders_list is not fills_list:
        seen_ids = {
            t.get("id") or t.get("order_id") or t.get("activity_id")
            for t in trades_source
            if isinstance(t, dict)
        }
        for order in orders_list:
            identifier = order.get("id") or order.get("order_id") or order.get("activity_id")
            if identifier and identifier in seen_ids:
                continue
            trades_source.append(order)

    for order in trades_source:
        if not isinstance(order, dict):
            continue
        filled_iso, filled_dt = _normalize_timestamp(
            order.get("_filled_at_local")
            or order.get("filled_at")
            or order.get("transaction_time")
            or order.get("processed_at"),
            tzinfo,
        )
        submitted_iso, submitted_dt = _normalize_timestamp(
            order.get("_submitted_at_local") or order.get("submitted_at"), tzinfo
        )
        event_dt = filled_dt or submitted_dt or as_of_dt
        event_iso = filled_iso or submitted_iso or as_of_ts_iso
        row_strategy_name = order.get("strategy_name") or account_strategy_name
        qty_val = order.get("filled_qty") or order.get("qty") or order.get("quantity") or order.get("cum_qty")
        realized_pl = order.get("realized_pl")
        realized_plpc = order.get("realized_plpc")
        if realized_pl is None:
            realized_pl = order.get("profit_loss")
        if realized_plpc is None:
            realized_plpc = order.get("profit_loss_pct")
        row = {
            "row_type": "trade",
            "date": _date_string(event_dt or snapshot_date, tzinfo),
            "timestamp": event_iso,
            "equity": None,
            "cash": None,
            "buying_power": None,
            "portfolio_value": None,
            "symbol": order.get("symbol"),
            "asset_class": order.get("asset_class") or order.get("class"),
            "qty": _safe_float(qty_val),
            "side": order.get("side"),
            "avg_entry_price": None,
            "current_price": None,
            "market_value": None,
            "unrealized_pl": None,
            "unrealized_plpc": None,
            "order_type": order.get("type"),
            "time_in_force": order.get("time_in_force"),
            "submitted_at": submitted_iso,
            "filled_at": filled_iso,
            "filled_avg_price": _safe_float(order.get("filled_avg_price") or order.get("price")),
            "status": order.get("status") or order.get("type"),
            "order_id": order.get("id") or order.get("order_id") or order.get("activity_id"),
            "mode_or_strategy": order.get("mode_or_strategy") or mode_or_strategy,
            "strategy_name": row_strategy_name,
            "underlying_symbol": order.get("underlying_symbol"),
            "notes": order.get("order_class"),
            "realized_pl": _safe_float(realized_pl),
            "realized_plpc": _safe_float(realized_plpc),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=CSV_COLUMNS)
    return df
