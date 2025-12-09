import types

import pytest

import app


class StubBroker:
    def __init__(self, equity: float, buying_power: float):
        self._equity = equity
        self._buying_power = buying_power
        self.submitted = []

    def get_account(self):
        return {"equity": self._equity, "buying_power": self._buying_power, "cash": self._buying_power}

    def submit_order(self, payload):
        self.submitted.append(payload)
        return {"status": "accepted", "payload": payload}


def test_sell_respects_notional_cap_without_force(monkeypatch):
    broker = StubBroker(equity=100000, buying_power=100000)
    monkeypatch.setattr(app, "paper_broker", broker)
    monkeypatch.setattr(app, "PAPER_MAX_POSITION_NOTIONAL", 8000)
    with pytest.raises(ValueError):
        app.place_guarded_paper_order(
            "TEST",
            qty=10,
            side="sell",
            price_hint=1000,  # notional 10,000
            support_brackets=False,
        )


def test_sell_allows_force_liquidation(monkeypatch):
    broker = StubBroker(equity=100000, buying_power=100000)
    monkeypatch.setattr(app, "paper_broker", broker)
    monkeypatch.setattr(app, "PAPER_MAX_POSITION_NOTIONAL", 8000)
    app.place_guarded_paper_order(
        "TEST",
        qty=10,
        side="sell",
        price_hint=1000,  # notional 10,000
        support_brackets=False,
        force_liquidation=True,
    )
    assert broker.submitted, "Order should be submitted when force_liquidation is True"
