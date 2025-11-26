import datetime as dt
import pathlib
import sys
import unittest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from paper_trading import CSV_COLUMNS, build_paper_trades_export


class BuildPaperTradesExportTest(unittest.TestCase):
    def setUp(self):
        self.snapshot_time = dt.datetime(2024, 1, 2, 15, 30, tzinfo=dt.timezone.utc)
        self.snapshot = {
            "date": dt.date(2024, 1, 2),
            "timezone": dt.timezone.utc,
            "as_of": self.snapshot_time,
            "account": {
                "equity": "10000",
                "cash": "5000",
                "buying_power": "15000",
                "portfolio_value": "10000",
                "updated_at": self.snapshot_time,
            },
            "positions": [
                {
                    "symbol": "AAPL",
                    "asset_class": "us_equity",
                    "qty": "10",
                    "avg_entry_price": "100",
                    "current_price": "110",
                    "market_value": "1100",
                    "unrealized_pl": "100",
                    "unrealized_plpc": "0.1",
                }
            ],
            "orders": [
                {
                    "symbol": "AAPL",
                    "asset_class": "us_equity",
                    "qty": "5",
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day",
                    "submitted_at": self.snapshot_time,
                    "filled_at": self.snapshot_time + dt.timedelta(minutes=5),
                    "filled_avg_price": "105",
                    "status": "filled",
                    "id": "order123",
                    "order_class": "simple",
                }
            ],
        }

    def test_columns_and_row_counts(self):
        df = build_paper_trades_export(
            self.snapshot,
            mode_or_strategy="balanced",
            strategy_name="hybrid_balanced",
        )
        self.assertListEqual(list(df.columns), CSV_COLUMNS)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0]["row_type"], "account_summary")
        self.assertEqual(df.iloc[1]["row_type"], "position")
        self.assertEqual(df.iloc[2]["row_type"], "trade")

    def test_populated_fields(self):
        df = build_paper_trades_export(
            self.snapshot,
            mode_or_strategy="balanced",
            strategy_name="hybrid_balanced",
        )
        account = df[df["row_type"] == "account_summary"].iloc[0]
        self.assertEqual(account["equity"], 10000.0)
        self.assertEqual(account["mode_or_strategy"], "balanced")
        position = df[df["row_type"] == "position"].iloc[0]
        self.assertEqual(position["symbol"], "AAPL")
        self.assertEqual(position["mode_or_strategy"], "balanced")
        trade = df[df["row_type"] == "trade"].iloc[0]
        self.assertEqual(trade["status"], "filled")
        self.assertEqual(trade["notes"], "simple")
        # filled timestamp should be used for trade date
        self.assertTrue(str(trade["timestamp"]).startswith("2024-01-02T15:35:00+00:00"))

    def test_trade_row_from_fills(self):
        fill_time = self.snapshot_time.replace(hour=14, minute=15)
        snapshot = dict(self.snapshot)
        snapshot["fills"] = [
            {
                "symbol": "MSFT",
                "side": "sell",
                "quantity": "3",
                "price": "310.5",
                "order_id": "fill123",
                "transaction_time": fill_time.isoformat(),
                "profit_loss": "12.5",
                "profit_loss_pct": "0.04",
            }
        ]
        snapshot["orders"] = []

        df = build_paper_trades_export(
            snapshot,
            mode_or_strategy="hybrid",
            strategy_name="sell_test",
        )
        trade = df[df["row_type"] == "trade"].iloc[0]
        self.assertEqual(trade["symbol"], "MSFT")
        self.assertEqual(trade["qty"], 3.0)
        self.assertEqual(trade["side"], "sell")
        self.assertEqual(trade["filled_avg_price"], 310.5)
        self.assertEqual(trade["order_id"], "fill123")
        self.assertEqual(trade["realized_pl"], 12.5)
        self.assertEqual(trade["realized_plpc"], 0.04)
        self.assertTrue(str(trade["timestamp"]).startswith("2024-01-02T14:15:00+00:00"))


if __name__ == "__main__":
    unittest.main()
