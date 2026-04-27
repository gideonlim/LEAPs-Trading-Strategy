"""Regression tests for reporting accuracy fixes.

Each test pins down a specific accuracy bug that would corrupt tax filings
or report numbers if regressed:
- Trade timestamp must be the actual broker fill time, not reconciliation time
- Holding period must be measured from purchase to fill, not to today
- Total return must not double-count realized P&L
- Tax CSV must not fabricate cost basis to 0 when entry price is unknown
"""
import csv
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

from alpaca.trading.enums import OrderStatus

from leaps_bot.config import (
    AllocationConfig,
    AppConfig,
    PricingConfig,
    SafetyConfig,
    StrategyConfig,
)
from leaps_bot.models import (
    AllocationRecord,
    DailySnapshot,
    PendingOrderRecord,
    PositionRecord,
    PositionSnapshot,
    TradeRecord,
)
from leaps_bot.pricing import RateFetcher
from leaps_bot.reporting import ReportGenerator
from leaps_bot.scheduler import DailyScheduler
from leaps_bot.state import BotState
from tests.conftest import FakeAccount, FakeClock


def _make_scheduler(pending_orders=None):
    config = AppConfig(
        dry_run=False,
        strategy=StrategyConfig(),
        pricing=PricingConfig(),
        allocation=AllocationConfig(quarterly_months=[3, 6, 9, 12], allocation_window_days=7),
        safety=SafetyConfig(no_trade_minutes_after_open=60),
    )
    client = MagicMock()
    client.is_market_open.return_value = True
    client.minutes_since_open.return_value = 90.0

    fake_account = FakeAccount()
    fake_account.portfolio_value = "55000.00"
    client.get_account.return_value = fake_account
    client.get_cash_available.return_value = 50000.0
    client.get_buying_power.return_value = 50000.0
    client.get_options_buying_power.return_value = 50000.0
    client.get_option_positions.return_value = []
    client.get_open_orders.return_value = []
    client.get_underlying_price.return_value = 550.0

    now = datetime(2026, 6, 3, 11, 0, 0)
    client.get_clock.return_value = FakeClock(
        is_open=True, timestamp=now,
        next_open=now.replace(hour=9, minute=30),
        next_close=now.replace(hour=16, minute=0),
    )

    state = BotState()
    if pending_orders:
        for p in pending_orders:
            state.add_pending_order(p)

    rate_fetcher = MagicMock(spec=RateFetcher)
    rate_fetcher.risk_free_rate = 0.045
    rate_fetcher.dividend_yield = 0.013

    scheduler = DailyScheduler(config, client, state, rate_fetcher)
    return scheduler, state, client


# ----------------------------------------------------------------------
# P1.1: Trade timestamp comes from broker, not reconciliation time
# ----------------------------------------------------------------------

def test_extract_fill_timestamp_prefers_filled_at():
    order = MagicMock()
    order.filled_at = datetime(2026, 12, 31, 15, 45, 0)
    order.updated_at = datetime(2027, 1, 2, 10, 0, 0)
    order.submitted_at = datetime(2026, 12, 30, 9, 0, 0)
    ts = DailyScheduler._extract_fill_timestamp(order)
    assert ts.startswith("2026-12-31T15:45")


def test_extract_fill_timestamp_falls_back_to_updated_at():
    """For partial fills, filled_at is None — updated_at reflects last fill."""
    order = MagicMock()
    order.filled_at = None
    order.updated_at = datetime(2026, 11, 15, 14, 30, 0)
    order.submitted_at = datetime(2026, 11, 15, 14, 0, 0)
    ts = DailyScheduler._extract_fill_timestamp(order)
    assert ts.startswith("2026-11-15T14:30")


def test_extract_fill_timestamp_falls_back_to_submitted_at():
    order = MagicMock(spec=["submitted_at"])
    order.submitted_at = datetime(2026, 6, 3, 10, 0, 0)
    ts = DailyScheduler._extract_fill_timestamp(order)
    assert ts.startswith("2026-06-03T10:00")


def test_extract_fill_timestamp_handles_none_order():
    ts = DailyScheduler._extract_fill_timestamp(None)
    # Falls back to now(); just verify we got *some* ISO timestamp
    assert "T" in ts


def test_trade_record_uses_broker_fill_time_not_reconciliation_time():
    """If a sell fills Dec 31 but the bot reconciles Jan 2, the trade record's
    timestamp must reflect Dec 31 — otherwise the tax year filter is wrong."""
    pending = PendingOrderRecord(
        order_id="o-eoy", action="sell", intent="close",
        option_symbol="SPY270318C00440000", qty=2,
        submitted_at="2026-12-30T15:30:00",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])
    # Position is in state so realized P&L can be computed
    state.add_position(PositionRecord(
        option_symbol="SPY270318C00440000", underlying="SPY", strike=440.0,
        expiry_date="2027-03-18", purchase_date="2026-03-01",
        original_dte=383, qty=2, avg_entry_price=100.0, order_id="o-buy",
    ))

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED
    fake_order.filled_qty = "2"
    fake_order.filled_avg_price = "120.00"
    # Order actually filled Dec 31
    fake_order.filled_at = datetime(2026, 12, 31, 15, 45, 0)
    fake_order.updated_at = datetime(2026, 12, 31, 15, 45, 0)
    fake_order.submitted_at = datetime(2026, 12, 30, 15, 30, 0)
    client.get_order.return_value = fake_order
    scheduler._submit_roll_buy = MagicMock()  # don't try to find replacement

    # Bot reconciles Jan 2 (after the holiday)
    scheduler.run()

    sells = [t for t in state.trades if t.action == "sell"]
    assert len(sells) == 1
    # Timestamp must be the broker fill time, not now()
    assert sells[0].timestamp.startswith("2026-12-31"), (
        f"Trade timestamp is {sells[0].timestamp}, expected 2026-12-31. "
        "Reconciliation-time timestamps would mis-classify tax year."
    )


def test_buy_trade_timestamp_uses_broker_fill_time():
    pending = PendingOrderRecord(
        order_id="o-buy", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=2,
        submitted_at="2026-06-02T15:30:00",
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED
    fake_order.filled_qty = "2"
    fake_order.filled_avg_price = "120.00"
    fake_order.filled_at = datetime(2026, 6, 2, 15, 35, 0)
    fake_order.updated_at = datetime(2026, 6, 2, 15, 35, 0)
    fake_order.submitted_at = datetime(2026, 6, 2, 15, 30, 0)
    client.get_order.return_value = fake_order

    scheduler.run()

    assert len(state.trades) == 1
    assert state.trades[0].timestamp.startswith("2026-06-02T15:35")
    # Position should also have purchase_date set from fill, not reconciliation
    pos = state.get_position("SPY270618C00440000")
    assert pos.purchase_date == "2026-06-02"


# ----------------------------------------------------------------------
# P1.2: Holding period from purchase → fill date, not purchase → today
# ----------------------------------------------------------------------

def test_holding_period_measured_from_purchase_to_fill_not_today():
    """A sale held for exactly 365 days at fill time must be classified short.
    If we used today (later) it could flip to long-term."""
    pending = PendingOrderRecord(
        order_id="o-1", action="sell", intent="close",
        option_symbol="SPY270318C00440000", qty=1,
        submitted_at="2026-03-01T10:00:00",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])
    state.add_position(PositionRecord(
        option_symbol="SPY270318C00440000", underlying="SPY", strike=440.0,
        expiry_date="2027-03-18", purchase_date="2025-03-01",
        original_dte=748, qty=1, avg_entry_price=100.0, order_id="o-buy",
    ))

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED
    fake_order.filled_qty = "1"
    fake_order.filled_avg_price = "120.00"
    # Filled exactly 365 days after purchase (2025-03-01 + 365 = 2026-03-01)
    fake_order.filled_at = datetime(2026, 3, 1, 14, 30, 0)
    fake_order.updated_at = datetime(2026, 3, 1, 14, 30, 0)
    fake_order.submitted_at = datetime(2026, 3, 1, 10, 0, 0)
    client.get_order.return_value = fake_order
    scheduler._submit_roll_buy = MagicMock()

    # Bot might run weeks later but holding period must reflect the actual fill date
    scheduler.run()

    sells = [t for t in state.trades if t.action == "sell"]
    assert len(sells) == 1
    # 365 days exactly from fill → short-term (need > 365 for long)
    assert sells[0].holding_days == 365


# ----------------------------------------------------------------------
# P1.3: Total return must not double-count realized P&L
# ----------------------------------------------------------------------

def test_total_return_does_not_double_count_realized_pnl():
    """Realized P&L is already in cash (and therefore portfolio_value).
    total_return_dollar must equal portfolio_value - total_invested, not
    portfolio_value + realized_pnl - total_invested.

    Self-consistent scenario: deposit $20k, buy 2 @ $100 = $20k (cash=0),
    sell 2 @ $125 = $25k (cash=$25k). Realized P&L = $5k. Total return = $5k.
    """
    state = BotState()
    state.record_allocation(AllocationRecord(
        quarter="2026-Q1", allocated_date="2026-03-01",
        amount=20000.0, contracts_bought=["SPY270318C00440000"],
    ))
    # Inception snapshot: $20k cash, no positions yet
    state.add_snapshot(DailySnapshot(
        date="2026-02-28", timestamp="2026-02-28T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0,
        num_positions=0, underlying_prices={"SPY": 540.0},
    ))
    # The buy
    state.add_trade(TradeRecord(
        timestamp="2026-03-01T15:33:00", order_id="o-buy",
        action="buy", intent="allocate",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=100.0, total_value=20000.0,
    ))
    # The closing sell with $5000 realized profit
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:30:00", order_id="o-sell",
        action="sell", intent="roll",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=125.0, total_value=25000.0,
        avg_entry_price=100.0, realized_pnl=5000.0, holding_days=259,
    ))
    # Post-sale snapshot: cash now $25k, no positions. Internally consistent
    # with the trades above (no external flow detected).
    state.add_snapshot(DailySnapshot(
        date="2026-11-16", timestamp="2026-11-16T15:33:00",
        cash=25000.0, options_buying_power=25000.0,
        portfolio_value=25000.0, positions_market_value=0.0,
        num_positions=0, underlying_prices={"SPY": 560.0},
    ))

    summary = ReportGenerator(state).compute_summary()

    # Correct: portfolio_value (25000, which already reflects the 5000 gain)
    # minus initial capital (20000) = 5000.
    # Bug would have been: 25000 + 5000 - 20000 = 10000 (double-counted)
    assert summary.total_invested == 20000.0
    assert summary.num_external_flows == 0
    assert summary.total_return_dollar == 5000.0
    assert summary.total_return_pct == 25.0


# ----------------------------------------------------------------------
# P1.4: Tax CSV must not fabricate cost basis to 0
# ----------------------------------------------------------------------

def test_tax_csv_leaves_cost_basis_blank_when_unknown(tmp_path, caplog):
    """A sell trade without avg_entry_price must produce a CSV row with
    empty cost_basis and gain_loss fields, not zeros that would inflate the
    apparent gain. A warning should be logged so the user knows to fix it."""
    import logging
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:30:00", order_id="orphan-1",
        action="sell", intent="close",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=120.0, total_value=24000.0,
        # Critical: avg_entry_price is None (e.g., desync, manual sell)
        avg_entry_price=None, realized_pnl=None, holding_days=None,
    ))

    out = tmp_path / "tax.csv"
    with caplog.at_level(logging.WARNING):
        ReportGenerator(state).export_tax_csv(out)

    with open(out) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["cost_basis"] == "", "cost_basis must be blank when entry price unknown, NOT 0"
    assert row["gain_loss"] == "", "gain_loss must be blank when cost basis unknown"
    assert row["term"] == "", "term must be blank when holding period unknown"
    # Proceeds is still computed (we know it from fill price)
    assert row["proceeds"] == "24000.00"
    # Warning was emitted
    assert any("missing cost basis" in r.message.lower() for r in caplog.records)


def test_tax_csv_normal_row_unaffected_by_missing_basis_handling(tmp_path):
    """Sells with proper entry prices still produce complete tax rows."""
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:30:00", order_id="good-1",
        action="sell", intent="roll",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=125.0, total_value=25000.0,
        avg_entry_price=100.0, realized_pnl=5000.0, holding_days=259,
    ))

    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["cost_basis"] == "20000.00"
    assert rows[0]["gain_loss"] == "5000.00"
    assert rows[0]["term"] == "short"


def test_tax_csv_term_blank_when_holding_days_missing(tmp_path):
    """Without holding_days we can't classify short vs long — must leave it blank,
    not default to 'short'."""
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:30:00", order_id="no-hold-1",
        action="sell", intent="close",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=1, fill_price=120.0, total_value=12000.0,
        avg_entry_price=100.0, realized_pnl=2000.0, holding_days=None,
    ))

    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["term"] == "", "term must be blank when holding period is unknown"
    # But cost_basis and gain_loss are still computable
    assert rows[0]["cost_basis"] == "10000.00"
    assert rows[0]["gain_loss"] == "2000.00"
