"""Tests for the read-only monitor mode.

Critical safety invariant: `scheduler.monitor()` must NEVER place orders,
regardless of state. The mid-day and EOD workflows depend on this — they
run after market hours / mid-session and the only thing they're allowed
to do is reconcile pending orders and capture snapshots.

These tests deliberately set up scenarios that would normally trigger
orders in `run()` (positions past their sell threshold, quarterly
allocation month, etc.) and verify monitor() doesn't fire any.
"""
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from alpaca.trading.enums import OrderStatus

from leaps_bot.config import (
    AllocationConfig,
    AppConfig,
    PricingConfig,
    SafetyConfig,
    StrategyConfig,
)
from leaps_bot.models import (
    OptionDetails,
    PendingOrderRecord,
    PositionRecord,
    now_utc_iso,
)
from leaps_bot.pricing import RateFetcher
from leaps_bot.scheduler import DailyScheduler
from leaps_bot.state import BotState
from tests.conftest import FakeAccount, FakeClock


def _make_scheduler(
    pending_orders=None,
    positions_in_account=None,
    market_open=True,
    minutes_since_open=90.0,
    dry_run=False,
):
    config = AppConfig(
        dry_run=dry_run,
        strategy=StrategyConfig(),
        pricing=PricingConfig(),
        allocation=AllocationConfig(quarterly_months=[3, 6, 9, 12], allocation_window_days=7),
        safety=SafetyConfig(no_trade_minutes_after_open=60),
    )
    client = MagicMock()
    client.is_market_open.return_value = market_open
    client.minutes_since_open.return_value = minutes_since_open

    fake_account = FakeAccount()
    fake_account.portfolio_value = "55000.00"
    client.get_account.return_value = fake_account
    client.get_cash_available.return_value = 50000.0
    client.get_buying_power.return_value = 50000.0
    client.get_options_buying_power.return_value = 50000.0
    client.get_option_positions.return_value = positions_in_account or []
    client.get_open_orders.return_value = []
    client.get_underlying_price.return_value = 550.0

    now = datetime(2026, 6, 3, 11, 0, 0)
    client.get_clock.return_value = FakeClock(
        is_open=market_open, timestamp=now,
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

    return DailyScheduler(config, client, state, rate_fetcher), state, client


# ----------------------------------------------------------------------
# Safety invariant: monitor must NEVER submit orders
# ----------------------------------------------------------------------

def test_monitor_does_not_submit_orders_when_position_past_sell_threshold():
    """A position that should normally trigger a sell-and-roll in run() must
    NOT trigger any orders in monitor() — the morning workflow's job, not the
    mid-day or EOD job."""
    today = date.today()
    expiry = today + timedelta(days=100)  # past 1/3 threshold of 365 DTE
    sym = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    fake_pos = MagicMock()
    fake_pos.symbol = sym
    fake_pos.qty = "2"
    fake_pos.asset_class = __import__("alpaca.trading.enums", fromlist=["AssetClass"]).AssetClass.US_OPTION

    scheduler, state, client = _make_scheduler(positions_in_account=[fake_pos])
    state.add_position(PositionRecord(
        option_symbol=sym, underlying="SPY", strike=440.0,
        expiry_date=expiry.isoformat(), purchase_date=(today - timedelta(days=265)).isoformat(),
        original_dte=365, qty=2, avg_entry_price=120.0, order_id="o1",
    ))

    scheduler.monitor()

    # No orders submitted
    client.submit_market_order.assert_not_called()
    client.submit_limit_order.assert_not_called()
    # No new pending orders created
    assert all(p.order_id == "o1-pending" for p in state.pending_orders) or state.pending_orders == []


def test_monitor_does_not_submit_orders_in_quarterly_window():
    """In a quarterly-allocation month, run() would deploy capital. monitor()
    must not, even if cash is sitting waiting."""
    from unittest.mock import patch
    scheduler, state, client = _make_scheduler()

    with patch("leaps_bot.allocator.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 3)  # Q2 quarterly month, day 3
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        scheduler.monitor()

    client.submit_market_order.assert_not_called()
    client.submit_limit_order.assert_not_called()
    assert len(state.allocations) == 0


def test_monitor_does_not_set_last_trade_date():
    """Even if monitor reconciles a fill, last_trade_date should NOT be set
    to today by monitor — that's the morning workflow's marker."""
    pending = PendingOrderRecord(
        order_id="o-fill", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=2,
        submitted_at=now_utc_iso(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED
    fake_order.filled_qty = "2"
    fake_order.filled_avg_price = "120.50"
    fake_order.filled_at = datetime(2026, 6, 2, 15, 35, 0, tzinfo=timezone.utc)
    fake_order.updated_at = fake_order.filled_at
    fake_order.submitted_at = fake_order.filled_at
    client.get_order.return_value = fake_order

    scheduler.monitor()

    assert state.last_trade_date is None, (
        "monitor must not set last_trade_date — that's the morning workflow's "
        "responsibility (so a missed morning run can be retried tomorrow)."
    )


# ----------------------------------------------------------------------
# Reconciliation: monitor must update state from broker fills
# ----------------------------------------------------------------------

def test_monitor_reconciles_filled_pending_buy():
    """A buy order that filled after the morning run must be picked up in
    the mid-day or EOD monitor pass."""
    pending = PendingOrderRecord(
        order_id="o-fill", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=2,
        submitted_at=now_utc_iso(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED
    fake_order.filled_qty = "2"
    fake_order.filled_avg_price = "120.50"
    fake_order.filled_at = datetime(2026, 6, 2, 15, 35, 0, tzinfo=timezone.utc)
    fake_order.updated_at = fake_order.filled_at
    fake_order.submitted_at = fake_order.filled_at
    client.get_order.return_value = fake_order

    scheduler.monitor()

    assert pending not in state.pending_orders, "Filled pending order must be cleared"
    assert state.get_position("SPY270618C00440000") is not None, "Position must be recorded"
    assert len(state.allocations) == 1, "Allocation must be recorded on fill"
    assert len(state.trades) == 1, "Trade record must be appended"


def test_monitor_reconciles_filled_pending_sell_and_clears_position():
    today = date.today()
    expiry = today + timedelta(days=100)
    sym = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    pending = PendingOrderRecord(
        order_id="sell-1", action="sell", intent="roll",
        option_symbol=sym, qty=2,
        submitted_at=now_utc_iso(),
        underlying="SPY",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])
    state.add_position(PositionRecord(
        option_symbol=sym, underlying="SPY", strike=440.0,
        expiry_date=expiry.isoformat(),
        purchase_date=(today - timedelta(days=265)).isoformat(),
        original_dte=365, qty=2, avg_entry_price=120.0, order_id="o0",
    ))

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED
    fake_order.filled_qty = "2"
    fake_order.filled_avg_price = "135.00"
    fake_order.filled_at = datetime(2026, 6, 3, 13, 30, 0, tzinfo=timezone.utc)
    fake_order.updated_at = fake_order.filled_at
    fake_order.submitted_at = fake_order.filled_at
    client.get_order.return_value = fake_order

    # Spy: a sell-with-roll-intent that fills DOES trigger a roll buy in run().
    # In monitor mode, we should NOT submit a roll buy either — even though the
    # sell filled, the next morning's workflow will handle the roll.
    scheduler._submit_roll_buy = MagicMock()

    scheduler.monitor()

    assert state.get_position(sym) is None, "Sold position must be removed"
    scheduler._submit_roll_buy.assert_not_called()


def test_monitor_records_partial_fill_without_clearing_pending():
    pending = PendingOrderRecord(
        order_id="o-partial", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=3,
        submitted_at=now_utc_iso(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    fake_order = MagicMock()
    fake_order.status = OrderStatus.PARTIALLY_FILLED
    fake_order.filled_qty = "1"
    fake_order.filled_avg_price = "120.0"
    fake_order.filled_at = None
    fake_order.updated_at = datetime(2026, 6, 2, 15, 35, 0, tzinfo=timezone.utc)
    fake_order.submitted_at = fake_order.updated_at
    client.get_order.return_value = fake_order

    scheduler.monitor()

    assert pending in state.pending_orders, "Partial fill must keep pending alive"
    assert pending.recorded_qty == 1
    pos = state.get_position("SPY270618C00440000")
    assert pos is not None and pos.qty == 1


# ----------------------------------------------------------------------
# Snapshot behavior
# ----------------------------------------------------------------------

def test_monitor_captures_snapshot_by_default():
    scheduler, state, _ = _make_scheduler()
    scheduler.monitor()
    assert len(state.snapshots) == 1, "Default monitor must capture snapshot"


def test_monitor_skips_snapshot_when_requested():
    """The mid-day workflow uses --no-snapshot to avoid noise in the time-series
    (EOD will capture the canonical daily anchor)."""
    scheduler, state, _ = _make_scheduler()
    scheduler.monitor(capture_snapshot=False)
    assert len(state.snapshots) == 0


def test_monitor_skips_snapshot_in_dry_run():
    scheduler, state, _ = _make_scheduler(dry_run=True)
    scheduler.monitor()
    assert len(state.snapshots) == 0


# ----------------------------------------------------------------------
# Run record + last_run
# ----------------------------------------------------------------------

def test_monitor_records_run_with_monitor_type():
    scheduler, state, _ = _make_scheduler()
    scheduler.monitor()
    assert len(state.runs) == 1
    assert state.runs[0].run_type == "monitor", (
        "Monitor passes must be recorded as run_type='monitor' so run history "
        "can distinguish them from trading runs for auditing."
    )
    # last_run is updated even on monitor passes (it's a heartbeat, not a trade marker)
    assert state.last_run is not None


def test_trade_run_records_trade_type():
    scheduler, state, _ = _make_scheduler()
    scheduler.run()
    assert len(state.runs) == 1
    assert state.runs[0].run_type == "trade"


def test_monitor_works_outside_market_hours():
    """The EOD workflow runs after market close. Monitor must not require
    the market to be open."""
    scheduler, state, client = _make_scheduler(market_open=False)
    summary = scheduler.monitor()
    # Should NOT be a "skipped" run — monitor doesn't have the market-hours guard
    assert not summary.get("skipped", False), (
        "Monitor must run after market close (that's when EOD workflow fires)."
    )
    assert len(state.runs) == 1


def test_monitor_works_inside_first_hour_window():
    """The first-hour-after-open guard is a TRADING guard. Monitor must not
    be blocked by it (e.g., if a manual workflow_dispatch fires at 10 AM)."""
    scheduler, state, _ = _make_scheduler(market_open=True, minutes_since_open=10.0)
    summary = scheduler.monitor()
    assert not summary.get("skipped", False)


# ----------------------------------------------------------------------
# Account safety check
# ----------------------------------------------------------------------

def test_monitor_aborts_if_account_blocked():
    scheduler, state, client = _make_scheduler()
    blocked = FakeAccount()
    blocked.trading_blocked = True
    client.get_account.return_value = blocked

    summary = scheduler.monitor()
    assert summary["errors"]
    # No snapshot, no fills processed
    assert len(state.snapshots) == 0
