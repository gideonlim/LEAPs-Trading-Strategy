"""Regression tests for round 2 hardening fixes.

- OrderStatus enum string handling
- State schema migration for legacy pending orders
- Partial-fill then terminal-status reconciliation (allocation must stick)
"""
import json
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
from leaps_bot.models import OptionDetails, PendingOrderRecord, PositionRecord
from leaps_bot.order_executor import OrderExecutor, _normalize_status
from leaps_bot.pricing import RateFetcher
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
    client.get_account.return_value = FakeAccount()
    client.get_cash_available.return_value = 50000.0
    client.get_buying_power.return_value = 50000.0
    client.get_options_buying_power.return_value = 50000.0
    client.get_option_positions.return_value = []
    client.get_open_orders.return_value = []

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


# -- P0: OrderStatus enum handling --

def test_normalize_orderstatus_enum():
    """str(OrderStatus.FILLED) is 'OrderStatus.FILLED' but .value is 'filled'.
    Must normalize to the value form so reconciliation can compare correctly."""
    assert _normalize_status(OrderStatus.FILLED) == "filled"
    assert _normalize_status(OrderStatus.PARTIALLY_FILLED) == "partially_filled"
    assert _normalize_status(OrderStatus.CANCELED) == "canceled"
    assert _normalize_status(OrderStatus.EXPIRED) == "expired"
    assert _normalize_status(OrderStatus.REJECTED) == "rejected"


def test_normalize_status_strings():
    assert _normalize_status("filled") == "filled"
    assert _normalize_status("FILLED") == "filled"
    assert _normalize_status("OrderStatus.FILLED") == "filled"


def test_normalize_status_handles_none():
    assert _normalize_status(None) == "unknown"


def test_check_order_status_returns_value_for_real_enum():
    client = MagicMock()
    config = AppConfig()
    executor = OrderExecutor(client, config)

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED
    client.get_order.return_value = fake_order

    assert executor.check_order_status("any-id") == "filled"


def test_real_enum_filled_triggers_fill_handling():
    """End-to-end: pending order with real Alpaca enum status must trigger fill logic."""
    pending = PendingOrderRecord(
        order_id="o-1", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=2,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED  # the real enum, not a string
    fake_order.filled_qty = "2"
    fake_order.filled_avg_price = "120.50"
    client.get_order.return_value = fake_order

    summary = scheduler.run()

    # If the enum-vs-string bug came back, the order would stay pending
    assert pending not in state.pending_orders, "Filled order must be cleared from pending"
    assert state.get_position("SPY270618C00440000") is not None, "Position must be recorded"
    assert len(state.allocations) == 1, "Allocation must be recorded"


# -- P1: State schema migration --

def test_load_legacy_pending_order_without_intent(tmp_path):
    """Old state files predate the `intent` field. Loading must not crash."""
    legacy_state = {
        "positions": [],
        "allocations": [],
        "pending_orders": [
            {
                "order_id": "legacy-1",
                "action": "buy",
                "option_symbol": "SPY270618C00440000",
                "qty": 2,
                "submitted_at": "2026-04-15T10:30:00",
            }
        ],
        "last_run": "2026-04-15T10:30:00",
        "last_trade_date": None,
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(legacy_state))

    state = BotState.load(path)
    assert len(state.pending_orders) == 1
    pending = state.pending_orders[0]
    assert pending.order_id == "legacy-1"
    # buy action defaults to intent="open" during migration
    assert pending.intent == "open"
    assert pending.recorded_qty == 0


def test_load_legacy_pending_sell_infers_close_intent(tmp_path):
    legacy_state = {
        "positions": [],
        "allocations": [],
        "pending_orders": [
            {
                "order_id": "legacy-2",
                "action": "sell",
                "option_symbol": "SPY270618C00440000",
                "qty": 2,
                "submitted_at": "2026-04-15T10:30:00",
            }
        ],
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(legacy_state))

    state = BotState.load(path)
    assert state.pending_orders[0].intent == "close"


def test_load_state_drops_unknown_extra_fields(tmp_path):
    """Forward compatibility: an old version of the bot loading newer state shouldn't crash."""
    state_with_extra = {
        "positions": [],
        "allocations": [],
        "pending_orders": [
            {
                "order_id": "o-1",
                "action": "buy",
                "intent": "open",
                "option_symbol": "SPY270618C00440000",
                "qty": 2,
                "submitted_at": "2026-04-15T10:30:00",
                "future_field_we_dont_know_about": "some-value",
            }
        ],
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state_with_extra))

    state = BotState.load(path)
    assert len(state.pending_orders) == 1


# -- P1: Partial fill then terminal status --

def test_partial_fill_then_expired_records_partial_position():
    """Order partially fills (qty=1 of 3), then expires. Must record the 1 filled
    contract as a position and the corresponding allocation. Otherwise the bot
    thinks no allocation happened and would re-allocate next day, deploying more capital."""
    pending = PendingOrderRecord(
        order_id="alloc-1", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=3,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    # First reconciliation: partially_filled with 1 filled
    partial_order = MagicMock()
    partial_order.status = OrderStatus.PARTIALLY_FILLED
    partial_order.filled_qty = "1"
    partial_order.filled_avg_price = "120.50"
    client.get_order.return_value = partial_order

    scheduler.run()

    # After partial: position recorded with qty=1, allocation recorded, pending still active
    pos = state.get_position("SPY270618C00440000")
    assert pos is not None and pos.qty == 1, "Partial fill must record partial position"
    assert len(state.allocations) == 1, "Partial fill of allocation must record allocation"
    assert pending in state.pending_orders, "Pending must remain while order is open"
    assert pending.recorded_qty == 1, "recorded_qty tracks idempotent fill state"

    # Second reconciliation: order now expired (no further fills)
    expired_order = MagicMock()
    expired_order.status = OrderStatus.EXPIRED
    expired_order.filled_qty = "1"
    expired_order.filled_avg_price = "120.50"
    client.get_order.return_value = expired_order

    scheduler.run()

    # After expire: pending cleared, but partial position + allocation persist
    assert pending not in state.pending_orders
    assert state.get_position("SPY270618C00440000") is not None
    assert state.get_position("SPY270618C00440000").qty == 1
    assert len(state.allocations) == 1
    # Quarter should still be marked allocated, blocking re-allocation
    assert state.has_allocated_this_quarter("2026-06-03")


def test_partial_fill_then_filled_does_not_double_count():
    """Order partially fills (1 of 3) then fully fills (3 of 3). Must end with
    position qty=3, not qty=4 (no double-counting)."""
    pending = PendingOrderRecord(
        order_id="alloc-1", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=3,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    partial_order = MagicMock()
    partial_order.status = OrderStatus.PARTIALLY_FILLED
    partial_order.filled_qty = "1"
    partial_order.filled_avg_price = "120.00"
    client.get_order.return_value = partial_order
    scheduler.run()
    assert state.get_position("SPY270618C00440000").qty == 1

    filled_order = MagicMock()
    filled_order.status = OrderStatus.FILLED
    filled_order.filled_qty = "3"
    filled_order.filled_avg_price = "121.00"
    client.get_order.return_value = filled_order
    scheduler.run()

    pos = state.get_position("SPY270618C00440000")
    assert pos.qty == 3, "Must end with total filled qty, not sum of increments"


def test_zero_fill_canceled_does_not_record_anything():
    """Order is canceled with zero fills: no position, no allocation."""
    pending = PendingOrderRecord(
        order_id="alloc-1", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=3,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    canceled_order = MagicMock()
    canceled_order.status = OrderStatus.CANCELED
    canceled_order.filled_qty = "0"
    canceled_order.filled_avg_price = "0"
    client.get_order.return_value = canceled_order

    scheduler.run()

    assert state.get_position("SPY270618C00440000") is None
    assert len(state.allocations) == 0
    assert pending not in state.pending_orders


def test_partial_sell_then_expired_reduces_position():
    """Sell partially fills (1 of 2), then expires. Position should be reduced by 1, not removed."""
    today = date.today()
    expiry = today + timedelta(days=100)
    sym = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    pending = PendingOrderRecord(
        order_id="sell-1", action="sell", intent="roll",
        option_symbol=sym, qty=2,
        submitted_at=datetime.now().isoformat(),
        underlying="SPY",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])
    state.add_position(PositionRecord(
        option_symbol=sym, underlying="SPY", strike=440.0,
        expiry_date=expiry.isoformat(), purchase_date=today.isoformat(),
        original_dte=365, qty=2, avg_entry_price=120.0, order_id="o0",
    ))

    # Spy on roll buy
    scheduler._submit_roll_buy = MagicMock()

    expired_with_partial = MagicMock()
    expired_with_partial.status = OrderStatus.EXPIRED
    expired_with_partial.filled_qty = "1"
    expired_with_partial.filled_avg_price = "125.0"
    client.get_order.return_value = expired_with_partial

    scheduler.run()

    pos = state.get_position(sym)
    assert pos is not None and pos.qty == 1, "Partially sold position should retain remainder"
    # Roll should still trigger since some fill happened, but only for the filled qty
    scheduler._submit_roll_buy.assert_called_once_with("SPY", 1, scheduler._submit_roll_buy.call_args[0][2])
