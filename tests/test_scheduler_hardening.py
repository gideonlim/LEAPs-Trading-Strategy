"""Regression tests for production hardening fixes.

Each test pins down a specific bug class that was found in code review:
- Roll buying before sell fill (would double exposure)
- Allocation recorded on submission (would skip future quarters if order expired)
- Dry-run mutating real state
- last_trade_date set on failed/dry orders (would suppress retries)
- Partial fill desync
"""
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

from leaps_bot.config import (
    AllocationConfig,
    AppConfig,
    PricingConfig,
    SafetyConfig,
    StrategyConfig,
)
from leaps_bot.models import (
    AllocationRecord,
    OptionDetails,
    OrderResult,
    PendingOrderRecord,
    PositionRecord,
)
from leaps_bot.pricing import RateFetcher
from leaps_bot.scheduler import DailyScheduler
from leaps_bot.state import BotState
from tests.conftest import FakeAccount, FakeClock


def _make(dry_run=False, positions_in_account=None, pending_orders=None, last_trade_date=None):
    config = AppConfig(
        dry_run=dry_run,
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
    client.get_option_positions.return_value = positions_in_account or []
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
    if last_trade_date:
        state.last_trade_date = last_trade_date

    rate_fetcher = MagicMock(spec=RateFetcher)
    rate_fetcher.risk_free_rate = 0.045
    rate_fetcher.dividend_yield = 0.013

    scheduler = DailyScheduler(config, client, state, rate_fetcher)
    return scheduler, state, client


# -- Dry-run must not mutate state --

def test_dry_run_does_not_set_last_trade_date():
    scheduler, state, client = _make(dry_run=True)

    # Force a sell scenario: position with expiry within sell threshold
    today = date.today()
    expiry = today + timedelta(days=100)  # past 1/3 threshold of 365-day option
    sym = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    fake_pos = MagicMock()
    fake_pos.symbol = sym
    fake_pos.qty = "2"
    fake_pos.asset_class = __import__("alpaca.trading.enums", fromlist=["AssetClass"]).AssetClass.US_OPTION
    client.get_option_positions.return_value = [fake_pos]

    state.add_position(PositionRecord(
        option_symbol=sym, underlying="SPY", strike=440.0,
        expiry_date=expiry.isoformat(), purchase_date=(today - timedelta(days=265)).isoformat(),
        original_dte=365, qty=2, avg_entry_price=120.0, order_id="o1",
    ))

    # Make get_option_snapshot return a usable snapshot for sell pricing
    client.get_option_snapshot.return_value = None

    summary = scheduler.run()
    assert state.last_trade_date is None, "dry-run must not set last_trade_date"
    assert len(state.pending_orders) == 0, "dry-run must not create pending orders"


def test_dry_run_does_not_record_allocation():
    """Dry-run during a quarterly month should not record an allocation."""
    scheduler, state, client = _make(dry_run=True)
    # June 3, 2026 is in Q2 month + within 7-day window — would trigger allocation
    # Set up a candidate so the allocator can produce dry-run buys

    from leaps_bot.models import ContractCandidate

    candidate = ContractCandidate(
        symbol="SPY270618C00440000", underlying="SPY",
        strike=440.0, expiry=date(2027, 6, 18),
        bid=119.0, ask=121.0, mid=120.0,
        delta=0.88, iv=0.20, open_interest=500,
        theoretical_price=120.5,
    )
    scheduler._finder.find_best_leaps_call = MagicMock(return_value=candidate)
    scheduler._finder.calculate_limit_price = MagicMock(return_value=120.5)

    summary = scheduler.run()
    assert len(state.allocations) == 0, "dry-run must not record allocations"
    assert len(state.pending_orders) == 0, "dry-run must not create pending orders"


# -- Roll only on confirmed sell fill --

def test_sell_does_not_immediately_buy_replacement():
    """The original bug: scheduler bought new LEAPs immediately when sell was submitted,
    causing double exposure if sell never filled."""
    scheduler, state, client = _make(dry_run=False)

    today = date.today()
    expiry = today + timedelta(days=100)
    sym = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    fake_pos = MagicMock()
    fake_pos.symbol = sym
    fake_pos.qty = "2"
    fake_pos.asset_class = __import__("alpaca.trading.enums", fromlist=["AssetClass"]).AssetClass.US_OPTION
    client.get_option_positions.return_value = [fake_pos]

    state.add_position(PositionRecord(
        option_symbol=sym, underlying="SPY", strike=440.0,
        expiry_date=expiry.isoformat(), purchase_date=(today - timedelta(days=265)).isoformat(),
        original_dte=365, qty=2, avg_entry_price=120.0, order_id="o1",
    ))

    # Mock the sell snapshot helper to skip data fetching
    client.get_option_snapshot.return_value = None

    # The order submission returns a real (non-dry-run) order
    sell_order = MagicMock()
    sell_order.id = "sell-order-456"
    client.submit_market_order.return_value = sell_order
    client.submit_limit_order.return_value = sell_order

    # Spy on the roll function — it should NOT be called this run
    scheduler._submit_roll_buy = MagicMock()

    summary = scheduler.run()

    scheduler._submit_roll_buy.assert_not_called()
    assert any(p.action == "sell" and p.intent == "roll" for p in state.pending_orders)


def test_roll_buy_triggered_on_sell_fill():
    """When a sell with intent=roll is reconciled as filled, the buy should be triggered."""
    today = date.today()
    expiry = today + timedelta(days=100)
    sym = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    pending = PendingOrderRecord(
        order_id="sell-456",
        action="sell",
        intent="roll",
        option_symbol=sym,
        qty=2,
        submitted_at=datetime.now().isoformat(),
        underlying="SPY",
    )

    scheduler, state, client = _make(dry_run=False, pending_orders=[pending])
    state.add_position(PositionRecord(
        option_symbol=sym, underlying="SPY", strike=440.0,
        expiry_date=expiry.isoformat(), purchase_date=(today - timedelta(days=265)).isoformat(),
        original_dte=365, qty=2, avg_entry_price=120.0, order_id="o1",
    ))

    # Order is now filled
    filled_order = MagicMock()
    filled_order.status = "filled"
    filled_order.filled_qty = "2"
    filled_order.filled_avg_price = "125.50"
    client.get_order.return_value = filled_order
    scheduler._executor.check_order_status = MagicMock(return_value="filled")

    # The roll buy should be submitted
    scheduler._submit_roll_buy = MagicMock()

    summary = scheduler.run()

    scheduler._submit_roll_buy.assert_called_once_with("SPY", 2, summary)
    assert state.get_position(sym) is None, "Position should be removed on sell fill"
    assert pending not in state.pending_orders, "Pending order should be cleared on fill"


# -- Allocation recorded only on fill --

def test_allocation_recorded_on_buy_fill_not_submission():
    """The original bug: allocator recorded the allocation when execute_buy returned
    success, even if the order later expired. This blocked future quarter allocations."""
    today = date.today()
    pending = PendingOrderRecord(
        order_id="alloc-789",
        action="buy",
        intent="allocate",
        option_symbol="SPY270618C00440000",
        qty=3,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )

    scheduler, state, client = _make(dry_run=False, pending_orders=[pending])

    filled_order = MagicMock()
    filled_order.status = "filled"
    filled_order.filled_avg_price = "120.50"
    client.get_order.return_value = filled_order
    scheduler._executor.check_order_status = MagicMock(return_value="filled")

    summary = scheduler.run()

    # Allocation should be recorded
    assert len(state.allocations) == 1
    assert state.allocations[0].quarter == "2026-Q2"
    # Position should be recorded
    assert state.get_position("SPY270618C00440000") is not None


def test_allocation_not_recorded_on_zero_fill_expired_order():
    """If an allocation buy order expires with ZERO fills, the quarter must NOT be marked allocated.
    (If it had partial fills, allocation IS recorded — see test_partial_fill_then_expired_records_partial_position
    in round 2 hardening tests.)"""
    pending = PendingOrderRecord(
        order_id="alloc-789",
        action="buy",
        intent="allocate",
        option_symbol="SPY270618C00440000",
        qty=3,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )

    scheduler, state, client = _make(dry_run=False, pending_orders=[pending])

    expired_zero_fill = MagicMock()
    expired_zero_fill.status = "expired"
    expired_zero_fill.filled_qty = "0"
    expired_zero_fill.filled_avg_price = "0"
    client.get_order.return_value = expired_zero_fill
    scheduler._executor.check_order_status = MagicMock(return_value="expired")

    summary = scheduler.run()
    assert len(state.allocations) == 0, "Allocation must not be recorded when order expires with zero fills"
    assert pending not in state.pending_orders, "Pending order should be cleared"


def test_should_allocate_blocked_by_pending_allocation():
    """If there's already a pending allocation for this quarter, don't submit another."""
    pending = PendingOrderRecord(
        order_id="alloc-789",
        action="buy",
        intent="allocate",
        option_symbol="SPY270618C00440000",
        qty=3,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )
    scheduler, state, _ = _make(dry_run=False, pending_orders=[pending])

    today_iso = "2026-06-03"
    assert state.has_allocated_this_quarter(today_iso), \
        "Pending allocation order should block re-allocation for that quarter"


# -- last_trade_date guard --

def test_last_trade_date_unchanged_when_order_fails():
    """If the only attempted action's order submission fails, last_trade_date must NOT be set."""
    today = date.today()
    expiry = today + timedelta(days=100)
    sym = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    fake_pos = MagicMock()
    fake_pos.symbol = sym
    fake_pos.qty = "2"
    fake_pos.asset_class = __import__("alpaca.trading.enums", fromlist=["AssetClass"]).AssetClass.US_OPTION

    scheduler, state, client = _make(dry_run=False)
    client.get_option_positions.return_value = [fake_pos]
    state.add_position(PositionRecord(
        option_symbol=sym, underlying="SPY", strike=440.0,
        expiry_date=expiry.isoformat(), purchase_date=(today - timedelta(days=265)).isoformat(),
        original_dte=365, qty=2, avg_entry_price=120.0, order_id="o1",
    ))
    client.get_option_snapshot.return_value = None
    # Sell order submission fails
    client.submit_limit_order.side_effect = Exception("API error")
    client.submit_market_order.side_effect = Exception("API error")

    summary = scheduler.run()
    assert state.last_trade_date is None, "Failed orders must not mark today as traded"


# -- Partial fill handling --

def test_partial_fill_keeps_pending_order_open():
    """A partially filled order should remain pending (not be cleared) so we can
    reconcile the rest in subsequent runs.

    Note: the partial-fill recording behavior (position + allocation are recorded
    for the filled portion) is covered in test_partial_fill_then_expired_records_partial_position
    in test_round2_hardening.py."""
    pending = PendingOrderRecord(
        order_id="alloc-789",
        action="buy",
        intent="allocate",
        option_symbol="SPY270618C00440000",
        qty=3,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make(dry_run=False, pending_orders=[pending])

    partial_order = MagicMock()
    partial_order.status = "partially_filled"
    partial_order.filled_qty = "1"
    partial_order.filled_avg_price = "120.0"
    client.get_order.return_value = partial_order
    scheduler._executor.check_order_status = MagicMock(return_value="partially_filled")

    summary = scheduler.run()

    assert pending in state.pending_orders, "Pending record must remain on partial fill"
    # The filled portion IS conservatively recorded (per round 2 hardening)
    assert pending.recorded_qty == 1
    assert state.get_position("SPY270618C00440000") is not None
    assert state.get_position("SPY270618C00440000").qty == 1
