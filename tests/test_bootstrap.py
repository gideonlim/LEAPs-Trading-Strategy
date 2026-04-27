"""Tests for first-run bootstrap allocation.

When the bot runs for the first time (no positions, no trades, no allocations),
it should deploy capital immediately rather than waiting for the next quarterly
window. This avoids weeks/months of idle cash after initial setup.
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
    ContractCandidate,
    OrderResult,
    PendingOrderRecord,
    PositionRecord,
    TradeRecord,
    now_utc_iso,
)
from leaps_bot.pricing import RateFetcher
from leaps_bot.scheduler import DailyScheduler
from leaps_bot.state import BotState
from tests.conftest import FakeAccount, FakeClock


def _make_scheduler(
    state=None,
    bootstrap_enabled=True,
    quarterly_months=None,
    allocation_window_days=7,
):
    config = AppConfig(
        dry_run=False,
        strategy=StrategyConfig(),
        pricing=PricingConfig(),
        allocation=AllocationConfig(
            quarterly_months=quarterly_months or [3, 6, 9, 12],
            max_cash_deploy_pct=0.90,
            min_cash_reserve=500.0,
            allocation_window_days=allocation_window_days,
            bootstrap_on_first_run=bootstrap_enabled,
        ),
        safety=SafetyConfig(no_trade_minutes_after_open=60),
    )
    client = MagicMock()
    client.is_market_open.return_value = True
    client.minutes_since_open.return_value = 90.0

    fake_account = FakeAccount()
    fake_account.portfolio_value = "50000.00"
    client.get_account.return_value = fake_account
    client.get_cash_available.return_value = 50000.0
    client.get_buying_power.return_value = 50000.0
    client.get_options_buying_power.return_value = 50000.0
    client.get_option_positions.return_value = []
    client.get_open_orders.return_value = []
    client.get_underlying_price.return_value = 550.0

    now = datetime(2026, 1, 15, 11, 0, 0)  # Mid-January — NOT a quarterly month
    client.get_clock.return_value = FakeClock(
        is_open=True, timestamp=now,
        next_open=now.replace(hour=9, minute=30),
        next_close=now.replace(hour=16, minute=0),
    )

    if state is None:
        state = BotState()

    rate_fetcher = MagicMock(spec=RateFetcher)
    rate_fetcher.risk_free_rate = 0.045
    rate_fetcher.dividend_yield = 0.013

    scheduler = DailyScheduler(config, client, state, rate_fetcher)
    return scheduler, state, client


# ----------------------------------------------------------------------
# Bootstrap triggers on genuine first run
# ----------------------------------------------------------------------

def test_bootstrap_triggers_on_empty_state():
    """First run with completely empty state should deploy capital
    immediately, even in January (not a quarterly month)."""
    scheduler, state, client = _make_scheduler()

    # Mock the allocator to track whether allocation was attempted
    scheduler._allocator.should_allocate_today = MagicMock(return_value=False)
    scheduler._handle_allocation = MagicMock()

    scheduler.run()

    scheduler._handle_allocation.assert_called_once()


def test_bootstrap_does_not_trigger_when_disabled():
    scheduler, state, client = _make_scheduler(bootstrap_enabled=False)

    scheduler._handle_allocation = MagicMock()

    scheduler.run()

    scheduler._handle_allocation.assert_not_called()


def test_bootstrap_does_not_trigger_when_trades_exist():
    """If the bot has traded before (even if positions are currently empty),
    it's NOT a first run — don't bootstrap. The user sold everything
    deliberately or positions rolled."""
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2025-12-15T14:30:00", order_id="old-1",
        action="buy", intent="allocate",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=100.0, total_value=20000.0,
    ))

    scheduler, _, _ = _make_scheduler(state=state)
    scheduler._handle_allocation = MagicMock()

    scheduler.run()

    scheduler._handle_allocation.assert_not_called()


def test_bootstrap_does_not_trigger_when_positions_exist():
    state = BotState()
    state.add_position(PositionRecord(
        option_symbol="SPY270318C00440000", underlying="SPY", strike=440.0,
        expiry_date="2027-03-18", purchase_date="2025-12-01",
        original_dte=365, qty=2, avg_entry_price=120.0, order_id="o1",
    ))

    scheduler, _, _ = _make_scheduler(state=state)
    scheduler._handle_allocation = MagicMock()

    scheduler.run()

    scheduler._handle_allocation.assert_not_called()


def test_bootstrap_does_not_trigger_when_allocations_exist():
    state = BotState()
    state.record_allocation(AllocationRecord(
        quarter="2025-Q4", allocated_date="2025-12-01",
        amount=20000.0, contracts_bought=["SPY270318C00440000"],
    ))

    scheduler, _, _ = _make_scheduler(state=state)
    scheduler._handle_allocation = MagicMock()

    scheduler.run()

    scheduler._handle_allocation.assert_not_called()


def test_desync_blocks_all_trading_not_just_bootstrap():
    """Critical safety: if local state is empty but the broker has positions,
    the ENTIRE trading branch must be blocked — not just bootstrap. Otherwise
    quarterly allocation could still deploy new capital, and position evaluation
    would use guessed original_dte for the broker's positions."""
    scheduler, state, client = _make_scheduler(quarterly_months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    fake_pos = MagicMock()
    fake_pos.symbol = "SPY270318C00440000"
    fake_pos.qty = "2"
    fake_pos.asset_class = __import__("alpaca.trading.enums", fromlist=["AssetClass"]).AssetClass.US_OPTION
    client.get_option_positions.return_value = [fake_pos]

    scheduler._handle_allocation = MagicMock()
    scheduler._handle_positions = MagicMock()

    summary = scheduler.run()

    scheduler._handle_allocation.assert_not_called()
    scheduler._handle_positions.assert_not_called()
    assert summary["errors"], "Desync must produce an error in summary"
    assert any("desync" in e.lower() for e in summary["errors"])


def test_desync_blocks_when_live_check_fails():
    """If the broker API is down and we can't verify live positions,
    all trading must be blocked — fail safe."""
    scheduler, state, client = _make_scheduler()
    client.get_option_positions.side_effect = Exception("API timeout")

    scheduler._handle_allocation = MagicMock()
    scheduler._handle_positions = MagicMock()

    summary = scheduler.run()

    scheduler._handle_allocation.assert_not_called()
    scheduler._handle_positions.assert_not_called()


def test_bootstrap_does_not_trigger_when_pending_orders_exist():
    """If there are pending orders (e.g., from a prior run that just submitted
    but didn't fill yet), this isn't a first run."""
    state = BotState()
    state.add_pending_order(PendingOrderRecord(
        order_id="o-1", action="buy", intent="allocate",
        option_symbol="SPY270318C00440000", qty=2,
        submitted_at=now_utc_iso(), quarter="2025-Q4",
    ))

    scheduler, _, _ = _make_scheduler(state=state)
    scheduler._handle_allocation = MagicMock()

    scheduler.run()

    scheduler._handle_allocation.assert_not_called()


# ----------------------------------------------------------------------
# Bootstrap records a proper allocation
# ----------------------------------------------------------------------

def test_bootstrap_records_allocation_for_current_quarter():
    """The bootstrap allocation should flow through the normal allocation
    pipeline (allocator.allocate → order_executor → pending_order). The
    pending order has intent=allocate and the right quarter so
    has_allocated_this_quarter blocks double-deployment when the next
    quarterly window opens."""
    scheduler, state, client = _make_scheduler()

    candidate = ContractCandidate(
        symbol="SPY270618C00440000", underlying="SPY",
        strike=440.0, expiry=date(2027, 6, 18),
        bid=119.0, ask=121.0, mid=120.0,
        delta=0.88, iv=0.20, open_interest=500,
        theoretical_price=120.5,
    )
    scheduler._finder.find_best_leaps_call = MagicMock(return_value=candidate)
    scheduler._finder.calculate_limit_price = MagicMock(return_value=120.5)

    buy_order = MagicMock()
    buy_order.id = "bootstrap-order-1"
    client.submit_limit_order.return_value = buy_order
    client.submit_market_order.return_value = buy_order

    summary = scheduler.run()

    # A pending order should have been created
    assert len(state.pending_orders) == 1
    pending = state.pending_orders[0]
    assert pending.intent == "allocate"
    assert pending.quarter  # should be set to current quarter
    assert pending.order_id == "bootstrap-order-1"


# ----------------------------------------------------------------------
# Bootstrap doesn't interfere with normal quarterly allocation
# ----------------------------------------------------------------------

def test_normal_quarterly_allocation_still_works_after_bootstrap():
    """After a bootstrap in Q1, a normal Q2 allocation should still work
    (bootstrap doesn't permanently disable quarterly logic)."""
    from unittest.mock import patch

    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2026-01-15T15:00:00+00:00", order_id="bootstrap-1",
        action="buy", intent="allocate",
        symbol="SPY270618C00440000", underlying="SPY",
        strike=440.0, expiry="2027-06-18",
        qty=2, fill_price=120.0, total_value=24000.0,
    ))
    state.record_allocation(AllocationRecord(
        quarter="2026-Q1", allocated_date="2026-01-15",
        amount=24000.0, contracts_bought=["SPY270618C00440000"],
    ))
    state.add_position(PositionRecord(
        option_symbol="SPY270618C00440000", underlying="SPY", strike=440.0,
        expiry_date="2027-06-18", purchase_date="2026-01-15",
        original_dte=519, qty=2, avg_entry_price=120.0, order_id="bootstrap-1",
    ))

    scheduler, _, _ = _make_scheduler(
        state=state,
        quarterly_months=[3, 6, 9, 12],
    )
    # State has history → neither bootstrap nor desync fires
    assert not scheduler._should_bootstrap()
    summary: dict = {"errors": []}
    assert not scheduler._is_state_desynced(summary)

    # Normal quarterly allocation should still be evaluated (not blocked)
    with patch("leaps_bot.allocator.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 3)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        assert scheduler._allocator.should_allocate_today()
