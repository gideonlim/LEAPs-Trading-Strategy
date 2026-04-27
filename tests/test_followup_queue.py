"""Tests for the FollowupAction queue.

Reconciliation queues follow-up actions (e.g., roll buys after a sell fills)
instead of submitting orders inline. This keeps the safety invariant clean:
- Monitor: reconciles fills, may queue followups, never submits
- Trading run: drains the queue, submits the queued orders

These tests verify the queueing behavior across the run/monitor split.
"""
import json
from datetime import date, datetime, timedelta, timezone
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
    FollowupAction,
    PendingOrderRecord,
    PositionRecord,
    now_utc_iso,
)
from leaps_bot.pricing import RateFetcher
from leaps_bot.scheduler import DailyScheduler
from leaps_bot.state import BotState
from tests.conftest import FakeAccount, FakeClock


def _make_scheduler(pending_orders=None, pending_followups=None):
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
    if pending_followups:
        state.pending_followups.extend(pending_followups)

    rate_fetcher = MagicMock(spec=RateFetcher)
    rate_fetcher.risk_free_rate = 0.045
    rate_fetcher.dividend_yield = 0.013

    return DailyScheduler(config, client, state, rate_fetcher), state, client


def _filled_sell_order(qty=2, fill_price=125.50):
    order = MagicMock()
    order.status = OrderStatus.FILLED
    order.filled_qty = str(qty)
    order.filled_avg_price = str(fill_price)
    order.filled_at = datetime(2026, 6, 3, 13, 30, 0, tzinfo=timezone.utc)
    order.updated_at = order.filled_at
    order.submitted_at = order.filled_at
    return order


# ----------------------------------------------------------------------
# Monitor queues, doesn't submit
# ----------------------------------------------------------------------

def test_monitor_queues_followup_for_filled_roll_sell():
    """A sell-with-roll-intent that fills during a monitor pass must be
    QUEUED as a FollowupAction. The roll buy itself is NOT submitted yet."""
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
    client.get_order.return_value = _filled_sell_order(qty=2)

    # Spy on _submit_roll_buy — must NOT be called during monitor
    scheduler._submit_roll_buy = MagicMock()

    scheduler.monitor()

    # Followup queued
    assert len(state.pending_followups) == 1
    f = state.pending_followups[0]
    assert f.action_type == "roll"
    assert f.underlying == "SPY"
    assert f.qty == 2
    assert f.sourced_from_order_id == "sell-1"
    # Buy NOT submitted
    scheduler._submit_roll_buy.assert_not_called()
    client.submit_market_order.assert_not_called()
    client.submit_limit_order.assert_not_called()


def test_monitor_does_not_queue_followup_when_zero_filled():
    """If a sell-with-roll cancels with 0 fills, no followup should be
    queued (we have nothing to roll)."""
    today = date.today()
    expiry = today + timedelta(days=100)
    sym = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    pending = PendingOrderRecord(
        order_id="sell-zero", action="sell", intent="roll",
        option_symbol=sym, qty=2,
        submitted_at=now_utc_iso(),
        underlying="SPY",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    canceled_order = MagicMock()
    canceled_order.status = OrderStatus.CANCELED
    canceled_order.filled_qty = "0"
    canceled_order.filled_avg_price = "0"
    canceled_order.filled_at = None
    canceled_order.updated_at = datetime(2026, 6, 3, 13, 30, 0, tzinfo=timezone.utc)
    canceled_order.submitted_at = canceled_order.updated_at
    client.get_order.return_value = canceled_order

    scheduler.monitor()

    assert state.pending_followups == []


# ----------------------------------------------------------------------
# Run drains the queue and submits orders
# ----------------------------------------------------------------------

def test_run_processes_queued_followup_and_submits_buy():
    """When run() starts and finds queued followups, it should drain them
    and submit the corresponding buy orders."""
    queued = FollowupAction(
        action_type="roll",
        underlying="SPY",
        qty=2,
        sourced_from_order_id="sell-1",
        queued_at=now_utc_iso(),
    )
    scheduler, state, client = _make_scheduler(pending_followups=[queued])

    scheduler._submit_roll_buy = MagicMock()

    scheduler.run()

    # Followup drained
    assert state.pending_followups == []
    # Roll buy submitted with the queued underlying + qty
    scheduler._submit_roll_buy.assert_called_once()
    args = scheduler._submit_roll_buy.call_args
    assert args[0][0] == "SPY"
    assert args[0][1] == 2


def test_run_drains_multiple_queued_followups():
    queued = [
        FollowupAction(
            action_type="roll", underlying="SPY", qty=1,
            sourced_from_order_id="sell-1", queued_at=now_utc_iso(),
        ),
        FollowupAction(
            action_type="roll", underlying="SPY", qty=3,
            sourced_from_order_id="sell-2", queued_at=now_utc_iso(),
        ),
    ]
    scheduler, state, client = _make_scheduler(pending_followups=queued)
    scheduler._submit_roll_buy = MagicMock()

    scheduler.run()

    assert scheduler._submit_roll_buy.call_count == 2
    assert state.pending_followups == []


def test_unknown_followup_action_type_is_preserved():
    """Future-proofing: an unknown action_type from a future bot version
    that wrote state we don't recognize should NOT be silently dropped."""
    queued = FollowupAction(
        action_type="future_action_we_dont_know_about",
        underlying="SPY", qty=2,
        sourced_from_order_id="sell-1",
        queued_at=now_utc_iso(),
    )
    scheduler, state, _ = _make_scheduler(pending_followups=[queued])
    scheduler._submit_roll_buy = MagicMock()

    scheduler.run()

    # Unknown actions stay queued for manual review, not silently dropped
    assert len(state.pending_followups) == 1
    scheduler._submit_roll_buy.assert_not_called()


# ----------------------------------------------------------------------
# Persistence: followup queue survives restart
# ----------------------------------------------------------------------

def test_followup_queue_round_trips_through_save_load(tmp_path):
    state = BotState()
    state.pending_followups.append(FollowupAction(
        action_type="roll",
        underlying="SPY",
        qty=2,
        sourced_from_order_id="sell-1",
        queued_at="2026-06-03T15:30:00+00:00",
    ))

    path = tmp_path / "state.json"
    state.save(path)
    loaded = BotState.load(path)

    assert len(loaded.pending_followups) == 1
    f = loaded.pending_followups[0]
    assert f.action_type == "roll"
    assert f.underlying == "SPY"
    assert f.qty == 2
    assert f.sourced_from_order_id == "sell-1"


def test_loading_state_without_followups_field_back_compat(tmp_path):
    """Old state files predate this field. Loading must not crash."""
    legacy = {
        "positions": [],
        "allocations": [],
        "pending_orders": [],
        "trades": [],
        "snapshots": [],
        "runs": [],
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(legacy))

    state = BotState.load(path)
    assert state.pending_followups == []


# ----------------------------------------------------------------------
# End-to-end: monitor queues, next run drains
# ----------------------------------------------------------------------

def test_failed_roll_followup_stays_queued_for_retry():
    """If _submit_roll_buy fails (no contract found, API error), the followup
    must stay in the queue so the next morning's run retries it. Without this,
    a transient failure permanently drops the roll and the strategy goes
    unintentionally unrolled."""
    queued = FollowupAction(
        action_type="roll", underlying="SPY", qty=2,
        sourced_from_order_id="sell-1", queued_at=now_utc_iso(),
    )
    scheduler, state, client = _make_scheduler(pending_followups=[queued])

    # Force _submit_roll_buy to fail: contract finder returns None
    scheduler._finder.find_best_leaps_call = MagicMock(return_value=None)

    scheduler.run()

    # Followup must stay queued, not be silently dropped
    assert len(state.pending_followups) == 1, (
        "Failed roll followup was dropped. It must stay queued so the next "
        "morning's run retries it."
    )
    assert state.pending_followups[0].sourced_from_order_id == "sell-1"


def test_successful_roll_followup_is_removed_from_queue():
    """After a successful _submit_roll_buy, the followup must be cleared
    so it's not retried again."""
    queued = FollowupAction(
        action_type="roll", underlying="SPY", qty=2,
        sourced_from_order_id="sell-1", queued_at=now_utc_iso(),
    )
    scheduler, state, client = _make_scheduler(pending_followups=[queued])

    # Force success: mock roll buy to return True
    scheduler._submit_roll_buy = MagicMock(return_value=True)

    scheduler.run()

    assert state.pending_followups == [], "Successful roll followup should be cleared"


def test_full_cycle_monitor_queues_then_run_executes():
    """Day 1 PM monitor: a roll sell fills → queue followup, no buy yet.
    Day 2 AM run: drains queue → submits buy."""
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
    client.get_order.return_value = _filled_sell_order(qty=2)

    scheduler._submit_roll_buy = MagicMock()

    # === Day 1 PM monitor ===
    scheduler.monitor()
    assert len(state.pending_followups) == 1
    scheduler._submit_roll_buy.assert_not_called()
    assert state.get_position(sym) is None  # sell reconciled, position removed
    assert pending not in state.pending_orders

    # === Day 2 AM run (no pending orders left, but followup queued) ===
    scheduler.run()
    scheduler._submit_roll_buy.assert_called_once()
    assert state.pending_followups == []
