"""Tests for reporting infrastructure: trade logs, daily snapshots, run records.

These verify the data needed for PDF performance reports is being captured
correctly and persisted across runs.
"""
import json
from dataclasses import asdict
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
    DailySnapshot,
    PendingOrderRecord,
    PositionRecord,
    PositionSnapshot,
    RunRecord,
    TradeRecord,
)
from leaps_bot.pricing import RateFetcher
from leaps_bot.scheduler import DailyScheduler
from leaps_bot.state import BotState
from tests.conftest import FakeAccount, FakeClock


def _make_scheduler(pending_orders=None, positions_in_account=None, dry_run=False):
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


# -- Trade logging --

def test_buy_fill_creates_trade_record():
    pending = PendingOrderRecord(
        order_id="o-1", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=2,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED
    fake_order.filled_qty = "2"
    fake_order.filled_avg_price = "120.50"
    client.get_order.return_value = fake_order

    scheduler.run()

    assert len(state.trades) == 1
    trade = state.trades[0]
    assert trade.action == "buy"
    assert trade.intent == "allocate"
    assert trade.symbol == "SPY270618C00440000"
    assert trade.qty == 2
    assert trade.fill_price == 120.50
    assert trade.total_value == 120.50 * 2 * 100  # multiplier
    assert trade.underlying == "SPY"
    assert trade.strike == 440.0
    assert trade.underlying_price == 550.0  # captured from client
    # Buys have no realized P&L
    assert trade.realized_pnl is None
    assert trade.avg_entry_price is None


def test_sell_fill_records_realized_pnl_and_holding_period():
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
    purchase_date = today - timedelta(days=200)
    state.add_position(PositionRecord(
        option_symbol=sym, underlying="SPY", strike=440.0,
        expiry_date=expiry.isoformat(), purchase_date=purchase_date.isoformat(),
        original_dte=300, qty=2, avg_entry_price=120.0, order_id="o0",
    ))

    fake_order = MagicMock()
    fake_order.status = OrderStatus.FILLED
    fake_order.filled_qty = "2"
    fake_order.filled_avg_price = "135.00"
    client.get_order.return_value = fake_order

    # Spy on roll buy so it doesn't try to find a replacement contract
    scheduler._submit_roll_buy = MagicMock()

    scheduler.run()

    sell_trades = [t for t in state.trades if t.action == "sell"]
    assert len(sell_trades) == 1
    trade = sell_trades[0]
    assert trade.symbol == sym
    assert trade.qty == 2
    assert trade.fill_price == 135.0
    assert trade.avg_entry_price == 120.0
    # Realized P&L = (135 - 120) * 2 * 100 = $3000
    assert trade.realized_pnl == 3000.0
    assert trade.holding_days == 200


def test_partial_fills_create_separate_trade_records():
    """A partial fill increment is its own TradeRecord; the next fill is another."""
    pending = PendingOrderRecord(
        order_id="o-1", action="buy", intent="allocate",
        option_symbol="SPY270618C00440000", qty=3,
        submitted_at=datetime.now().isoformat(),
        quarter="2026-Q2",
    )
    scheduler, state, client = _make_scheduler(pending_orders=[pending])

    partial = MagicMock()
    partial.status = OrderStatus.PARTIALLY_FILLED
    partial.filled_qty = "1"
    partial.filled_avg_price = "120.0"
    client.get_order.return_value = partial
    scheduler.run()
    assert len(state.trades) == 1 and state.trades[0].qty == 1

    full = MagicMock()
    full.status = OrderStatus.FILLED
    full.filled_qty = "3"
    full.filled_avg_price = "121.0"
    client.get_order.return_value = full
    scheduler.run()

    assert len(state.trades) == 2
    # Second trade is the increment (2 contracts)
    assert state.trades[1].qty == 2


def test_realized_pnl_aggregator():
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2026-01-01T10:00:00", order_id="t1",
        action="sell", intent="roll", symbol="SPY260418C00440000",
        underlying="SPY", strike=440.0, expiry="2026-04-18",
        qty=2, fill_price=130.0, total_value=26000.0,
        avg_entry_price=120.0, realized_pnl=2000.0,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-04-01T10:00:00", order_id="t2",
        action="sell", intent="roll", symbol="SPY270417C00450000",
        underlying="SPY", strike=450.0, expiry="2027-04-17",
        qty=1, fill_price=115.0, total_value=11500.0,
        avg_entry_price=125.0, realized_pnl=-1000.0,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-04-02T10:00:00", order_id="t3",
        action="buy", intent="open", symbol="SPY270417C00440000",
        underlying="SPY", strike=440.0, expiry="2027-04-17",
        qty=1, fill_price=125.0, total_value=12500.0,
    ))
    # Buys don't contribute; net realized = 2000 - 1000 = 1000
    assert state.get_realized_pnl() == 1000.0


# -- Daily snapshot --

def test_daily_snapshot_captured_on_full_run():
    scheduler, state, client = _make_scheduler()

    scheduler.run()

    assert len(state.snapshots) == 1
    snap = state.snapshots[0]
    assert snap.date == date.today().isoformat()
    assert snap.cash == 50000.0
    assert snap.portfolio_value == 55000.0  # from FakeAccount.portfolio_value
    assert snap.options_buying_power == 50000.0
    assert "SPY" in snap.underlying_prices
    assert snap.underlying_prices["SPY"] == 550.0


def test_daily_snapshot_captures_position_marks():
    fake_pos = MagicMock()
    fake_pos.symbol = "SPY270618C00440000"
    fake_pos.qty = "2"
    fake_pos.avg_entry_price = "120.00"
    fake_pos.current_price = "130.00"
    fake_pos.market_value = "26000.00"
    fake_pos.cost_basis = "24000.00"
    fake_pos.unrealized_pl = "2000.00"
    fake_pos.unrealized_plpc = "0.0833"
    fake_pos.asset_class = __import__("alpaca.trading.enums", fromlist=["AssetClass"]).AssetClass.US_OPTION

    scheduler, state, client = _make_scheduler(positions_in_account=[fake_pos])
    scheduler.run()

    snap = state.snapshots[0]
    assert snap.num_positions == 1
    assert snap.positions_market_value == 26000.0

    pos_snap = snap.positions[0]
    assert pos_snap.symbol == "SPY270618C00440000"
    assert pos_snap.qty == 2
    assert pos_snap.avg_entry_price == 120.0
    assert pos_snap.current_price == 130.0
    assert pos_snap.market_value == 26000.0
    assert pos_snap.unrealized_pl == 2000.0
    assert pos_snap.underlying == "SPY"
    assert pos_snap.strike == 440.0


def test_no_snapshot_when_skipped():
    scheduler, state, client = _make_scheduler()
    client.is_market_open.return_value = False  # market closed → preflight skips

    scheduler.run()

    assert len(state.snapshots) == 0, "No snapshot when run is skipped"


def test_no_snapshot_in_dry_run():
    scheduler, state, client = _make_scheduler(dry_run=True)
    scheduler.run()
    assert len(state.snapshots) == 0, "Dry-run must not write snapshots (state mutation)"


# -- Run record --

def test_run_record_appended_on_normal_run():
    scheduler, state, client = _make_scheduler()
    scheduler.run()

    assert len(state.runs) == 1
    run = state.runs[0]
    assert not run.skipped
    assert run.skip_reason is None
    assert run.duration_seconds >= 0
    assert run.portfolio_value == 55000.0


def test_run_record_appended_on_skipped_run():
    scheduler, state, client = _make_scheduler()
    client.is_market_open.return_value = False
    scheduler.run()

    assert len(state.runs) == 1
    run = state.runs[0]
    assert run.skipped
    assert run.skip_reason and "closed" in run.skip_reason.lower()


def test_run_record_skipped_in_dry_run():
    scheduler, state, client = _make_scheduler(dry_run=True)
    scheduler.run()
    assert len(state.runs) == 0, "Dry-run must not write run records"


# -- Persistence --

def test_trades_and_snapshots_round_trip(tmp_path):
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2026-06-03T15:33:00",
        order_id="o-1", action="buy", intent="allocate",
        symbol="SPY270618C00440000",
        underlying="SPY", strike=440.0, expiry="2027-06-18",
        qty=2, fill_price=120.5, total_value=24100.0,
        underlying_price=550.0,
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-06-03",
        timestamp="2026-06-03T15:33:00",
        cash=50000.0,
        options_buying_power=50000.0,
        portfolio_value=74100.0,
        positions_market_value=24100.0,
        num_positions=1,
        underlying_prices={"SPY": 550.0},
        positions=[PositionSnapshot(
            symbol="SPY270618C00440000", underlying="SPY",
            strike=440.0, expiry="2027-06-18",
            qty=2, avg_entry_price=120.5, current_price=120.5,
            market_value=24100.0, cost_basis=24100.0,
            unrealized_pl=0.0, unrealized_plpc=0.0,
            days_remaining=380,
        )],
    ))
    state.add_run(RunRecord(
        timestamp="2026-06-03T15:33:00",
        duration_seconds=5.2,
        skipped=False,
        actions=["fill_buy:SPY270618C00440000"],
        real_trades_today=True,
        portfolio_value=74100.0,
    ))

    path = tmp_path / "state.json"
    state.save(path)

    loaded = BotState.load(path)
    assert len(loaded.trades) == 1
    assert loaded.trades[0].symbol == "SPY270618C00440000"
    assert loaded.trades[0].underlying_price == 550.0

    assert len(loaded.snapshots) == 1
    snap = loaded.snapshots[0]
    assert snap.cash == 50000.0
    assert snap.underlying_prices["SPY"] == 550.0
    assert len(snap.positions) == 1
    assert snap.positions[0].unrealized_pl == 0.0

    assert len(loaded.runs) == 1
    assert loaded.runs[0].real_trades_today


def test_load_state_without_reporting_fields_back_compat(tmp_path):
    """Old state files (pre-reporting) load fine with empty reporting lists."""
    legacy = {
        "positions": [],
        "allocations": [],
        "pending_orders": [],
        "last_run": "2026-04-15T10:30:00",
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(legacy))

    state = BotState.load(path)
    assert state.trades == []
    assert state.snapshots == []
    assert state.runs == []
