from datetime import date, timedelta
from unittest.mock import MagicMock

from leaps_bot.config import AppConfig, SafetyConfig, StrategyConfig
from leaps_bot.models import Action, PositionRecord
from leaps_bot.position_manager import PositionManager
from leaps_bot.state import BotState
from tests.conftest import FakePosition


def _make_manager(positions, state_positions=None, sell_fraction=0.333, emergency_days=30):
    config = AppConfig(
        strategy=StrategyConfig(sell_threshold_fraction=sell_fraction),
        safety=SafetyConfig(emergency_sell_days=emergency_days),
    )
    state = BotState()
    if state_positions:
        for sp in state_positions:
            state.add_position(sp)

    client = MagicMock()
    client.get_option_positions.return_value = positions
    return PositionManager(client, config, state)


def test_hold_position():
    today = date.today()
    expiry = today + timedelta(days=300)
    symbol = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    mgr = _make_manager(
        positions=[FakePosition(symbol=symbol, qty="2")],
        state_positions=[PositionRecord(
            option_symbol=symbol, underlying="SPY", strike=440.0,
            expiry_date=expiry.isoformat(), purchase_date=(today - timedelta(days=65)).isoformat(),
            original_dte=365, qty=2, avg_entry_price=120.0, order_id="o1",
        )],
    )
    actions = mgr.evaluate_positions()
    assert len(actions) == 1
    assert actions[0].action == Action.HOLD


def test_sell_at_threshold():
    today = date.today()
    # 365 DTE originally, 1/3 = 122 days. Position has 120 days left → should sell
    expiry = today + timedelta(days=120)
    symbol = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    mgr = _make_manager(
        positions=[FakePosition(symbol=symbol, qty="2")],
        state_positions=[PositionRecord(
            option_symbol=symbol, underlying="SPY", strike=440.0,
            expiry_date=expiry.isoformat(), purchase_date=(today - timedelta(days=245)).isoformat(),
            original_dte=365, qty=2, avg_entry_price=120.0, order_id="o1",
        )],
    )
    actions = mgr.evaluate_positions()
    assert len(actions) == 1
    assert actions[0].action == Action.SELL


def test_emergency_sell():
    today = date.today()
    expiry = today + timedelta(days=25)
    symbol = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    mgr = _make_manager(
        positions=[FakePosition(symbol=symbol, qty="1")],
        state_positions=[PositionRecord(
            option_symbol=symbol, underlying="SPY", strike=440.0,
            expiry_date=expiry.isoformat(), purchase_date=(today - timedelta(days=340)).isoformat(),
            original_dte=365, qty=1, avg_entry_price=120.0, order_id="o1",
        )],
        emergency_days=30,
    )
    actions = mgr.evaluate_positions()
    assert len(actions) == 1
    assert actions[0].action == Action.EMERGENCY_SELL


def test_18_month_option_sell_threshold():
    today = date.today()
    # 548 DTE originally (18 months), 1/3 = 183 days. Position has 180 days left → should sell
    original_dte = 548
    expiry = today + timedelta(days=180)
    symbol = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    mgr = _make_manager(
        positions=[FakePosition(symbol=symbol, qty="3")],
        state_positions=[PositionRecord(
            option_symbol=symbol, underlying="SPY", strike=440.0,
            expiry_date=expiry.isoformat(),
            purchase_date=(today - timedelta(days=original_dte - 180)).isoformat(),
            original_dte=original_dte, qty=3, avg_entry_price=130.0, order_id="o2",
        )],
    )
    actions = mgr.evaluate_positions()
    assert actions[0].action == Action.SELL


def test_no_positions():
    mgr = _make_manager(positions=[])
    actions = mgr.evaluate_positions()
    assert actions == []
