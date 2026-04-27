import json
from pathlib import Path

from leaps_bot.models import AllocationRecord, PositionRecord
from leaps_bot.state import BotState


def test_save_and_load(tmp_path):
    path = tmp_path / "state.json"
    state = BotState()
    state.add_position(PositionRecord(
        option_symbol="SPY270417C00440000",
        underlying="SPY",
        strike=440.0,
        expiry_date="2027-04-17",
        purchase_date="2026-06-15",
        original_dte=306,
        qty=2,
        avg_entry_price=120.50,
        order_id="order-123",
    ))
    state.last_run = "2026-06-15T11:00:00"
    state.save(path)

    loaded = BotState.load(path)
    assert len(loaded.positions) == 1
    assert loaded.positions[0].option_symbol == "SPY270417C00440000"
    assert loaded.positions[0].original_dte == 306
    assert loaded.last_run == "2026-06-15T11:00:00"


def test_load_missing_file(tmp_path):
    path = tmp_path / "nonexistent.json"
    state = BotState.load(path)
    assert len(state.positions) == 0
    assert state.last_run is None


def test_get_position():
    state = BotState()
    state.add_position(PositionRecord(
        option_symbol="SPY270417C00440000",
        underlying="SPY", strike=440.0,
        expiry_date="2027-04-17", purchase_date="2026-06-15",
        original_dte=306, qty=2, avg_entry_price=120.50,
        order_id="order-123",
    ))
    assert state.get_position("SPY270417C00440000") is not None
    assert state.get_position("NONEXISTENT") is None


def test_remove_position():
    state = BotState()
    state.add_position(PositionRecord(
        option_symbol="SPY270417C00440000",
        underlying="SPY", strike=440.0,
        expiry_date="2027-04-17", purchase_date="2026-06-15",
        original_dte=306, qty=2, avg_entry_price=120.50,
        order_id="order-123",
    ))
    state.remove_position("SPY270417C00440000")
    assert len(state.positions) == 0


def test_quarterly_allocation_tracking():
    state = BotState()
    today = "2026-06-15"
    assert state.current_quarter_key(today) == "2026-Q2"
    assert not state.has_allocated_this_quarter(today)

    state.record_allocation(AllocationRecord(
        quarter="2026-Q2",
        allocated_date="2026-06-15",
        amount=10000.0,
        contracts_bought=["SPY270417C00440000"],
    ))
    assert state.has_allocated_this_quarter(today)


def test_quarter_key_boundaries():
    state = BotState()
    assert state.current_quarter_key("2026-01-15") == "2026-Q1"
    assert state.current_quarter_key("2026-03-31") == "2026-Q1"
    assert state.current_quarter_key("2026-04-01") == "2026-Q2"
    assert state.current_quarter_key("2026-07-01") == "2026-Q3"
    assert state.current_quarter_key("2026-10-01") == "2026-Q4"
    assert state.current_quarter_key("2026-12-31") == "2026-Q4"
