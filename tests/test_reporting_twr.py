"""Regression tests for time-weighted return and ordering fixes.

Each test pins down a specific bug from code review:
- Chart compared raw portfolio value to SPY without cash-flow adjustment
  (deposits looked like outperformance)
- total_invested = sum(allocations) overstates external capital because
  each quarter's allocation can include redeployed gains from prior periods
- YTD return didn't use TWR — deposits inflated the percentage
- Snapshots/trades consumed in insertion order, not timestamp order
"""
import csv
from datetime import date, datetime, timedelta

import pytest

from leaps_bot.models import (
    AllocationRecord,
    DailySnapshot,
    PositionSnapshot,
    TradeRecord,
)
from leaps_bot.reporting import CashFlow, ReportGenerator
from leaps_bot.state import BotState


# ----------------------------------------------------------------------
# External cash flow detection
# ----------------------------------------------------------------------

def test_no_external_flows_for_self_consistent_snapshots():
    """Cash changes match trade activity → no flows detected."""
    state = BotState()
    state.add_snapshot(DailySnapshot(
        date="2026-03-01", timestamp="2026-03-01T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
        underlying_prices={"SPY": 540.0},
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-03-02T15:33:00", order_id="o-1",
        action="buy", intent="allocate",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=100.0, total_value=20000.0,
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-03-03", timestamp="2026-03-03T15:33:00",
        cash=0.0, options_buying_power=0.0,
        portfolio_value=20000.0, positions_market_value=20000.0, num_positions=1,
        underlying_prices={"SPY": 540.0},
    ))

    flows = ReportGenerator(state).compute_external_flows()
    assert flows == []


def test_detects_external_deposit():
    """Cash jumped without trade activity explaining it → deposit detected."""
    state = BotState()
    state.add_snapshot(DailySnapshot(
        date="2026-03-01", timestamp="2026-03-01T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))
    # No trades. Cash went up by $10k → must be a deposit.
    state.add_snapshot(DailySnapshot(
        date="2026-04-01", timestamp="2026-04-01T15:30:00",
        cash=30000.0, options_buying_power=30000.0,
        portfolio_value=30000.0, positions_market_value=0.0, num_positions=0,
    ))

    flows = ReportGenerator(state).compute_external_flows()
    assert len(flows) == 1
    assert flows[0].amount == pytest.approx(10000.0)
    assert flows[0].timestamp == "2026-04-01T15:30:00"


def test_detects_external_withdrawal():
    state = BotState()
    state.add_snapshot(DailySnapshot(
        date="2026-03-01", timestamp="2026-03-01T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-04-01", timestamp="2026-04-01T15:30:00",
        cash=15000.0, options_buying_power=15000.0,
        portfolio_value=15000.0, positions_market_value=0.0, num_positions=0,
    ))

    flows = ReportGenerator(state).compute_external_flows()
    assert len(flows) == 1
    assert flows[0].amount == pytest.approx(-5000.0)


def test_filters_small_residuals():
    """Small cash differences (rounding, fees) below threshold are ignored."""
    state = BotState()
    state.add_snapshot(DailySnapshot(
        date="2026-03-01", timestamp="2026-03-01T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-04-01", timestamp="2026-04-01T15:30:00",
        cash=19990.0,  # $10 difference, well below $50 threshold
        options_buying_power=19990.0,
        portfolio_value=19990.0, positions_market_value=0.0, num_positions=0,
    ))

    flows = ReportGenerator(state).compute_external_flows()
    assert flows == [], "Residual below threshold must not be reported as a flow"


# ----------------------------------------------------------------------
# P1.1: TWR chart
# ----------------------------------------------------------------------

def test_twr_isolates_returns_from_contributions():
    """A pure deposit between snapshots must NOT show as a return.

    Scenario:
      Day 1: $10,000 cash, $10,000 portfolio
      Day 90: still $10,000 (no trades, no gains, no losses)
      Day 91: user deposits $10,000 → $20,000 cash
      Day 180: still $20,000 (no trades)

    Naive return: $20k/$10k - 1 = +100%. WRONG.
    TWR: each period flat, so 0%. CORRECT.
    """
    state = BotState()
    state.add_snapshot(DailySnapshot(
        date="2026-01-01", timestamp="2026-01-01T15:30:00",
        cash=10000.0, options_buying_power=10000.0,
        portfolio_value=10000.0, positions_market_value=0.0, num_positions=0,
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-04-01", timestamp="2026-04-01T15:30:00",
        cash=10000.0, options_buying_power=10000.0,
        portfolio_value=10000.0, positions_market_value=0.0, num_positions=0,
    ))
    # User deposits $10k between these snapshots
    state.add_snapshot(DailySnapshot(
        date="2026-04-02", timestamp="2026-04-02T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-07-01", timestamp="2026-07-01T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))

    series = ReportGenerator(state).compute_twr_series()
    final_growth = series[-1][1]
    assert final_growth == pytest.approx(1.0, abs=0.001), (
        f"TWR should be 1.0 (no return — only deposits), got {final_growth}. "
        "If this fails, contributions are leaking into the return calc."
    )


def test_twr_captures_actual_gains():
    """A real $5k gain on a $20k position must show as +25% TWR (no flows)."""
    state = BotState()
    state.add_snapshot(DailySnapshot(
        date="2026-03-01", timestamp="2026-03-01T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-03-02T15:33:00", order_id="b1",
        action="buy", intent="allocate",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=100.0, total_value=20000.0,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-11-01T14:30:00", order_id="s1",
        action="sell", intent="close",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=125.0, total_value=25000.0,
        avg_entry_price=100.0, realized_pnl=5000.0, holding_days=244,
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-11-02", timestamp="2026-11-02T15:30:00",
        cash=25000.0, options_buying_power=25000.0,
        portfolio_value=25000.0, positions_market_value=0.0, num_positions=0,
    ))

    series = ReportGenerator(state).compute_twr_series()
    final_growth = series[-1][1]
    # 25000/20000 = 1.25 → +25%
    assert final_growth == pytest.approx(1.25, abs=0.001)


def test_twr_with_deposit_and_gain():
    """Mix scenario: $10k initial, deposit $10k mid-period, then position gains.

    The TWR should reflect ONLY the position's gain, not the deposit.
    """
    state = BotState()
    # Period 1: flat (no return)
    state.add_snapshot(DailySnapshot(
        date="2026-01-01", timestamp="2026-01-01T15:30:00",
        cash=10000.0, options_buying_power=10000.0,
        portfolio_value=10000.0, positions_market_value=0.0, num_positions=0,
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-04-01", timestamp="2026-04-01T15:30:00",
        cash=10000.0, options_buying_power=10000.0,
        portfolio_value=10000.0, positions_market_value=0.0, num_positions=0,
    ))
    # Deposit $10k. Cash now $20k.
    state.add_snapshot(DailySnapshot(
        date="2026-04-15", timestamp="2026-04-15T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))
    # Position grows from 20k → 22k (10% gain on the period)
    state.add_snapshot(DailySnapshot(
        date="2026-07-01", timestamp="2026-07-01T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=22000.0, positions_market_value=2000.0, num_positions=0,
    ))

    series = ReportGenerator(state).compute_twr_series()
    final_growth = series[-1][1]
    # Period 1: flat (1.0)
    # Period 2 (with deposit): (20000 - 10000 deposit) / 10000 = 1.0 (flat after stripping flow)
    # Period 3: 22000 / 20000 = 1.10 (10% gain)
    # Cumulative: 1.0 * 1.0 * 1.10 = 1.10
    assert final_growth == pytest.approx(1.10, abs=0.005)


# ----------------------------------------------------------------------
# P1.2: total_invested reflects net external contributions
# ----------------------------------------------------------------------

def test_total_invested_is_net_external_contributions_not_allocation_sum():
    """When the bot redeploys realized gains in a subsequent quarter, the
    new allocation amount includes those gains. Summing allocation amounts
    overstates external capital. total_invested should be initial deposit
    plus subsequent external deposits only."""
    state = BotState()
    # Initial deposit: $20k
    state.add_snapshot(DailySnapshot(
        date="2026-02-28", timestamp="2026-02-28T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))
    # Q1 deploys $20k
    state.record_allocation(AllocationRecord(
        quarter="2026-Q1", allocated_date="2026-03-01",
        amount=20000.0, contracts_bought=["SPY270318C00440000"],
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-03-01T15:33:00", order_id="b1",
        action="buy", intent="allocate",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=100.0, total_value=20000.0,
    ))
    # Q1 sells for $25k profit → cash = $25k (no new deposit)
    state.add_trade(TradeRecord(
        timestamp="2026-05-15T14:30:00", order_id="s1",
        action="sell", intent="roll",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=125.0, total_value=25000.0,
        avg_entry_price=100.0, realized_pnl=5000.0, holding_days=75,
    ))
    # Q2 redeploys all $25k (gains included)
    state.record_allocation(AllocationRecord(
        quarter="2026-Q2", allocated_date="2026-06-01",
        amount=25000.0, contracts_bought=["SPY280317C00440000"],
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-06-01T15:33:00", order_id="b2",
        action="buy", intent="allocate",
        symbol="SPY280317C00440000", underlying="SPY",
        strike=440.0, expiry="2028-03-17",
        qty=2, fill_price=125.0, total_value=25000.0,
    ))
    # Final snapshot: $25k in positions, $0 cash
    state.add_snapshot(DailySnapshot(
        date="2026-06-02", timestamp="2026-06-02T15:30:00",
        cash=0.0, options_buying_power=0.0,
        portfolio_value=25000.0, positions_market_value=25000.0, num_positions=1,
    ))

    summary = ReportGenerator(state).compute_summary()

    # total_deployed sums allocation amounts: $20k + $25k = $45k.
    # But user only ever contributed $20k externally — the second allocation
    # was funded by realized gains from Q1.
    assert summary.total_deployed == 45000.0
    assert summary.total_invested == 20000.0, (
        "total_invested must reflect net external capital, not allocation sum. "
        "Q2's $25k allocation is realized gains being redeployed, not new capital."
    )
    # Total return: $25k portfolio - $20k contributed = $5k
    assert summary.total_return_dollar == 5000.0


# ----------------------------------------------------------------------
# P1.2: YTD uses TWR
# ----------------------------------------------------------------------

def test_ytd_return_strips_out_deposits():
    """YTD must reflect investment performance only, not capital additions."""
    state = BotState()
    current_year = date.today().year
    # Year-start snapshot
    state.add_snapshot(DailySnapshot(
        date=f"{current_year}-01-02", timestamp=f"{current_year}-01-02T15:30:00",
        cash=10000.0, options_buying_power=10000.0,
        portfolio_value=10000.0, positions_market_value=0.0, num_positions=0,
    ))
    # User deposits $10k mid-year (no investment performance change)
    state.add_snapshot(DailySnapshot(
        date=f"{current_year}-04-01", timestamp=f"{current_year}-04-01T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))
    # No further changes
    state.add_snapshot(DailySnapshot(
        date=f"{current_year}-06-01", timestamp=f"{current_year}-06-01T15:30:00",
        cash=20000.0, options_buying_power=20000.0,
        portfolio_value=20000.0, positions_market_value=0.0, num_positions=0,
    ))

    summary = ReportGenerator(state).compute_summary()
    # YTD return must be 0% — no actual investment gains, only a deposit.
    # Naive ratio (20k/10k - 1 = 100%) would be wrong.
    assert summary.ytd_return_pct == pytest.approx(0.0, abs=0.1), (
        f"YTD return inflated by deposit: got {summary.ytd_return_pct}%, expected ~0%"
    )


# ----------------------------------------------------------------------
# P2.1: Snapshots sorted by timestamp, not insertion order
# ----------------------------------------------------------------------

def test_summary_uses_timestamp_order_not_insertion_order():
    """If snapshots are appended out of order, results must still be correct.

    Use position-MV-driven growth (no cash change) so the flow detector
    doesn't infer phantom deposits.
    """
    state = BotState()
    # Append in REVERSE chronological order
    state.add_snapshot(DailySnapshot(
        date="2026-12-30", timestamp="2026-12-30T15:30:00",
        cash=0.0, options_buying_power=0.0,
        portfolio_value=15000.0, positions_market_value=15000.0, num_positions=1,
        underlying_prices={"SPY": 600.0},
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-01-02", timestamp="2026-01-02T15:30:00",
        cash=0.0, options_buying_power=0.0,
        portfolio_value=10000.0, positions_market_value=10000.0, num_positions=1,
        underlying_prices={"SPY": 540.0},
    ))

    summary = ReportGenerator(state).compute_summary()
    assert summary.inception_date == "2026-01-02"
    assert summary.total_invested == 10000.0
    assert summary.portfolio_value == 15000.0
    # SPY benchmark goes from 540 → 600 = +11.1%
    assert summary.spy_total_return_pct == pytest.approx(11.111, abs=0.1)
    assert summary.num_external_flows == 0


def test_twr_uses_timestamp_order_not_insertion_order():
    """Position appreciates 10% — gain comes from positions_market_value, not
    cash, so no flow is detected. Even with reversed insertion order, TWR
    must reflect chronological growth from $10k → $11k."""
    state = BotState()
    state.add_snapshot(DailySnapshot(
        date="2026-04-01", timestamp="2026-04-01T15:30:00",
        cash=0.0, options_buying_power=0.0,
        portfolio_value=11000.0, positions_market_value=11000.0, num_positions=1,
    ))
    state.add_snapshot(DailySnapshot(
        date="2026-01-01", timestamp="2026-01-01T15:30:00",
        cash=0.0, options_buying_power=0.0,
        portfolio_value=10000.0, positions_market_value=10000.0, num_positions=1,
    ))

    series = ReportGenerator(state).compute_twr_series()
    # Sorted by timestamp: oldest first
    assert series[0][0] == "2026-01-01T15:30:00"
    assert series[-1][1] == pytest.approx(1.10, abs=0.001)


# ----------------------------------------------------------------------
# P2.2: Trade history sorted by timestamp in PDF table
# ----------------------------------------------------------------------

def test_trade_history_table_sorted_by_timestamp_descending():
    """When trades are inserted out of order (e.g., late-arriving fills from
    prior days), the PDF table must still show them in chronological order."""
    state = BotState()
    # Insert in non-chronological order
    state.add_trade(TradeRecord(
        timestamp="2026-07-01T10:00:00", order_id="recent",
        action="buy", intent="open",
        symbol="SPY280317C00440000", underlying="SPY",
        strike=440.0, expiry="2028-03-17",
        qty=1, fill_price=130.0, total_value=13000.0,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-01-15T14:00:00", order_id="older",
        action="buy", intent="allocate",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=100.0, total_value=20000.0,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-04-01T10:00:00", order_id="middle",
        action="sell", intent="close",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=110.0, total_value=22000.0,
        avg_entry_price=100.0, realized_pnl=2000.0, holding_days=76,
    ))

    # We can't easily inspect the rendered PDF table directly, but we can call
    # the table builder and verify the row order it produces.
    table = ReportGenerator(state)._trades_table()
    # ReportLab Table stores rows in `_cellvalues`. First row is header.
    rows = table._cellvalues
    order_ids = [row[3] for row in rows[1:]]  # symbol column index
    # All three trades represented
    assert len(order_ids) == 3
    # Most recent (2026-07-01) first, oldest (2026-01-15) last
    dates_in_order = [row[0] for row in rows[1:]]
    assert dates_in_order == ["2026-07-01", "2026-04-01", "2026-01-15"]
