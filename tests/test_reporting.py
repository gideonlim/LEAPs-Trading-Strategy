"""Tests for the reporting module: summary computation, PDF generation, CSV exports."""
import csv
from datetime import date

import pytest

from leaps_bot.models import (
    AllocationRecord,
    DailySnapshot,
    PositionSnapshot,
    TradeRecord,
)
from leaps_bot.reporting import ReportGenerator
from leaps_bot.state import BotState


def _populated_state() -> BotState:
    """A realistic state with self-consistent snapshots and trades.

    Initial deposit: $50,500 cash (no positions).
    Q1 buy 2 @ $100 = $20,000  → cash $30,500, position cost $20,000
    Q2 buy 1 @ $110 = $11,000  → cash $19,500, position cost $31,000
    Q1 sell 2 @ $120 = $24,000 → cash $43,500, P&L +$4,000
    Q1 roll buy 2 @ $125 = $25,000 → cash $18,500, position cost $36,000
    Q2 sell 1 @ $105 = $10,500 → cash $29,000, P&L -$500
    Final position (roll): 2 contracts marked at $147.50 → MV $29,500
    Final portfolio = $29,000 cash + $29,500 MV = $58,500
    No external deposits/withdrawals → flow detector should find none.
    """
    state = BotState()

    state.record_allocation(AllocationRecord(
        quarter="2026-Q1", allocated_date="2026-03-02",
        amount=20000.0, contracts_bought=["SPY270318C00440000"],
    ))
    state.record_allocation(AllocationRecord(
        quarter="2026-Q2", allocated_date="2026-06-02",
        amount=11000.0, contracts_bought=["SPY270618C00450000"],
    ))

    state.add_trade(TradeRecord(
        timestamp="2026-03-02T15:33:00", order_id="o-1",
        action="buy", intent="allocate",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=100.0, total_value=20000.0,
        underlying_price=540.0,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-06-02T15:33:00", order_id="o-4",
        action="buy", intent="allocate",
        symbol="SPY270618C00450000", underlying="SPY",
        strike=450.0, expiry="2027-06-18",
        qty=1, fill_price=110.0, total_value=11000.0,
        underlying_price=550.0,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:30:00", order_id="o-2",
        action="sell", intent="roll",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=120.0, total_value=24000.0,
        underlying_price=560.0,
        avg_entry_price=100.0, realized_pnl=4000.0, holding_days=258,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:35:00", order_id="o-3",
        action="buy", intent="open",
        symbol="SPY280317C00450000", underlying="SPY",
        strike=450.0, expiry="2028-03-17",
        qty=2, fill_price=125.0, total_value=25000.0,
        underlying_price=560.0,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-12-15T14:30:00", order_id="o-5",
        action="sell", intent="close",
        symbol="SPY270618C00450000", underlying="SPY",
        strike=450.0, expiry="2027-06-18",
        qty=1, fill_price=105.0, total_value=10500.0,
        underlying_price=545.0,
        avg_entry_price=110.0, realized_pnl=-500.0, holding_days=196,
    ))

    # Inception snapshot — pure cash, before any trading
    state.add_snapshot(DailySnapshot(
        date="2026-03-01", timestamp="2026-03-01T15:30:00",
        cash=50500.0, options_buying_power=50500.0,
        portfolio_value=50500.0, positions_market_value=0.0,
        num_positions=0, underlying_prices={"SPY": 540.0},
        positions=[],
    ))
    # Final snapshot — after all trades, cash is internally consistent
    state.add_snapshot(DailySnapshot(
        date="2026-12-30", timestamp="2026-12-30T15:33:00",
        cash=29000.0, options_buying_power=29000.0,
        portfolio_value=58500.0, positions_market_value=29500.0,
        num_positions=1, underlying_prices={"SPY": 580.0},
        positions=[PositionSnapshot(
            symbol="SPY280317C00450000", underlying="SPY",
            strike=450.0, expiry="2028-03-17",
            qty=2, avg_entry_price=125.0, current_price=147.5,
            market_value=29500.0, cost_basis=25000.0,
            unrealized_pl=4500.0, unrealized_plpc=0.18,
            days_remaining=443,
        )],
    ))
    return state


# ---- Summary computation ----

def test_summary_basic_fields():
    state = _populated_state()
    summary = ReportGenerator(state).compute_summary()

    assert summary.cash == 29000.0
    assert summary.portfolio_value == 58500.0
    assert summary.positions_market_value == 29500.0
    assert summary.num_open_positions == 1
    # total_invested = net external capital. With self-consistent fixture
    # there are no detected flows after the initial $50,500 deposit.
    assert summary.total_invested == 50500.0
    # total_deployed is the sum of allocation amounts (Q1 $20k + Q2 $11k = $31k)
    assert summary.total_deployed == 31000.0
    assert summary.num_external_flows == 0
    assert summary.num_trades == 5


def test_summary_realized_pnl():
    state = _populated_state()
    summary = ReportGenerator(state).compute_summary()
    # Two sells: +4000 and -500 = +3500 net
    assert summary.total_realized_pnl == 3500.0


def test_summary_unrealized_pnl():
    state = _populated_state()
    summary = ReportGenerator(state).compute_summary()
    # Single open position with $4500 unrealized P&L
    assert summary.total_unrealized_pnl == 4500.0


def test_summary_total_return():
    state = _populated_state()
    summary = ReportGenerator(state).compute_summary()
    # total_return_dollar = portfolio_value - net external contributions
    # = 58500 - 50500 = 8000
    assert summary.total_return_dollar == 8000.0
    # total_return_pct uses TWR. With one period (start $50,500 → end $58,500,
    # no external flows), TWR = 58500/50500 - 1 ≈ 15.84%
    assert summary.total_return_pct == pytest.approx(15.8416, abs=0.01)


def test_summary_spy_benchmark():
    state = _populated_state()
    summary = ReportGenerator(state).compute_summary()
    # SPY went from 540 to 580 → ~7.4%
    assert summary.spy_total_return_pct is not None
    assert 7.0 < summary.spy_total_return_pct < 8.0


def test_summary_handles_empty_state():
    state = BotState()
    summary = ReportGenerator(state).compute_summary()
    assert summary.num_trades == 0
    assert summary.num_open_positions == 0
    assert summary.total_invested == 0.0
    assert summary.spy_total_return_pct is None


# ---- PDF generation ----

def test_pdf_generated_to_file(tmp_path):
    state = _populated_state()
    out = tmp_path / "report.pdf"
    path = ReportGenerator(state).generate_pdf(out)

    assert path == out
    assert out.exists()
    # PDFs start with the magic header "%PDF"
    assert out.read_bytes()[:4] == b"%PDF"
    # Should be more than just a header
    assert out.stat().st_size > 5000


def test_pdf_works_with_empty_state(tmp_path):
    """Should not crash with no trades, snapshots, or positions."""
    state = BotState()
    out = tmp_path / "empty.pdf"
    ReportGenerator(state).generate_pdf(out)
    assert out.exists()
    assert out.read_bytes()[:4] == b"%PDF"


def test_pdf_handles_single_snapshot(tmp_path):
    """One snapshot is not enough for a chart, but PDF should still generate."""
    state = BotState()
    state.add_snapshot(DailySnapshot(
        date="2026-06-03", timestamp="2026-06-03T15:33:00",
        cash=50000.0, options_buying_power=50000.0,
        portfolio_value=50000.0, positions_market_value=0,
        num_positions=0, underlying_prices={"SPY": 550.0},
    ))
    out = tmp_path / "single.pdf"
    ReportGenerator(state).generate_pdf(out)
    assert out.exists()


def test_portfolio_chart_returns_png_bytes():
    state = _populated_state()
    png = ReportGenerator(state).generate_portfolio_chart()
    assert png is not None
    # PNG signature: 89 50 4E 47
    assert png[:4] == b"\x89PNG"


def test_portfolio_chart_returns_none_with_one_snapshot():
    state = BotState()
    state.add_snapshot(DailySnapshot(
        date="2026-06-03", timestamp="2026-06-03T15:33:00",
        cash=50000.0, options_buying_power=50000.0,
        portfolio_value=50000.0, positions_market_value=0,
        num_positions=0, underlying_prices={"SPY": 550.0},
    ))
    assert ReportGenerator(state).generate_portfolio_chart() is None


def test_pnl_chart_returns_png_bytes():
    state = _populated_state()
    png = ReportGenerator(state).generate_pnl_chart()
    assert png is not None
    assert png[:4] == b"\x89PNG"


# ---- Trade CSV ----

def test_export_trades_csv_writes_all_trades(tmp_path):
    state = _populated_state()
    out = tmp_path / "trades.csv"
    ReportGenerator(state).export_trades_csv(out)

    with open(out) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 5
    # Header check
    assert "realized_pnl" in rows[0]
    assert "underlying_price" in rows[0]
    # Sell trades have realized P&L
    sell_rows = [r for r in rows if r["action"] == "sell"]
    assert len(sell_rows) == 2
    assert all(r["realized_pnl"] != "" for r in sell_rows)
    # Buy trades have no realized P&L
    buy_rows = [r for r in rows if r["action"] == "buy"]
    assert all(r["realized_pnl"] == "" for r in buy_rows)


def test_export_trades_csv_empty(tmp_path):
    state = BotState()
    out = tmp_path / "empty.csv"
    ReportGenerator(state).export_trades_csv(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows == []


# ---- Tax CSV ----

def test_export_tax_csv_only_includes_sells(tmp_path):
    state = _populated_state()
    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out)

    with open(out) as f:
        rows = list(csv.DictReader(f))

    # 5 trades → 2 sells
    assert len(rows) == 2
    # Required tax columns present
    required = {"description", "date_acquired", "date_sold", "qty_contracts",
                "proceeds", "cost_basis", "gain_loss", "term"}
    assert required <= set(rows[0].keys())


def test_export_tax_csv_computes_term_short_vs_long(tmp_path):
    state = BotState()
    # Held 200 days = short-term
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:30:00", order_id="st-1",
        action="sell", intent="close",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=1, fill_price=120.0, total_value=12000.0,
        avg_entry_price=100.0, realized_pnl=2000.0, holding_days=200,
    ))
    # Held 400 days = long-term
    state.add_trade(TradeRecord(
        timestamp="2027-04-15T14:30:00", order_id="lt-1",
        action="sell", intent="close",
        symbol="SPY280317C00450000", underlying="SPY",
        strike=450.0, expiry="2028-03-17",
        qty=1, fill_price=130.0, total_value=13000.0,
        avg_entry_price=110.0, realized_pnl=2000.0, holding_days=400,
    ))

    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))

    by_id = {r["order_id"]: r for r in rows}
    assert by_id["st-1"]["term"] == "short"
    assert by_id["lt-1"]["term"] == "long"


def test_export_tax_csv_filters_by_year(tmp_path):
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:30:00", order_id="2026-1",
        action="sell", intent="close",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=1, fill_price=120.0, total_value=12000.0,
        avg_entry_price=100.0, realized_pnl=2000.0, holding_days=200,
    ))
    state.add_trade(TradeRecord(
        timestamp="2027-03-20T14:30:00", order_id="2027-1",
        action="sell", intent="close",
        symbol="SPY280317C00450000", underlying="SPY",
        strike=450.0, expiry="2028-03-17",
        qty=1, fill_price=130.0, total_value=13000.0,
        avg_entry_price=110.0, realized_pnl=2000.0, holding_days=300,
    ))

    out = tmp_path / "tax-2026.csv"
    ReportGenerator(state).export_tax_csv(out, year=2026)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["order_id"] == "2026-1"


def test_export_tax_csv_proceeds_and_cost_basis(tmp_path):
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:30:00", order_id="t-1",
        action="sell", intent="close",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=120.0, total_value=24000.0,
        avg_entry_price=100.0, realized_pnl=4000.0, holding_days=250,
    ))

    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))

    row = rows[0]
    # 2 contracts × 100 multiplier × $120 = $24,000 proceeds
    assert float(row["proceeds"]) == 24000.0
    # 2 × 100 × $100 = $20,000 cost basis
    assert float(row["cost_basis"]) == 20000.0
    assert float(row["gain_loss"]) == 4000.0
    assert row["qty_contracts"] == "2"


def test_export_tax_csv_date_acquired_derived_from_holding_days(tmp_path):
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2026-11-15T14:30:00", order_id="t-1",
        action="sell", intent="close",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=1, fill_price=120.0, total_value=12000.0,
        avg_entry_price=100.0, realized_pnl=2000.0, holding_days=100,
    ))

    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))

    # 2026-11-15 minus 100 days = 2026-08-07
    assert rows[0]["date_acquired"] == "2026-08-07"
    assert rows[0]["date_sold"] == "2026-11-15"
