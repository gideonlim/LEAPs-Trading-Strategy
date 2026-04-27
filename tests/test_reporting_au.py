"""Tests for Australian tax reporting features:
- FY filter (Jul 1 to Jun 30)
- cgt_discount_eligible column with 12-month threshold
- Optional flat AUD/USD conversion adding _aud columns
"""
import csv

import pytest

from leaps_bot.models import TradeRecord
from leaps_bot.reporting import ReportGenerator
from leaps_bot.state import BotState


def _state_with_trades(trades: list[TradeRecord]) -> BotState:
    state = BotState()
    for t in trades:
        state.add_trade(t)
    return state


# ---- AU FY filtering ----

def test_fy_2026_includes_july_2025_through_june_2026(tmp_path):
    """AU FY 2026 = Jul 1, 2025 → Jun 30, 2026."""
    state = _state_with_trades([
        # Out of range (FY 2025): Jun 30 2025
        TradeRecord(
            timestamp="2025-06-30T14:30:00", order_id="fy2025",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=100.0, realized_pnl=2000.0, holding_days=200,
        ),
        # In range: first day of FY 2026
        TradeRecord(
            timestamp="2025-07-01T14:30:00", order_id="fy2026-start",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=125.0, total_value=12500.0,
            avg_entry_price=100.0, realized_pnl=2500.0, holding_days=210,
        ),
        # In range: last day of FY 2026
        TradeRecord(
            timestamp="2026-06-30T14:30:00", order_id="fy2026-end",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=130.0, total_value=13000.0,
            avg_entry_price=100.0, realized_pnl=3000.0, holding_days=300,
        ),
        # Out of range (FY 2027): Jul 1 2026
        TradeRecord(
            timestamp="2026-07-01T14:30:00", order_id="fy2027",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=135.0, total_value=13500.0,
            avg_entry_price=100.0, realized_pnl=3500.0, holding_days=400,
        ),
    ])

    out = tmp_path / "tax-fy2026.csv"
    ReportGenerator(state).export_tax_csv(out, fy=2026)

    with open(out) as f:
        rows = list(csv.DictReader(f))
    order_ids = {r["order_id"] for r in rows}
    assert order_ids == {"fy2026-start", "fy2026-end"}


def test_fy_uses_cgt_discount_eligible_column(tmp_path):
    state = _state_with_trades([
        TradeRecord(
            timestamp="2025-12-15T14:30:00", order_id="t1",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=100.0, realized_pnl=2000.0, holding_days=200,
        ),
    ])

    out = tmp_path / "tax-fy2026.csv"
    ReportGenerator(state).export_tax_csv(out, fy=2026)

    with open(out) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    assert "cgt_discount_eligible" in fields
    assert "term" not in fields, "Should not output 'term' in AU mode"
    # 200 days < 365 → not eligible
    assert rows[0]["cgt_discount_eligible"] == "no"


def test_cgt_discount_eligible_yes_when_held_over_12_months(tmp_path):
    state = _state_with_trades([
        TradeRecord(
            timestamp="2026-01-15T14:30:00", order_id="long-hold",
            action="sell", intent="close",
            symbol="SPY280317C00440000", underlying="SPY",
            strike=440.0, expiry="2028-03-17",
            qty=1, fill_price=140.0, total_value=14000.0,
            avg_entry_price=100.0, realized_pnl=4000.0, holding_days=400,
        ),
        TradeRecord(
            timestamp="2026-02-15T14:30:00", order_id="short-hold",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=100.0, realized_pnl=2000.0, holding_days=350,
        ),
    ])

    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out, fy=2026)

    with open(out) as f:
        rows = {r["order_id"]: r for r in csv.DictReader(f)}

    assert rows["long-hold"]["cgt_discount_eligible"] == "yes"
    assert rows["short-hold"]["cgt_discount_eligible"] == "no"


def test_cgt_discount_at_exactly_365_days_is_not_eligible(tmp_path):
    """Boundary: AU 50% CGT discount requires holding for MORE THAN 12 months,
    so exactly 365 days held is NOT eligible."""
    state = _state_with_trades([
        TradeRecord(
            timestamp="2026-03-01T14:30:00", order_id="t-365",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=100.0, realized_pnl=2000.0, holding_days=365,
        ),
        TradeRecord(
            timestamp="2026-03-02T14:30:00", order_id="t-366",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=100.0, realized_pnl=2000.0, holding_days=366,
        ),
    ])
    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out, fy=2026)
    with open(out) as f:
        rows = {r["order_id"]: r for r in csv.DictReader(f)}
    assert rows["t-365"]["cgt_discount_eligible"] == "no"
    assert rows["t-366"]["cgt_discount_eligible"] == "yes"


# ---- US year filter still produces 'term' column ----

def test_year_mode_produces_term_column(tmp_path):
    state = _state_with_trades([
        TradeRecord(
            timestamp="2026-06-15T14:30:00", order_id="t1",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=100.0, realized_pnl=2000.0, holding_days=200,
        ),
    ])

    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out, year=2026)

    with open(out) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    assert "term" in fields
    assert "cgt_discount_eligible" not in fields
    assert rows[0]["term"] == "short"


# ---- year and fy mutual exclusion ----

def test_year_and_fy_together_raises(tmp_path):
    state = BotState()
    out = tmp_path / "tax.csv"
    with pytest.raises(ValueError, match="not both"):
        ReportGenerator(state).export_tax_csv(out, year=2026, fy=2026)


# ---- AUD conversion ----

def test_aud_rate_adds_aud_columns(tmp_path):
    state = _state_with_trades([
        TradeRecord(
            timestamp="2026-03-15T14:30:00", order_id="t1",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=2, fill_price=120.0, total_value=24000.0,
            avg_entry_price=100.0, realized_pnl=4000.0, holding_days=200,
        ),
    ])

    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out, fy=2026, aud_rate=1.50)

    with open(out) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    # New columns present
    assert "proceeds_aud" in fields
    assert "cost_basis_aud" in fields
    assert "gain_loss_aud" in fields
    # Original USD columns still present
    assert "proceeds" in fields
    assert "cost_basis" in fields
    assert "gain_loss" in fields

    row = rows[0]
    # USD: 2 × $120 × 100 = $24,000 proceeds, $20,000 basis, $4,000 gain
    assert float(row["proceeds"]) == 24000.0
    assert float(row["cost_basis"]) == 20000.0
    assert float(row["gain_loss"]) == 4000.0
    # AUD: × 1.50
    assert float(row["proceeds_aud"]) == 36000.0
    assert float(row["cost_basis_aud"]) == 30000.0
    assert float(row["gain_loss_aud"]) == 6000.0


def test_aud_rate_with_missing_basis_leaves_aud_blank(tmp_path):
    """When cost basis is unknown, AUD-converted basis must also be blank
    (not 0, not multiplied), to avoid fabricated tax figures."""
    state = _state_with_trades([
        TradeRecord(
            timestamp="2026-03-15T14:30:00", order_id="orphan",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=None, realized_pnl=None, holding_days=None,
        ),
    ])

    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out, fy=2026, aud_rate=1.50)

    with open(out) as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    # Proceeds in USD and AUD are computable
    assert float(row["proceeds"]) == 12000.0
    assert float(row["proceeds_aud"]) == 18000.0
    # Cost basis & gain/loss must be blank in BOTH currencies
    assert row["cost_basis"] == ""
    assert row["cost_basis_aud"] == ""
    assert row["gain_loss"] == ""
    assert row["gain_loss_aud"] == ""


def test_aud_rate_warning_logged(tmp_path, caplog):
    """User must be warned that flat-rate AUD conversion is not strictly
    ATO-compliant — they should verify with their accountant."""
    import logging
    state = _state_with_trades([
        TradeRecord(
            timestamp="2026-03-15T14:30:00", order_id="t1",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=100.0, realized_pnl=2000.0, holding_days=200,
        ),
    ])

    out = tmp_path / "tax.csv"
    with caplog.at_level(logging.WARNING):
        ReportGenerator(state).export_tax_csv(out, fy=2026, aud_rate=1.52)

    assert any("flat rate" in r.message.lower() and "rba" in r.message.lower() for r in caplog.records)


# ---- Description / column layout when no aud_rate ----

def test_no_aud_rate_means_no_aud_columns(tmp_path):
    state = _state_with_trades([
        TradeRecord(
            timestamp="2026-03-15T14:30:00", order_id="t1",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=100.0, realized_pnl=2000.0, holding_days=200,
        ),
    ])
    out = tmp_path / "tax.csv"
    ReportGenerator(state).export_tax_csv(out, fy=2026)
    with open(out) as f:
        fields = csv.DictReader(f).fieldnames
    assert "proceeds_aud" not in fields
    assert "cost_basis_aud" not in fields
    assert "gain_loss_aud" not in fields
