"""Tests for the FX rate provider — both the Frankfurter HTTP source
(with mocked HTTP) and the FlatFXProvider, plus integration into
the tax CSV export.
"""
import csv
import json
from datetime import date
from io import BytesIO
from unittest.mock import patch

import pytest

from leaps_bot.fx import FlatFXProvider, FrankfurterFXProvider
from leaps_bot.models import TradeRecord
from leaps_bot.reporting import ReportGenerator
from leaps_bot.state import BotState


# ----------------------------------------------------------------------
# FlatFXProvider
# ----------------------------------------------------------------------

def test_flat_provider_returns_same_rate_regardless_of_date():
    p = FlatFXProvider(1.52)
    assert p.get_rate(date(2025, 1, 1)) == 1.52
    assert p.get_rate(date(2026, 6, 30)) == 1.52
    assert "1.52" in p.describe()


# ----------------------------------------------------------------------
# FrankfurterFXProvider — mocked HTTP
# ----------------------------------------------------------------------

def _mock_response(payload: dict):
    """Returns a context manager that mimics urlopen's return value."""
    body = json.dumps(payload).encode()

    class _CM:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *exc):
            return False

        def read(self_inner):
            return body

    return _CM()


def test_frankfurter_prefetches_range_and_caches(tmp_path):
    cache = tmp_path / "fx_cache.json"
    payload = {
        "amount": 1.0,
        "base": "USD",
        "rates": {
            "2026-03-15": {"AUD": 1.5234},
            "2026-03-16": {"AUD": 1.5240},
            "2026-03-17": {"AUD": 1.5251},
        },
    }
    with patch("leaps_bot.fx.urlopen", return_value=_mock_response(payload)) as mock_urlopen:
        p = FrankfurterFXProvider(cache_path=cache)
        p.prefetch_range(date(2026, 3, 15), date(2026, 3, 17))

    assert mock_urlopen.call_count == 1
    # Cache populated
    assert p.get_rate(date(2026, 3, 15)) == pytest.approx(1.5234)
    assert p.get_rate(date(2026, 3, 16)) == pytest.approx(1.5240)
    # Cache persisted to disk
    assert cache.exists()
    with open(cache) as f:
        on_disk = json.load(f)
    assert on_disk["2026-03-15"] == pytest.approx(1.5234)


def test_frankfurter_falls_back_to_preceding_business_day(tmp_path):
    """ECB doesn't publish on weekends. Asking for a Saturday should
    return Friday's rate (with a small offset)."""
    cache = tmp_path / "fx_cache.json"
    # Saturday March 14 has no rate; Friday March 13 does
    payload = {
        "rates": {
            "2026-03-13": {"AUD": 1.5200},  # Friday
            # 2026-03-14 (Sat) and 2026-03-15 (Sun) intentionally missing
            "2026-03-16": {"AUD": 1.5240},  # Monday
        },
    }
    with patch("leaps_bot.fx.urlopen", return_value=_mock_response(payload)):
        p = FrankfurterFXProvider(cache_path=cache)
        p.prefetch_range(date(2026, 3, 13), date(2026, 3, 16))

    # Saturday → Friday's rate
    assert p.get_rate(date(2026, 3, 14)) == pytest.approx(1.5200)
    # Sunday → Friday's rate (skipping Saturday which is also missing)
    assert p.get_rate(date(2026, 3, 15)) == pytest.approx(1.5200)
    # Monday → its own rate
    assert p.get_rate(date(2026, 3, 16)) == pytest.approx(1.5240)


def test_frankfurter_returns_none_when_far_outside_cached_range(tmp_path):
    cache = tmp_path / "fx_cache.json"
    payload = {"rates": {"2026-03-15": {"AUD": 1.5234}}}
    with patch("leaps_bot.fx.urlopen", return_value=_mock_response(payload)):
        p = FrankfurterFXProvider(cache_path=cache)
        p.prefetch_range(date(2026, 3, 15), date(2026, 3, 15))

    # 8 days before — outside the 7-day fallback window
    assert p.get_rate(date(2026, 3, 23)) is None


def test_frankfurter_handles_http_failure_gracefully(tmp_path, caplog):
    import logging
    from urllib.error import URLError

    cache = tmp_path / "fx_cache.json"
    with patch("leaps_bot.fx.urlopen", side_effect=URLError("connection refused")):
        p = FrankfurterFXProvider(cache_path=cache)
        with caplog.at_level(logging.WARNING):
            p.prefetch_range(date(2026, 3, 15), date(2026, 3, 17))

    # Failed lookups return None, don't crash
    assert p.get_rate(date(2026, 3, 15)) is None
    assert any("failed to fetch" in r.getMessage().lower() for r in caplog.records)


def test_frankfurter_loads_cache_on_init(tmp_path):
    """A second invocation of the bot reads cached rates from disk without
    re-fetching."""
    cache = tmp_path / "fx_cache.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "2026-03-15": 1.5234,
        "2026-03-16": 1.5240,
    }))

    # No HTTP mock — must NOT make a network call
    with patch("leaps_bot.fx.urlopen") as mock_urlopen:
        p = FrankfurterFXProvider(cache_path=cache)
        rate = p.get_rate(date(2026, 3, 15))

    assert rate == pytest.approx(1.5234)
    mock_urlopen.assert_not_called()


def test_frankfurter_prefetch_skips_when_range_already_loaded(tmp_path):
    cache = tmp_path / "fx_cache.json"
    payload = {"rates": {"2026-03-15": {"AUD": 1.5234}, "2026-03-16": {"AUD": 1.524}}}
    with patch("leaps_bot.fx.urlopen", return_value=_mock_response(payload)) as mock_urlopen:
        p = FrankfurterFXProvider(cache_path=cache)
        p.prefetch_range(date(2026, 3, 15), date(2026, 3, 16))
        # Calling again with a subset of the same range should not re-fetch
        p.prefetch_range(date(2026, 3, 15), date(2026, 3, 16))

    assert mock_urlopen.call_count == 1


# ----------------------------------------------------------------------
# Integration: per-trade FX in tax CSV
# ----------------------------------------------------------------------

def _state_with_two_trades_at_different_dates():
    state = BotState()
    state.add_trade(TradeRecord(
        timestamp="2025-08-15T14:30:00", order_id="t1",
        action="sell", intent="close",
        symbol="SPY270318C00440000", underlying="SPY",
        strike=440.0, expiry="2027-03-18",
        qty=2, fill_price=120.0, total_value=24000.0,
        avg_entry_price=100.0, realized_pnl=4000.0, holding_days=200,
    ))
    state.add_trade(TradeRecord(
        timestamp="2026-04-10T14:30:00", order_id="t2",
        action="sell", intent="close",
        symbol="SPY280317C00450000", underlying="SPY",
        strike=450.0, expiry="2028-03-17",
        qty=1, fill_price=130.0, total_value=13000.0,
        avg_entry_price=110.0, realized_pnl=2000.0, holding_days=300,
    ))
    return state


def test_export_uses_per_trade_fx_rate(tmp_path):
    """Each trade's AUD-converted figures should use the FX rate ON THAT
    TRADE'S DATE, not a flat rate."""
    cache = tmp_path / "fx_cache.json"
    payload = {
        "rates": {
            # Slightly different rates for each trade date
            "2025-08-15": {"AUD": 1.5500},  # rate when Aug trade closed
            "2026-04-10": {"AUD": 1.5800},  # rate when Apr trade closed
        },
    }
    state = _state_with_two_trades_at_different_dates()

    with patch("leaps_bot.fx.urlopen", return_value=_mock_response(payload)):
        provider = FrankfurterFXProvider(cache_path=cache)
        out = tmp_path / "tax.csv"
        ReportGenerator(state).export_tax_csv(out, fy=2026, fx_provider=provider)

    with open(out) as f:
        rows = {r["order_id"]: r for r in csv.DictReader(f)}

    # t1: $24k × 1.55 = $37,200 AUD proceeds
    assert float(rows["t1"]["proceeds_aud"]) == pytest.approx(37200.0)
    assert float(rows["t1"]["fx_rate"]) == pytest.approx(1.5500, abs=0.0001)

    # t2: $13k × 1.58 = $20,540 AUD proceeds — different rate proves it's per-trade
    assert float(rows["t2"]["proceeds_aud"]) == pytest.approx(20540.0)
    assert float(rows["t2"]["fx_rate"]) == pytest.approx(1.5800, abs=0.0001)


def test_export_per_trade_fx_uses_only_one_http_request(tmp_path):
    """Even with 50 trades, prefetch should make a single range request."""
    cache = tmp_path / "fx_cache.json"
    state = BotState()
    payload_rates = {}
    for day in range(1, 11):  # 10 trades across 10 different dates
        iso = f"2026-04-{day:02d}"
        state.add_trade(TradeRecord(
            timestamp=f"{iso}T14:30:00", order_id=f"t{day}",
            action="sell", intent="close",
            symbol="SPY270318C00440000", underlying="SPY",
            strike=440.0, expiry="2027-03-18",
            qty=1, fill_price=120.0, total_value=12000.0,
            avg_entry_price=100.0, realized_pnl=2000.0, holding_days=200,
        ))
        payload_rates[iso] = {"AUD": 1.55}

    with patch(
        "leaps_bot.fx.urlopen",
        return_value=_mock_response({"rates": payload_rates}),
    ) as mock_urlopen:
        provider = FrankfurterFXProvider(cache_path=cache)
        out = tmp_path / "tax.csv"
        ReportGenerator(state).export_tax_csv(out, fy=2026, fx_provider=provider)

    assert mock_urlopen.call_count == 1, (
        f"Expected exactly 1 HTTP request via prefetch, got {mock_urlopen.call_count}"
    )


def test_missing_fx_rate_leaves_aud_blank_with_warning(tmp_path, caplog):
    """If a trade's date has no FX rate (and no fallback within 7 days),
    AUD columns must be blank — never zero, never wrong."""
    import logging
    cache = tmp_path / "fx_cache.json"
    # Empty payload — no rates at all
    with patch("leaps_bot.fx.urlopen", return_value=_mock_response({"rates": {}})):
        provider = FrankfurterFXProvider(cache_path=cache)
        state = _state_with_two_trades_at_different_dates()
        out = tmp_path / "tax.csv"

        with caplog.at_level(logging.WARNING):
            ReportGenerator(state).export_tax_csv(out, fy=2026, fx_provider=provider)

    with open(out) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        assert r["fx_rate"] == "", f"FX rate should be blank for {r['order_id']}"
        assert r["proceeds_aud"] == ""
        assert r["cost_basis_aud"] == ""
        assert r["gain_loss_aud"] == ""

    assert any("no fx rate" in r.getMessage().lower() for r in caplog.records)


def test_aud_rate_and_fx_provider_mutex(tmp_path):
    state = BotState()
    out = tmp_path / "tax.csv"
    with pytest.raises(ValueError, match="not both"):
        ReportGenerator(state).export_tax_csv(
            out, fy=2026,
            aud_rate=1.5,
            fx_provider=FlatFXProvider(1.6),
        )


def test_per_trade_fx_logs_info_about_source(tmp_path, caplog):
    import logging
    cache = tmp_path / "fx_cache.json"
    payload = {"rates": {"2026-04-10": {"AUD": 1.55}}}
    state = _state_with_two_trades_at_different_dates()

    with patch("leaps_bot.fx.urlopen", return_value=_mock_response(payload)):
        provider = FrankfurterFXProvider(cache_path=cache)
        out = tmp_path / "tax.csv"
        with caplog.at_level(logging.INFO):
            ReportGenerator(state).export_tax_csv(out, fy=2026, fx_provider=provider)

    info_messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Frankfurter" in info_messages or "ECB" in info_messages
    # Should also mention the RBA caveat
    assert "RBA" in info_messages
