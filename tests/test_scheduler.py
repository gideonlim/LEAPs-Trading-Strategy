from datetime import date, datetime
from unittest.mock import MagicMock, patch

from leaps_bot.config import AppConfig, AllocationConfig, SafetyConfig, StrategyConfig
from leaps_bot.pricing import RateFetcher
from leaps_bot.scheduler import DailyScheduler
from leaps_bot.state import BotState
from tests.conftest import FakeAccount, FakeClock


def _make_scheduler(is_open=True, minutes_since=90, dry_run=True):
    config = AppConfig(
        dry_run=dry_run,
        strategy=StrategyConfig(),
        allocation=AllocationConfig(quarterly_months=[3, 6, 9, 12]),
        safety=SafetyConfig(no_trade_minutes_after_open=60),
    )
    client = MagicMock()
    client.is_market_open.return_value = is_open
    client.minutes_since_open.return_value = minutes_since
    client.get_account.return_value = FakeAccount()
    client.get_cash_available.return_value = 50000.0
    client.get_buying_power.return_value = 50000.0
    client.get_option_positions.return_value = []
    client.get_open_orders.return_value = []

    now = datetime(2026, 6, 15, 11, 0, 0)
    client.get_clock.return_value = FakeClock(
        is_open=is_open, timestamp=now,
        next_open=now.replace(hour=9, minute=30),
        next_close=now.replace(hour=16, minute=0),
    )

    state = BotState()
    rate_fetcher = MagicMock(spec=RateFetcher)
    rate_fetcher.risk_free_rate = 0.045
    rate_fetcher.dividend_yield = 0.013

    scheduler = DailyScheduler(config, client, state, rate_fetcher)
    return scheduler, state


def test_skip_when_market_closed():
    scheduler, state = _make_scheduler(is_open=False)
    summary = scheduler.run()
    assert summary["skipped"]
    assert "closed" in summary["actions"][0]["reason"].lower()


def test_skip_first_hour():
    scheduler, state = _make_scheduler(is_open=True, minutes_since=30)
    summary = scheduler.run()
    assert summary["skipped"]
    assert "first-hour" in summary["actions"][0]["reason"].lower()


def test_run_with_no_positions_no_allocation():
    with patch("leaps_bot.allocator.date") as mock_date:
        mock_date.today.return_value = date(2026, 5, 15)  # Not a quarterly month
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        scheduler, state = _make_scheduler()
        summary = scheduler.run()
        assert not summary["skipped"]
        assert state.last_run is not None


def test_idempotent_run():
    scheduler, state = _make_scheduler()
    state.last_trade_date = date.today().isoformat()

    summary = scheduler.run()
    assert not summary["skipped"]
    # Should still run (monitoring) but not trigger new trades
