from datetime import date
from unittest.mock import MagicMock, patch

from leaps_bot.allocator import QuarterlyAllocator
from leaps_bot.config import AllocationConfig, AppConfig, StrategyConfig
from leaps_bot.models import AllocationRecord, ContractCandidate
from leaps_bot.state import BotState


def _make_allocator(month=6, cash=50000.0, options_bp=None, already_allocated=False, window_days=7):
    config = AppConfig(
        dry_run=True,
        strategy=StrategyConfig(order_type="limit"),
        allocation=AllocationConfig(
            quarterly_months=[3, 6, 9, 12],
            max_cash_deploy_pct=0.90,
            min_cash_reserve=500.0,
            allocation_window_days=window_days,
        ),
    )
    state = BotState()
    if already_allocated:
        state.record_allocation(AllocationRecord(
            quarter=f"2026-Q{(month - 1) // 3 + 1}",
            allocated_date=f"2026-{month:02d}-01",
            amount=10000.0,
        ))

    client = MagicMock()
    client.get_cash_available.return_value = cash
    client.get_options_buying_power.return_value = options_bp if options_bp is not None else cash
    finder = MagicMock()
    executor = MagicMock()

    allocator = QuarterlyAllocator(client, config, state, finder, executor)
    return allocator, state, finder, executor


def test_should_allocate_in_quarterly_month_within_window():
    with patch("leaps_bot.allocator.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 3)  # within 7-day window
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        allocator, *_ = _make_allocator(month=6)
        assert allocator.should_allocate_today()


def test_should_not_allocate_in_non_quarterly_month():
    with patch("leaps_bot.allocator.date") as mock_date:
        mock_date.today.return_value = date(2026, 5, 3)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        allocator, *_ = _make_allocator(month=5)
        assert not allocator.should_allocate_today()


def test_should_not_allocate_if_already_done():
    with patch("leaps_bot.allocator.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 3)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        allocator, *_ = _make_allocator(month=6, already_allocated=True)
        assert not allocator.should_allocate_today()


def test_should_not_allocate_outside_window():
    with patch("leaps_bot.allocator.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 15)  # past 7-day window
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        allocator, *_ = _make_allocator(month=6, window_days=7)
        assert not allocator.should_allocate_today()


def test_calculate_deployment():
    allocator, *_ = _make_allocator(cash=50000.0)
    deploy = allocator.calculate_deployment()
    # min(50000 * 0.90, 50000 - 500) = min(45000, 49500) = 45000
    assert deploy == 45000.0


def test_calculate_deployment_low_cash():
    allocator, *_ = _make_allocator(cash=400.0)
    deploy = allocator.calculate_deployment()
    # min(400 * 0.90, 400 - 500) = min(360, -100) → max(0, -100) = 0
    assert deploy == 0.0


def test_allocate_dry_run():
    with patch("leaps_bot.allocator.date") as mock_date:
        mock_date.today.return_value = date(2026, 6, 3)
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

        allocator, state, finder, executor = _make_allocator(cash=50000.0)
        candidate = ContractCandidate(
            symbol="SPY270618C00440000", underlying="SPY",
            strike=440.0, expiry=date(2027, 6, 18),
            bid=119.0, ask=121.0, mid=120.0,
            delta=0.88, iv=0.20, open_interest=500,
            theoretical_price=120.5,
        )
        finder.find_best_leaps_call.return_value = candidate
        finder.calculate_limit_price.return_value = 120.50

        from leaps_bot.models import OrderResult
        executor.execute_buy.return_value = OrderResult(
            success=True, order_id=None, symbol="SPY270618C00440000",
            side="buy", qty=3, price=120.50,
            message="dry-run", dry_run=True,
        )

        results = allocator.allocate()
        assert len(results) == 1
        assert results[0].success
        executor.execute_buy.assert_called_once()
        # 45000 / (121 * 100) = 3.71 → 3 contracts
        call_args = executor.execute_buy.call_args
        assert call_args[0][1] == 3  # qty

        # Allocator must NOT record allocation in state — that happens on fill
        # confirmation in the scheduler. Recording on submission would block
        # next quarter's allocation if this order expired unfilled.
        assert len(state.allocations) == 0
