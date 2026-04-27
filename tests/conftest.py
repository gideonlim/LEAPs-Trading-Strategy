from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from leaps_bot.config import (
    AllocationConfig,
    AppConfig,
    DataConfig,
    PricingConfig,
    SafetyConfig,
    StrategyConfig,
)
from leaps_bot.state import BotState


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(
        paper_trading=True,
        dry_run=True,
        log_level="DEBUG",
        strategy=StrategyConfig(),
        pricing=PricingConfig(),
        allocation=AllocationConfig(),
        safety=SafetyConfig(),
        data=DataConfig(),
        api_key="test-key",
        secret_key="test-secret",
    )


@pytest.fixture
def state() -> BotState:
    return BotState()


@dataclass
class FakeQuote:
    bid_price: float
    ask_price: float
    bid_size: int = 10
    ask_size: int = 10


@dataclass
class FakeGreeks:
    delta: float
    gamma: float = 0.01
    theta: float = -0.05
    vega: float = 0.1
    rho: float = 0.05


@dataclass
class FakeSnapshot:
    latest_quote: FakeQuote | None
    greeks: FakeGreeks | None
    implied_volatility: float = 0.20


@dataclass
class FakeContract:
    symbol: str
    strike_price: float
    expiration_date: date
    open_interest: int = 500
    status: str = "active"


@dataclass
class FakePosition:
    symbol: str
    qty: str
    market_value: str = "15000"
    asset_class: str = "us_option"


@dataclass
class FakeClock:
    is_open: bool
    timestamp: datetime
    next_open: datetime
    next_close: datetime


@dataclass
class FakeAccount:
    cash: str = "50000.00"
    buying_power: str = "50000.00"
    trading_blocked: bool = False


@dataclass
class FakeOrder:
    id: str = "order-123"
    status: str = "filled"
    filled_avg_price: str = "120.50"


@pytest.fixture
def mock_client(config):
    client = MagicMock()
    client._config = config

    client.get_account.return_value = FakeAccount()
    client.get_cash_available.return_value = 50000.0
    client.get_buying_power.return_value = 50000.0
    client.get_underlying_price.return_value = 550.0
    client.is_market_open.return_value = True
    client.minutes_since_open.return_value = 90.0
    client.get_option_positions.return_value = []
    client.get_open_orders.return_value = []

    now = datetime(2026, 6, 15, 11, 0, 0)
    client.get_clock.return_value = FakeClock(
        is_open=True,
        timestamp=now,
        next_open=now.replace(hour=9, minute=30),
        next_close=now.replace(hour=16, minute=0),
    )

    return client
