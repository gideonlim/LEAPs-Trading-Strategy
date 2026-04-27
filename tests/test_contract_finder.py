from datetime import date, timedelta
from unittest.mock import MagicMock

from leaps_bot.config import AppConfig, PricingConfig, SafetyConfig, StrategyConfig
from leaps_bot.contract_finder import ContractFinder
from leaps_bot.pricing import RateFetcher
from tests.conftest import FakeContract, FakeGreeks, FakeQuote, FakeSnapshot


def _make_finder(spot=550.0, contracts=None, chain=None):
    config = AppConfig(
        strategy=StrategyConfig(
            itm_depth_pct=0.20,
            min_expiry_months=12,
            max_expiry_months=18,
            limit_offset_pct=0.02,
        ),
        pricing=PricingConfig(
            max_extrinsic_pct=0.25,
            price_divergence_warn_pct=0.10,
        ),
        safety=SafetyConfig(
            min_delta=0.80,
            min_open_interest=100,
            max_bid_ask_spread_pct=0.10,
        ),
    )
    client = MagicMock()
    client.get_underlying_price.return_value = spot
    client.get_buying_power.return_value = 50000.0
    client.find_option_contracts.return_value = contracts or []
    client.get_option_chain.return_value = chain or {}

    rate_fetcher = RateFetcher(config.pricing)
    return ContractFinder(client, config, rate_fetcher)


def test_expiry_window():
    finder = _make_finder()
    gte, lte = finder._expiry_window()
    today = date.today()
    assert gte > today + timedelta(days=350)
    assert lte > gte


def test_find_best_filters_low_delta():
    expiry = date.today() + timedelta(days=400)
    symbol = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    contracts = [FakeContract(symbol=symbol, strike_price=440.0, expiration_date=expiry)]
    chain = {
        symbol: FakeSnapshot(
            latest_quote=FakeQuote(bid_price=119.0, ask_price=121.0),
            greeks=FakeGreeks(delta=0.50),  # Too low
            implied_volatility=0.20,
        ),
    }
    finder = _make_finder(contracts=contracts, chain=chain)
    result = finder.find_best_leaps_call("SPY")
    assert result is None  # Filtered out by delta < 0.80


def test_find_best_selects_good_candidate():
    expiry = date.today() + timedelta(days=400)
    symbol = f"SPY{expiry.strftime('%y%m%d')}C00440000"

    contracts = [FakeContract(symbol=symbol, strike_price=440.0, expiration_date=expiry, open_interest=500)]
    chain = {
        symbol: FakeSnapshot(
            latest_quote=FakeQuote(bid_price=119.0, ask_price=121.0),
            greeks=FakeGreeks(delta=0.88),
            implied_volatility=0.20,
        ),
    }
    finder = _make_finder(contracts=contracts, chain=chain)
    result = finder.find_best_leaps_call("SPY")
    assert result is not None
    assert result.symbol == symbol
    assert result.strike == 440.0
    assert result.delta == 0.88


def test_calculate_limit_price_buy():
    from leaps_bot.models import ContractCandidate

    finder = _make_finder()
    candidate = ContractCandidate(
        symbol="SPY270618C00440000", underlying="SPY",
        strike=440.0, expiry=date(2027, 6, 18),
        bid=119.0, ask=121.0, mid=120.0,
        delta=0.88, iv=0.20, open_interest=500,
        theoretical_price=118.0,
    )
    # Buy: mid + (ask - mid) * 0.02 = 120 + 1 * 0.02 = 120.02
    # (test config uses default limit_offset_pct=0.02)
    price = finder.calculate_limit_price(candidate, "buy")
    assert price == 120.02


def test_calculate_limit_price_sell():
    from leaps_bot.models import ContractCandidate

    finder = _make_finder()
    candidate = ContractCandidate(
        symbol="SPY270618C00440000", underlying="SPY",
        strike=440.0, expiry=date(2027, 6, 18),
        bid=119.0, ask=121.0, mid=120.0,
        delta=0.88, iv=0.20, open_interest=500,
        theoretical_price=118.0,
    )
    # Sell: mid - (mid - bid) * 0.02 = 120 - 1 * 0.02 = 119.98
    price = finder.calculate_limit_price(candidate, "sell")
    assert price == 119.98
