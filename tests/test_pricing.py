from leaps_bot.pricing import black_scholes_call, intrinsic_value, RateFetcher
from leaps_bot.config import PricingConfig


def test_intrinsic_value_itm():
    assert intrinsic_value(550.0, 440.0) == 110.0


def test_intrinsic_value_otm():
    assert intrinsic_value(550.0, 600.0) == 0.0


def test_black_scholes_deep_itm():
    # Deep ITM call: S=550, K=440, 1 year, 4.5% rate, 1.3% div, 20% vol
    price = black_scholes_call(S=550.0, K=440.0, T=1.0, r=0.045, q=0.013, sigma=0.20)
    intrinsic = 110.0
    # Deep ITM should be worth more than intrinsic (time value, rates)
    assert price > intrinsic
    # But not absurdly more — extrinsic should be reasonable for deep ITM
    assert price < intrinsic * 1.25


def test_black_scholes_atm():
    # ATM call has significant time value
    price = black_scholes_call(S=550.0, K=550.0, T=1.0, r=0.045, q=0.013, sigma=0.20)
    assert price > 30  # ATM 1-year call on $550 stock should have substantial value
    assert price < 100


def test_black_scholes_zero_time():
    # At expiry, should be intrinsic
    price = black_scholes_call(S=550.0, K=440.0, T=0.0, r=0.045, q=0.013, sigma=0.20)
    assert price == 110.0


def test_black_scholes_zero_vol():
    # With zero vol, should converge to discounted intrinsic
    price = black_scholes_call(S=550.0, K=440.0, T=0.0, r=0.045, q=0.013, sigma=0.0)
    assert price == 110.0


def test_rate_fetcher_defaults():
    config = PricingConfig(risk_free_rate=0.05, dividend_yield=0.015)
    fetcher = RateFetcher(config)
    assert fetcher.risk_free_rate == 0.05
    assert fetcher.dividend_yield == 0.015
