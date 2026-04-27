from __future__ import annotations

import logging
import math
from urllib.error import URLError
from urllib.request import Request, urlopen

from scipy.stats import norm

from leaps_bot.config import PricingConfig

logger = logging.getLogger(__name__)


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """European call price. S=spot, K=strike, T=years, r=risk-free, q=dividend yield, sigma=IV."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def intrinsic_value(underlying_price: float, strike: float) -> float:
    return max(0.0, underlying_price - strike)


class RateFetcher:
    """Fetches risk-free rate and dividend yield with config fallbacks."""

    def __init__(self, config: PricingConfig):
        self._config = config
        self._risk_free_rate: float | None = None
        self._dividend_yield: float | None = None

    @property
    def risk_free_rate(self) -> float:
        if self._risk_free_rate is not None:
            return self._risk_free_rate
        return self._config.risk_free_rate

    @property
    def dividend_yield(self) -> float:
        if self._dividend_yield is not None:
            return self._dividend_yield
        return self._config.dividend_yield

    def fetch_rates(self) -> None:
        self._try_fetch_treasury_rate()
        self._try_fetch_dividend_yield()

    def _try_fetch_treasury_rate(self) -> None:
        try:
            from datetime import date as _date, timedelta

            # FRED (Federal Reserve Economic Data) CSV endpoint — free, no API
            # key, returns the 1-year constant maturity treasury rate (DGS1).
            start = (_date.today() - timedelta(days=10)).isoformat()
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1&cosd={start}"
            req = Request(url, headers={"User-Agent": "leaps-bot/1.0"})
            response = urlopen(req, timeout=10)  # noqa: S310
            body = response.read().decode("utf-8")

            # Parse last non-header line with a valid rate
            rate = None
            for line in reversed(body.strip().split("\n")):
                parts = line.split(",")
                if len(parts) == 2:
                    try:
                        rate = float(parts[1]) / 100.0
                        break
                    except ValueError:
                        continue

            if rate is not None and 0 < rate < 0.20:
                self._risk_free_rate = rate
                logger.info("Fetched 1Y treasury rate from FRED: %.4f", rate)
            else:
                logger.warning("FRED returned unexpected rate value: %s", rate)
        except Exception as e:
            logger.warning("Failed to fetch treasury rate, using config default: %s", e)

    def _try_fetch_dividend_yield(self) -> None:
        # SPY dividend yield is ~1.2-1.4%. Fetching live data requires a paid API
        # or scraping. For now we rely on the config default and log a note.
        logger.info(
            "Using config dividend yield: %.4f (auto-fetch not available on free tier)",
            self._config.dividend_yield,
        )
