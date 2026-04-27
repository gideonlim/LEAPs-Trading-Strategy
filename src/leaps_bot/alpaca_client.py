from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionLatestQuoteRequest, OptionSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    AssetClass,
    ContractType,
    OrderSide,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

from leaps_bot.config import AppConfig

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class AlpacaClient:
    def __init__(self, config: AppConfig):
        self._config = config
        self._trading = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper_trading,
        )
        self._data = OptionHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )

    def _retry_read(self, fn, *args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning("Retry %d/%d after error: %s (waiting %ds)", attempt + 1, MAX_RETRIES, e, delay)
                time.sleep(delay)

    # -- Account --

    def get_account(self):
        return self._retry_read(self._trading.get_account)

    def get_cash_available(self) -> float:
        acct = self.get_account()
        return _safe_float(acct.cash)

    def get_buying_power(self) -> float:
        acct = self.get_account()
        return _safe_float(acct.buying_power)

    def get_options_buying_power(self) -> float:
        """Returns options buying power if available, else falls back to regular buying power."""
        acct = self.get_account()
        opt_bp = getattr(acct, "options_buying_power", None)
        if opt_bp is not None:
            return _safe_float(opt_bp)
        return _safe_float(acct.buying_power)

    # -- Market state --

    def get_clock(self):
        return self._retry_read(self._trading.get_clock)

    def is_market_open(self) -> bool:
        return self.get_clock().is_open

    def minutes_since_open(self) -> float:
        clock = self.get_clock()
        if not clock.is_open:
            return -1.0
        now = clock.timestamp
        open_time = clock.next_open if not clock.is_open else clock.timestamp
        # Clock provides next_open/next_close; when open, elapsed = now - (next_close - 6.5h market day)
        # Simpler: use the actual open time from today
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        elapsed = (now - market_open).total_seconds() / 60.0
        return max(0.0, elapsed)

    # -- Positions --

    def get_all_positions(self):
        return self._retry_read(self._trading.get_all_positions)

    def get_option_positions(self) -> list:
        positions = self.get_all_positions()
        return [p for p in positions if getattr(p, "asset_class", None) == AssetClass.US_OPTION]

    # -- Contract discovery --

    def find_option_contracts(
        self,
        underlying: str,
        expiry_gte: date,
        expiry_lte: date,
        strike_lte: float | None = None,
        strike_gte: float | None = None,
    ) -> list:
        """Paginated search for option contracts. Always sets expiry filters explicitly."""
        all_contracts = []
        page_token = None

        while True:
            params = {
                "underlying_symbols": [underlying],
                "type": ContractType.CALL,
                "expiration_date_gte": expiry_gte.isoformat(),
                "expiration_date_lte": expiry_lte.isoformat(),
                "status": "active",
                "limit": 1000,
            }
            if strike_lte is not None:
                params["strike_price_lte"] = str(strike_lte)
            if strike_gte is not None:
                params["strike_price_gte"] = str(strike_gte)
            if page_token:
                params["page_token"] = page_token

            request = GetOptionContractsRequest(**params)
            response = self._retry_read(self._trading.get_option_contracts, request)

            contracts = response.option_contracts if response.option_contracts else []
            all_contracts.extend(contracts)

            page_token = getattr(response, "next_page_token", None)
            if not page_token:
                break

        logger.info("Found %d option contracts for %s (%s to %s)", len(all_contracts), underlying, expiry_gte, expiry_lte)
        return all_contracts

    # -- Option market data --

    def get_option_chain(
        self,
        underlying: str,
        expiry_gte: date | None = None,
        expiry_lte: date | None = None,
        strike_lte: float | None = None,
        strike_gte: float | None = None,
    ) -> dict:
        """Get option chain snapshots with quotes and Greeks."""
        params: dict[str, Any] = {
            "underlying_symbol": underlying,
            "feed": self._config.data.feed,
            "type": "call",
        }
        if expiry_gte:
            params["expiration_date_gte"] = expiry_gte.isoformat()
        if expiry_lte:
            params["expiration_date_lte"] = expiry_lte.isoformat()
        if strike_lte is not None:
            params["strike_price_lte"] = str(strike_lte)
        if strike_gte is not None:
            params["strike_price_gte"] = str(strike_gte)

        request = OptionChainRequest(**params)
        return self._retry_read(self._data.get_option_chain, request)

    def get_option_snapshot(self, symbol: str) -> Any:
        request = OptionSnapshotRequest(symbol_or_symbols=symbol, feed=self._config.data.feed)
        return self._retry_read(self._data.get_option_snapshot, request)

    def get_option_latest_quote(self, symbol: str) -> Any:
        request = OptionLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._config.data.feed)
        return self._retry_read(self._data.get_option_latest_quote, request)

    # -- Stock price --

    def get_underlying_price(self, symbol: str) -> float:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        stock_client = StockHistoricalDataClient(
            api_key=self._config.api_key,
            secret_key=self._config.secret_key,
        )
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trades = self._retry_read(stock_client.get_stock_latest_trade, request)
        trade = trades.get(symbol) if isinstance(trades, dict) else trades
        return float(trade.price)

    # -- Orders (no retries on submission) --

    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
    ):
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        logger.info("Submitting MARKET order: %s %d %s", side.value, qty, symbol)
        return self._trading.submit_order(request)

    def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        limit_price: float,
    ):
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=round(limit_price, 2),
            time_in_force=TimeInForce.DAY,
        )
        logger.info("Submitting LIMIT order: %s %d %s @ $%.2f", side.value, qty, symbol, limit_price)
        return self._trading.submit_order(request)

    def get_order(self, order_id: str):
        return self._retry_read(self._trading.get_order_by_id, order_id)

    def get_open_orders(self):
        return self._retry_read(self._trading.get_orders)

    def close_position(self, symbol: str):
        logger.info("Closing position: %s", symbol)
        return self._trading.close_position(symbol)
