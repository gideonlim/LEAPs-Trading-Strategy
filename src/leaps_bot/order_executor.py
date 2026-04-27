from __future__ import annotations

import logging

from alpaca.trading.enums import OrderSide, TimeInForce

from leaps_bot.alpaca_client import AlpacaClient
from leaps_bot.config import AppConfig
from leaps_bot.models import OrderResult

logger = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(self, client: AlpacaClient, config: AppConfig):
        self._client = client
        self._config = config

    def execute_buy(
        self,
        symbol: str,
        qty: int,
        limit_price: float | None = None,
    ) -> OrderResult:
        return self._execute(symbol, qty, OrderSide.BUY, "buy", limit_price)

    def execute_sell(
        self,
        symbol: str,
        qty: int,
        limit_price: float | None = None,
    ) -> OrderResult:
        return self._execute(symbol, qty, OrderSide.SELL, "sell", limit_price)

    def _execute(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        side_str: str,
        limit_price: float | None,
    ) -> OrderResult:
        use_limit = self._config.strategy.order_type == "limit" and limit_price is not None
        price_str = f"@ ${limit_price:.2f} LIMIT" if use_limit else "@ MARKET"

        if self._config.dry_run:
            logger.info("[DRY-RUN] WOULD %s %dx %s %s", side_str.upper(), qty, symbol, price_str)
            return OrderResult(
                success=True,
                order_id=None,
                symbol=symbol,
                side=side_str,
                qty=qty,
                price=limit_price,
                message=f"Dry-run: would {side_str} {qty}x {symbol} {price_str}",
                dry_run=True,
            )

        try:
            if use_limit:
                order = self._client.submit_limit_order(symbol, qty, side, limit_price)
            else:
                order = self._client.submit_market_order(symbol, qty, side)

            order_id = str(order.id)
            logger.info("Order submitted: %s %dx %s %s (id=%s)", side_str.upper(), qty, symbol, price_str, order_id)
            return OrderResult(
                success=True,
                order_id=order_id,
                symbol=symbol,
                side=side_str,
                qty=qty,
                price=limit_price,
                message=f"Order {order_id}: {side_str} {qty}x {symbol} {price_str}",
            )
        except Exception as e:
            logger.error("Order FAILED: %s %dx %s %s — %s", side_str.upper(), qty, symbol, price_str, e)

            # TIF fallback: if the error suggests TIF issue, we already use DAY
            # which is the safest default for options
            if "time_in_force" in str(e).lower():
                logger.warning("TIF-related error detected. Options require TimeInForce.DAY.")

            return OrderResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side=side_str,
                qty=qty,
                price=limit_price,
                message=f"Failed: {e}",
            )

    def check_order_status(self, order_id: str) -> str:
        """Returns lowercase order status string: 'filled', 'partially_filled', 'canceled', etc.

        Alpaca's `Order.status` is an `OrderStatus` enum where `str(OrderStatus.FILLED)` is
        `'OrderStatus.FILLED'` but `.value` is `'filled'`. We normalize to the value form
        so callers can compare against lowercase literals consistently.
        """
        try:
            order = self._client.get_order(order_id)
            return _normalize_status(order.status)
        except Exception as e:
            logger.error("Failed to check order %s: %s", order_id, e)
            return "unknown"


def _normalize_status(status) -> str:
    if status is None:
        return "unknown"
    # Enum: prefer .value (e.g., 'filled')
    value = getattr(status, "value", None)
    if value is not None:
        return str(value).lower()
    # Plain string: strip any "OrderStatus." prefix and lowercase
    s = str(status)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s.lower()
