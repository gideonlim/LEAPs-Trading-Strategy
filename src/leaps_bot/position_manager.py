from __future__ import annotations

import logging
from datetime import date

from leaps_bot.alpaca_client import AlpacaClient, _safe_float
from leaps_bot.config import AppConfig
from leaps_bot.models import Action, OptionDetails, PositionAction
from leaps_bot.state import BotState

logger = logging.getLogger(__name__)


class PositionManager:
    def __init__(self, client: AlpacaClient, config: AppConfig, state: BotState):
        self._client = client
        self._config = config
        self._state = state

    def evaluate_positions(self) -> list[PositionAction]:
        positions = self._client.get_option_positions()
        if not positions:
            logger.info("No option positions to evaluate")
            return []

        actions = []
        today = date.today()

        for pos in positions:
            symbol = pos.symbol
            qty = abs(int(_safe_float(pos.qty)))
            if qty == 0:
                continue

            details = OptionDetails.from_occ_symbol(symbol)
            days_remaining = (details.expiry - today).days

            state_record = self._state.get_position(symbol)
            if state_record:
                original_dte = state_record.original_dte
            else:
                # Fallback: estimate from purchase date if we have position data
                # but no state record (e.g., positions opened before bot was set up)
                original_dte = days_remaining + 90  # conservative guess
                logger.warning(
                    "No state record for %s, estimating original DTE as %d",
                    symbol, original_dte,
                )

            sell_threshold_days = int(original_dte * self._config.strategy.sell_threshold_fraction)
            emergency_days = self._config.safety.emergency_sell_days

            if days_remaining <= emergency_days:
                action = PositionAction(
                    action=Action.EMERGENCY_SELL,
                    option_symbol=symbol,
                    qty=qty,
                    days_remaining=days_remaining,
                    original_dte=original_dte,
                    sell_threshold_days=sell_threshold_days,
                    reason=f"EMERGENCY: {days_remaining} days to expiry (< {emergency_days}d safety limit)",
                )
                logger.critical(
                    "EMERGENCY SELL: %s has only %d days to expiry", symbol, days_remaining
                )
            elif days_remaining <= sell_threshold_days:
                action = PositionAction(
                    action=Action.SELL,
                    option_symbol=symbol,
                    qty=qty,
                    days_remaining=days_remaining,
                    original_dte=original_dte,
                    sell_threshold_days=sell_threshold_days,
                    reason=(
                        f"Sell threshold reached: {days_remaining}d remaining "
                        f"≤ {sell_threshold_days}d (1/{1/self._config.strategy.sell_threshold_fraction:.0f} of {original_dte}d)"
                    ),
                )
                logger.info("SELL: %s — %s", symbol, action.reason)
            else:
                action = PositionAction(
                    action=Action.HOLD,
                    option_symbol=symbol,
                    qty=qty,
                    days_remaining=days_remaining,
                    original_dte=original_dte,
                    sell_threshold_days=sell_threshold_days,
                    reason=f"Holding: {days_remaining}d remaining (sell at ≤{sell_threshold_days}d)",
                )
                logger.info(
                    "HOLD: %s — %dd remaining, sell at ≤%dd",
                    symbol, days_remaining, sell_threshold_days,
                )

            actions.append(action)

        sell_count = sum(1 for a in actions if a.action != Action.HOLD)
        logger.info(
            "Position evaluation: %d positions, %d to sell, %d to hold",
            len(actions), sell_count, len(actions) - sell_count,
        )
        return actions
