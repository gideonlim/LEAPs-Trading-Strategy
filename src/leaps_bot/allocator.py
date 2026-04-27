from __future__ import annotations

import logging
import math
from datetime import date

from leaps_bot.alpaca_client import AlpacaClient
from leaps_bot.config import AppConfig
from leaps_bot.contract_finder import ContractFinder
from leaps_bot.models import OrderResult
from leaps_bot.order_executor import OrderExecutor
from leaps_bot.state import BotState

logger = logging.getLogger(__name__)


class QuarterlyAllocator:
    def __init__(
        self,
        client: AlpacaClient,
        config: AppConfig,
        state: BotState,
        contract_finder: ContractFinder,
        order_executor: OrderExecutor,
    ):
        self._client = client
        self._config = config
        self._state = state
        self._finder = contract_finder
        self._executor = order_executor

    def should_allocate_today(self) -> bool:
        today = date.today()
        if today.month not in self._config.allocation.quarterly_months:
            return False

        # Enforce allocation window: only within the first N calendar days of the month
        window = self._config.allocation.allocation_window_days
        if today.day > window:
            logger.info(
                "Past allocation window (day %d > first %d days of month), skipping",
                today.day, window,
            )
            return False

        today_iso = today.isoformat()
        if self._state.has_allocated_this_quarter(today_iso):
            logger.info("Already allocated for %s", self._state.current_quarter_key(today_iso))
            return False

        logger.info(
            "Allocation eligible: day %d of month %d (within %d-day window, no prior allocation this quarter)",
            today.day, today.month, window,
        )
        return True

    def calculate_deployment(self) -> float:
        # Use the more conservative of cash and options buying power. On a cash
        # account they're typically equal; on margin, options BP can be lower
        # due to options-specific margin rules.
        cash = self._client.get_cash_available()
        options_bp = self._client.get_options_buying_power()
        available = min(cash, options_bp) if options_bp > 0 else cash

        reserve = self._config.allocation.min_cash_reserve
        max_deploy = available * self._config.allocation.max_cash_deploy_pct

        amount = min(max_deploy, available - reserve)
        amount = max(0.0, amount)

        logger.info(
            "Allocation calc: cash=$%.2f, options_bp=$%.2f, reserve=$%.2f, max_deploy=$%.2f → deploy=$%.2f",
            cash, options_bp, reserve, max_deploy, amount,
        )
        return amount

    def allocate(self) -> list[OrderResult]:
        """Submit buy order(s) for quarterly allocation.

        State mutations (recording the allocation) happen in the scheduler's
        pending order reconciliation when the buy fills, NOT here. This prevents
        marking a quarter as 'allocated' when the limit order may expire unfilled.
        """
        deployment = self.calculate_deployment()
        if deployment <= 0:
            logger.warning("No cash available for allocation")
            return []

        candidate = self._finder.find_best_leaps_call()
        if candidate is None:
            logger.error("No suitable LEAPs contract found for allocation")
            return []

        cost_per_contract = candidate.ask * 100
        if cost_per_contract <= 0:
            logger.error("Invalid contract cost: $%.2f", cost_per_contract)
            return []

        max_contracts = math.floor(deployment / cost_per_contract)
        if max_contracts < 1:
            logger.warning(
                "Cannot afford even 1 contract of %s (cost $%.2f, available $%.2f)",
                candidate.symbol, cost_per_contract, deployment,
            )
            return []

        qty = max_contracts
        logger.info(
            "Allocating: %d contracts of %s (cost ~$%.2f each, total ~$%.2f)",
            qty, candidate.symbol, cost_per_contract, qty * cost_per_contract,
        )

        limit_price = self._finder.calculate_limit_price(candidate, "buy")
        result = self._executor.execute_buy(candidate.symbol, qty, limit_price)
        return [result]
