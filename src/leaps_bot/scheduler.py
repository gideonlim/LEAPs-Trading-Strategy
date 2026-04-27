from __future__ import annotations

import logging
from datetime import date, datetime

from leaps_bot.allocator import QuarterlyAllocator
from leaps_bot.alpaca_client import AlpacaClient, _safe_float
from leaps_bot.config import AppConfig
from leaps_bot.contract_finder import ContractFinder
from leaps_bot.models import (
    Action,
    AllocationRecord,
    ContractCandidate,
    OptionDetails,
    OrderResult,
    PendingOrderRecord,
    PositionRecord,
)
from leaps_bot.order_executor import OrderExecutor
from leaps_bot.position_manager import PositionManager
from leaps_bot.pricing import RateFetcher, black_scholes_call
from leaps_bot.state import BotState

logger = logging.getLogger(__name__)


class DailyScheduler:
    def __init__(
        self,
        config: AppConfig,
        client: AlpacaClient,
        state: BotState,
        rate_fetcher: RateFetcher,
    ):
        self._config = config
        self._client = client
        self._state = state
        self._rates = rate_fetcher

        self._finder = ContractFinder(client, config, rate_fetcher)
        self._position_mgr = PositionManager(client, config, state)
        self._executor = OrderExecutor(client, config)
        self._allocator = QuarterlyAllocator(client, config, state, self._finder, self._executor)

    def run(self) -> dict:
        summary: dict = {"actions": [], "errors": [], "skipped": False, "real_trades_today": False}
        today_iso = date.today().isoformat()
        now_str = datetime.now().isoformat()

        # last_run is a monitoring timestamp — always update, even on dry-run
        self._state.last_run = now_str

        if not self._preflight(summary):
            return summary

        # Idempotency: if we already triggered real trades today, only do monitoring
        already_traded = self._state.last_trade_date == today_iso
        if already_traded:
            logger.info("Already traded today (%s), running monitoring only", today_iso)

        self._rates.fetch_rates()

        # Pending order reconciliation runs every time — this is monitoring,
        # and is what mutates state on confirmed fills
        self._reconcile_pending_orders(summary)

        if not already_traded:
            self._handle_positions(summary)

            if self._allocator.should_allocate_today():
                self._handle_allocation(summary)

        # Mark last_trade_date only if a real (non-dry-run) order was actually
        # accepted by the broker today
        if summary["real_trades_today"] and not self._config.dry_run:
            self._state.last_trade_date = today_iso

        self._log_summary(summary)
        return summary

    # -- Preflight --

    def _preflight(self, summary: dict) -> bool:
        if not self._client.is_market_open():
            logger.info("Market is closed, skipping")
            summary["skipped"] = True
            summary["actions"].append({"type": "skip", "reason": "Market closed"})
            return False

        minutes = self._client.minutes_since_open()
        min_required = self._config.safety.no_trade_minutes_after_open
        if minutes < min_required:
            logger.info("Only %.0f minutes since open (need %d)", minutes, min_required)
            summary["skipped"] = True
            summary["actions"].append({
                "type": "skip",
                "reason": f"First-hour window: {minutes:.0f}m < {min_required}m",
            })
            return False

        try:
            acct = self._client.get_account()
            if getattr(acct, "trading_blocked", False):
                logger.error("Trading is blocked on this account")
                summary["errors"].append("Account trading blocked")
                return False
            if getattr(acct, "account_blocked", False):
                logger.error("Account is blocked")
                summary["errors"].append("Account blocked")
                return False

            options_level = getattr(acct, "options_trading_level", None)
            if options_level is not None and int(options_level) < 2:
                logger.error("Options trading level %s < 2 (need at least Level 2 to buy calls)", options_level)
                summary["errors"].append(f"Options level too low: {options_level}")
                return False
        except Exception as e:
            logger.error("Account check failed: %s", e)
            summary["errors"].append(f"Account check failed: {e}")
            return False

        return True

    # -- Pending order reconciliation (the only place state-changing fills are recorded) --

    # Order statuses that mean "we will not get any more fills on this order"
    _TERMINAL_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "done_for_day"})

    def _reconcile_pending_orders(self, summary: dict) -> None:
        for pending in list(self._state.pending_orders):
            try:
                status = self._executor.check_order_status(pending.order_id)
            except Exception as e:
                logger.warning("Could not check order %s: %s", pending.order_id, e)
                continue

            logger.info(
                "Pending %s order %s (intent=%s, %s): status=%s",
                pending.action, pending.order_id, pending.intent,
                pending.option_symbol, status,
            )

            # Fetch current fill details (for partial AND terminal states, we may have new qty to record)
            try:
                order = self._client.get_order(pending.order_id)
                filled_qty = int(_safe_float(getattr(order, "filled_qty", 0)))
                avg_price = _safe_float(getattr(order, "filled_avg_price", 0))
            except Exception as e:
                logger.warning("Could not fetch fill details for %s: %s", pending.order_id, e)
                filled_qty = pending.recorded_qty
                avg_price = 0.0

            new_qty = max(0, filled_qty - pending.recorded_qty)

            if new_qty > 0:
                self._record_fill_increment(pending, new_qty, avg_price, summary)
                pending.recorded_qty = filled_qty

            is_terminal = status in self._TERMINAL_STATUSES
            if is_terminal:
                if filled_qty == 0:
                    logger.warning(
                        "Order %s %s without any fill: %s — clearing pending record",
                        pending.order_id, status, pending.option_symbol,
                    )
                elif filled_qty < pending.qty:
                    logger.warning(
                        "Order %s %s with partial fill %d of %d: %s",
                        pending.order_id, status, filled_qty, pending.qty,
                        pending.option_symbol,
                    )

                summary["actions"].append({
                    "type": status,
                    "symbol": pending.option_symbol,
                    "intent": pending.intent,
                    "filled_qty": filled_qty,
                    "ordered_qty": pending.qty,
                })

                # Roll forward only after sell is confirmed terminal AND had at
                # least some fill (we're not rolling on zero-fill cancellations)
                if (
                    pending.action == "sell"
                    and pending.intent == "roll"
                    and pending.underlying
                    and filled_qty > 0
                ):
                    self._submit_roll_buy(pending.underlying, filled_qty, summary)

                self._state.remove_pending_order(pending.order_id)
            elif status == "partially_filled":
                summary["actions"].append({
                    "type": "partial_fill",
                    "symbol": pending.option_symbol,
                    "filled_qty": filled_qty,
                    "total_qty": pending.qty,
                    "intent": pending.intent,
                })
            # else: still pending (new, accepted, pending_new, etc.) — leave it

    def _record_fill_increment(
        self,
        pending: PendingOrderRecord,
        new_qty: int,
        avg_price: float,
        summary: dict,
    ) -> None:
        """Record `new_qty` shares filled on this order. Idempotent across reconciliations."""
        symbol = pending.option_symbol

        if pending.action == "buy":
            details = OptionDetails.from_occ_symbol(symbol)
            existing = self._state.get_position(symbol)
            if existing is not None:
                # Update qty + weighted avg price
                total_qty = existing.qty + new_qty
                if total_qty > 0:
                    existing.avg_entry_price = (
                        (existing.avg_entry_price * existing.qty + avg_price * new_qty) / total_qty
                    )
                existing.qty = total_qty
                logger.info(
                    "Position updated on fill increment: %s now qty=%d (added %d)",
                    symbol, total_qty, new_qty,
                )
            else:
                self._state.add_position(PositionRecord(
                    option_symbol=symbol,
                    underlying=details.underlying,
                    strike=details.strike,
                    expiry_date=details.expiry.isoformat(),
                    purchase_date=date.today().isoformat(),
                    original_dte=(details.expiry - date.today()).days,
                    qty=new_qty,
                    avg_entry_price=avg_price,
                    order_id=pending.order_id,
                ))
                logger.info(
                    "Position recorded on fill: %s qty=%d @ $%.2f",
                    symbol, new_qty, avg_price,
                )

            # For allocation intent: record the allocation on the FIRST fill
            # (any fill = capital deployed = quarter is allocated). Subsequent
            # increments update the existing record's amount.
            if pending.intent == "allocate" and pending.quarter:
                existing_alloc = next(
                    (a for a in self._state.allocations if a.quarter == pending.quarter),
                    None,
                )
                fill_value = avg_price * new_qty * 100
                if existing_alloc is None:
                    self._state.record_allocation(AllocationRecord(
                        quarter=pending.quarter,
                        allocated_date=date.today().isoformat(),
                        amount=fill_value,
                        contracts_bought=[symbol],
                    ))
                    logger.info(
                        "Allocation recorded for %s on first fill: $%.2f",
                        pending.quarter, fill_value,
                    )
                else:
                    existing_alloc.amount += fill_value
                    if symbol not in existing_alloc.contracts_bought:
                        existing_alloc.contracts_bought.append(symbol)

            summary["actions"].append({
                "type": "fill_buy",
                "symbol": symbol,
                "qty": new_qty,
                "intent": pending.intent,
                "price": avg_price,
            })

        elif pending.action == "sell":
            existing = self._state.get_position(symbol)
            if existing is not None:
                existing.qty -= new_qty
                if existing.qty <= 0:
                    self._state.remove_position(symbol)
                    logger.info("Position removed after sell fill: %s", symbol)
                else:
                    logger.info(
                        "Position reduced on sell fill: %s now qty=%d (sold %d)",
                        symbol, existing.qty, new_qty,
                    )
            else:
                logger.warning(
                    "Sell fill for %s but no position record found", symbol,
                )

            summary["actions"].append({
                "type": "fill_sell",
                "symbol": symbol,
                "qty": new_qty,
                "intent": pending.intent,
            })

    # -- Position handling: submit sells, do NOT roll until fill confirmed --

    def _handle_positions(self, summary: dict) -> None:
        actions = self._position_mgr.evaluate_positions()
        for action in actions:
            if action.action in (Action.SELL, Action.EMERGENCY_SELL):
                self._submit_sell(action, summary)

    def _submit_sell(self, action, summary: dict) -> None:
        if self._state.has_pending_roll(action.option_symbol):
            logger.info("Sell already pending for %s, skipping", action.option_symbol)
            return

        sell_candidate = self._get_sell_snapshot(action.option_symbol)
        limit_price = None
        if sell_candidate and self._config.strategy.order_type == "limit":
            limit_price = self._finder.calculate_limit_price(sell_candidate, "sell")

        result = self._executor.execute_sell(action.option_symbol, action.qty, limit_price)

        details = OptionDetails.from_occ_symbol(action.option_symbol)
        intent = "roll"  # always roll forward after selling LEAPs

        summary["actions"].append({
            "type": "sell_submitted",
            "symbol": action.option_symbol,
            "qty": action.qty,
            "intent": intent,
            "reason": action.reason,
            "success": result.success,
            "dry_run": result.dry_run,
        })

        # Only mutate state on real (non-dry-run) successful submission
        if result.success and not result.dry_run and result.order_id:
            self._state.add_pending_order(PendingOrderRecord(
                order_id=result.order_id,
                action="sell",
                intent=intent,
                option_symbol=action.option_symbol,
                qty=action.qty,
                submitted_at=datetime.now().isoformat(),
                underlying=details.underlying,
            ))
            summary["real_trades_today"] = True

    def _submit_roll_buy(self, underlying: str, qty: int, summary: dict) -> None:
        candidate = self._finder.find_best_leaps_call(underlying)
        if candidate is None:
            logger.error("Roll forward failed: no LEAPs contract found for %s", underlying)
            summary["errors"].append(f"Roll forward: no contract for {underlying}")
            return

        limit_price = self._finder.calculate_limit_price(candidate, "buy")
        result = self._executor.execute_buy(candidate.symbol, qty, limit_price)

        summary["actions"].append({
            "type": "roll_buy_submitted",
            "symbol": candidate.symbol,
            "qty": qty,
            "success": result.success,
            "dry_run": result.dry_run,
        })

        if result.success and not result.dry_run and result.order_id:
            self._state.add_pending_order(PendingOrderRecord(
                order_id=result.order_id,
                action="buy",
                intent="open",  # already part of a roll, but the buy itself just opens
                option_symbol=candidate.symbol,
                qty=qty,
                submitted_at=datetime.now().isoformat(),
                underlying=underlying,
            ))
            summary["real_trades_today"] = True

    # -- Allocation handling: submit buy, record allocation only on fill --

    def _handle_allocation(self, summary: dict) -> None:
        results = self._allocator.allocate()
        for r in results:
            summary["actions"].append({
                "type": "allocate_submitted",
                "symbol": r.symbol,
                "qty": r.qty,
                "success": r.success,
                "dry_run": r.dry_run,
            })

            if r.success and not r.dry_run and r.order_id:
                today_iso = date.today().isoformat()
                self._state.add_pending_order(PendingOrderRecord(
                    order_id=r.order_id,
                    action="buy",
                    intent="allocate",
                    option_symbol=r.symbol,
                    qty=r.qty,
                    submitted_at=datetime.now().isoformat(),
                    quarter=self._state.current_quarter_key(today_iso),
                ))
                summary["real_trades_today"] = True

    # -- Helpers --

    def _get_sell_snapshot(self, symbol: str) -> ContractCandidate | None:
        try:
            snapshots = self._client.get_option_snapshot(symbol)
            snapshot = snapshots.get(symbol) if isinstance(snapshots, dict) else snapshots
            if not snapshot or not snapshot.latest_quote:
                return None

            bid = _safe_float(getattr(snapshot.latest_quote, "bid_price", 0))
            ask = _safe_float(getattr(snapshot.latest_quote, "ask_price", 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            if mid <= 0:
                return None

            details = OptionDetails.from_occ_symbol(symbol)
            iv = _safe_float(getattr(snapshot, "implied_volatility", 0.20))
            T = (details.expiry - date.today()).days / 365.0
            underlying_price = self._client.get_underlying_price(details.underlying)
            theo = black_scholes_call(
                S=underlying_price, K=details.strike, T=T,
                r=self._rates.risk_free_rate, q=self._rates.dividend_yield,
                sigma=iv if iv > 0 else 0.20,
            )
            return ContractCandidate(
                symbol=symbol, underlying=details.underlying,
                strike=details.strike, expiry=details.expiry,
                bid=bid, ask=ask, mid=mid, delta=0, iv=iv,
                open_interest=0, theoretical_price=theo,
            )
        except Exception as e:
            logger.warning("Could not get snapshot for %s: %s", symbol, e)
            return None

    def _log_summary(self, summary: dict) -> None:
        logger.info("=== Daily Run Summary ===")
        if summary["skipped"]:
            logger.info("Run skipped: %s", summary["actions"][0].get("reason", "unknown"))
            return

        for action in summary["actions"]:
            logger.info("  %s: %s", action.get("type", "?").upper(), action)

        for err in summary["errors"]:
            logger.error("  ERROR: %s", err)

        try:
            cash = self._client.get_cash_available()
            bp = self._client.get_buying_power()
            positions = self._client.get_option_positions()
            logger.info(
                "  Account: cash=$%.2f, buying_power=$%.2f, option_positions=%d",
                cash, bp, len(positions),
            )
        except Exception:
            pass

        logger.info("=== End Summary ===")

    def get_status(self) -> dict:
        try:
            acct = self._client.get_account()
            cash = _safe_float(acct.cash)
            bp = _safe_float(acct.buying_power)
            opt_bp = _safe_float(getattr(acct, "options_buying_power", None))
        except Exception as e:
            return {"error": f"Account error: {e}"}

        positions = []
        try:
            for pos in self._client.get_option_positions():
                details = OptionDetails.from_occ_symbol(pos.symbol)
                days_rem = (details.expiry - date.today()).days
                state_rec = self._state.get_position(pos.symbol)
                original_dte = state_rec.original_dte if state_rec else days_rem + 90
                sell_at = int(original_dte * self._config.strategy.sell_threshold_fraction)
                positions.append({
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "market_value": getattr(pos, "market_value", "?"),
                    "days_remaining": days_rem,
                    "sell_at_days": sell_at,
                    "underlying": details.underlying,
                    "strike": details.strike,
                    "expiry": details.expiry.isoformat(),
                })
        except Exception as e:
            return {"error": f"Position error: {e}", "cash": cash, "buying_power": bp}

        today_iso = date.today().isoformat()
        return {
            "cash": cash,
            "buying_power": bp,
            "options_buying_power": opt_bp,
            "positions": positions,
            "last_run": self._state.last_run,
            "last_trade_date": self._state.last_trade_date,
            "current_quarter": self._state.current_quarter_key(today_iso),
            "allocated_this_quarter": self._state.has_allocated_this_quarter(today_iso),
            "pending_orders": [
                {"id": o.order_id, "action": o.action, "intent": o.intent, "symbol": o.option_symbol}
                for o in self._state.pending_orders
            ],
        }
