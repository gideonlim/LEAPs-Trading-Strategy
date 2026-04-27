from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from leaps_bot.allocator import QuarterlyAllocator
from leaps_bot.alpaca_client import AlpacaClient, _safe_float
from leaps_bot.config import AppConfig
from leaps_bot.contract_finder import ContractFinder
from leaps_bot.models import (
    Action,
    AllocationRecord,
    ContractCandidate,
    DailySnapshot,
    FollowupAction,
    OptionDetails,
    OrderResult,
    PendingOrderRecord,
    PositionRecord,
    PositionSnapshot,
    RunRecord,
    TradeRecord,
    now_utc_iso,
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
        # Use UTC-aware timestamps consistently. Naive datetime.now() would
        # interpret as local time on the host, which would skew ordering
        # against tz-aware broker timestamps when running off a non-UTC box.
        run_start = datetime.now(timezone.utc)
        now_str = run_start.isoformat()

        # last_run is a monitoring timestamp — always update, even on dry-run
        self._state.last_run = now_str

        try:
            if not self._preflight(summary):
                return summary

            already_traded = self._state.last_trade_date == today_iso
            if already_traded:
                logger.info("Already traded today (%s), running monitoring only", today_iso)

            self._rates.fetch_rates()
            self._reconcile_pending_orders(summary)

            if not already_traded:
                # Drain any followup actions queued by prior monitor passes
                # (e.g., roll buys for sells that filled mid-day yesterday)
                self._process_pending_followups(summary)
                self._handle_positions(summary)
                if self._allocator.should_allocate_today():
                    self._handle_allocation(summary)

            if summary["real_trades_today"] and not self._config.dry_run:
                self._state.last_trade_date = today_iso

            # Capture daily portfolio snapshot — only on full runs (preflight passed)
            # and not in dry-run mode (snapshots are persistent state).
            if not self._config.dry_run:
                snapshot = self._capture_daily_snapshot(today_iso, now_str)
                if snapshot is not None:
                    self._state.add_snapshot(snapshot)
                    summary["portfolio_value"] = snapshot.portfolio_value

            self._log_summary(summary)
            return summary
        finally:
            # Always record the run, even on early exit / skip / exception
            if not self._config.dry_run:
                duration = (datetime.now(timezone.utc) - run_start).total_seconds()
                self._record_run(summary, now_str, duration)

    def monitor(self, capture_snapshot: bool = True) -> dict:
        """Read-only reconciliation pass. NEVER originates orders.

        Used by the mid-day and EOD monitor workflows to:
        1. Reconcile pending orders (record fills, partial fills, expirations)
        2. Optionally capture a portfolio snapshot

        This method is SAFE to invoke at any time — including outside market
        hours and outside the no-trade window. The hard invariant is that
        it never calls `_handle_positions`, `_handle_allocation`, or any
        order-submission path. It only observes broker state and updates
        local bookkeeping.
        """
        summary: dict = {
            "actions": [], "errors": [], "skipped": False,
            "real_trades_today": False, "monitor_only": True,
        }
        today_iso = date.today().isoformat()
        run_start = datetime.now(timezone.utc)
        now_str = run_start.isoformat()

        self._state.last_run = now_str

        try:
            # Account preflight — but skip the market-hours / no-trade-window
            # checks that gate _trading_. Monitor is allowed when market is
            # closed (that's when EOD reconciliation typically happens).
            try:
                acct = self._client.get_account()
                if getattr(acct, "trading_blocked", False):
                    logger.error("Trading is blocked on this account")
                    summary["errors"].append("Account trading blocked")
                    return summary
            except Exception as e:
                logger.error("Account check failed: %s", e)
                summary["errors"].append(f"Account check failed: {e}")
                return summary

            # Reconcile any pending orders. This is the work the morning
            # workflow couldn't do (orders submitted at 10:33 AM may have
            # filled at 10:35 AM, but we wouldn't know until tomorrow without
            # a mid-day or EOD pass).
            self._reconcile_pending_orders(summary)

            if capture_snapshot and not self._config.dry_run:
                snapshot = self._capture_daily_snapshot(today_iso, now_str)
                if snapshot is not None:
                    self._state.add_snapshot(snapshot)
                    summary["portfolio_value"] = snapshot.portfolio_value

            self._log_summary(summary)
            return summary
        finally:
            if not self._config.dry_run:
                duration = (datetime.now(timezone.utc) - run_start).total_seconds()
                self._record_run(summary, now_str, duration)

    def _capture_daily_snapshot(self, today_iso: str, timestamp: str) -> DailySnapshot | None:
        try:
            acct = self._client.get_account()
        except Exception as e:
            logger.warning("Could not fetch account for snapshot: %s", e)
            return None

        cash = _safe_float(getattr(acct, "cash", 0))
        opt_bp = _safe_float(getattr(acct, "options_buying_power", None))
        portfolio_value = _safe_float(getattr(acct, "portfolio_value", None))

        position_snaps: list[PositionSnapshot] = []
        positions_market_value = 0.0
        underlying_prices: dict[str, float] = {}

        try:
            positions = self._client.get_option_positions()
        except Exception as e:
            logger.warning("Could not fetch positions for snapshot: %s", e)
            positions = []

        for pos in positions:
            try:
                sym = pos.symbol
                details = OptionDetails.from_occ_symbol(sym)
                qty = int(_safe_float(getattr(pos, "qty", 0)))
                avg_entry = _safe_float(getattr(pos, "avg_entry_price", 0))
                current_price = _safe_float(getattr(pos, "current_price", 0))
                market_value = _safe_float(getattr(pos, "market_value", 0))
                cost_basis = _safe_float(getattr(pos, "cost_basis", 0))
                unrealized_pl = _safe_float(getattr(pos, "unrealized_pl", 0))
                unrealized_plpc = _safe_float(getattr(pos, "unrealized_plpc", 0))
                days_remaining = (details.expiry - date.today()).days

                position_snaps.append(PositionSnapshot(
                    symbol=sym,
                    underlying=details.underlying,
                    strike=details.strike,
                    expiry=details.expiry.isoformat(),
                    qty=qty,
                    avg_entry_price=avg_entry,
                    current_price=current_price,
                    market_value=market_value,
                    cost_basis=cost_basis,
                    unrealized_pl=unrealized_pl,
                    unrealized_plpc=unrealized_plpc,
                    days_remaining=days_remaining,
                ))
                positions_market_value += market_value

                if details.underlying not in underlying_prices:
                    spot = self._safe_get_underlying_price(details.underlying)
                    if spot is not None:
                        underlying_prices[details.underlying] = spot
            except Exception as e:
                logger.warning("Skipping position %s in snapshot: %s", getattr(pos, "symbol", "?"), e)

        # Always capture SPY price for benchmarking, even with no positions
        if "SPY" not in underlying_prices:
            spy = self._safe_get_underlying_price("SPY")
            if spy is not None:
                underlying_prices["SPY"] = spy

        # If account.portfolio_value isn't available, derive it
        if portfolio_value <= 0:
            portfolio_value = cash + positions_market_value

        return DailySnapshot(
            date=today_iso,
            timestamp=timestamp,
            cash=cash,
            options_buying_power=opt_bp,
            portfolio_value=portfolio_value,
            positions_market_value=positions_market_value,
            num_positions=len(position_snaps),
            underlying_prices=underlying_prices,
            positions=position_snaps,
        )

    def _record_run(self, summary: dict, timestamp: str, duration: float) -> None:
        action_summaries = [
            f"{a.get('type', '?')}:{a.get('symbol', '')}" for a in summary.get("actions", [])
        ]
        skip_reason = None
        if summary.get("skipped"):
            for a in summary.get("actions", []):
                if a.get("type") == "skip":
                    skip_reason = a.get("reason")
                    break

        self._state.add_run(RunRecord(
            timestamp=timestamp,
            duration_seconds=duration,
            skipped=summary.get("skipped", False),
            skip_reason=skip_reason,
            actions=action_summaries,
            errors=list(summary.get("errors", [])),
            real_trades_today=summary.get("real_trades_today", False),
            portfolio_value=summary.get("portfolio_value"),
            run_type="monitor" if summary.get("monitor_only") else "trade",
        ))

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
            order = None
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
                fill_timestamp = self._extract_fill_timestamp(order)
                self._record_fill_increment(pending, new_qty, avg_price, fill_timestamp, summary)
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
                # least some fill (we're not rolling on zero-fill cancellations).
                # CRITICAL: this only QUEUES the followup. The actual buy is
                # submitted by `_process_pending_followups`, which is called
                # ONLY from `run()`, never from `monitor()`. This keeps the
                # safety invariant that monitor never originates orders.
                if (
                    pending.action == "sell"
                    and pending.intent == "roll"
                    and pending.underlying
                    and filled_qty > 0
                ):
                    self._state.pending_followups.append(FollowupAction(
                        action_type="roll",
                        underlying=pending.underlying,
                        qty=filled_qty,
                        sourced_from_order_id=pending.order_id,
                        queued_at=now_utc_iso(),
                    ))
                    logger.info(
                        "Queued roll followup: buy %d %s LEAPs (sourced from sell %s)",
                        filled_qty, pending.underlying, pending.order_id,
                    )

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
        fill_timestamp: str,
        summary: dict,
    ) -> None:
        """Record `new_qty` shares filled on this order. Idempotent across reconciliations.

        `fill_timestamp` should be the broker's actual fill time (ISO datetime), not
        the reconciliation time. This matters for tax classification near year-end
        and for accurate holding-period calculation.

        Also appends a TradeRecord for reporting (with realized P&L computed for sells).
        """
        symbol = pending.option_symbol
        details = OptionDetails.from_occ_symbol(symbol)
        underlying_price = self._safe_get_underlying_price(details.underlying)
        fill_date = self._parse_iso_date(fill_timestamp)

        if pending.action == "buy":
            existing = self._state.get_position(symbol)
            if existing is not None:
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
                    purchase_date=fill_date.isoformat(),
                    original_dte=(details.expiry - fill_date).days,
                    qty=new_qty,
                    avg_entry_price=avg_price,
                    order_id=pending.order_id,
                ))
                logger.info(
                    "Position recorded on fill: %s qty=%d @ $%.2f",
                    symbol, new_qty, avg_price,
                )

            # For allocation intent: record on first fill, accumulate on subsequent
            if pending.intent == "allocate" and pending.quarter:
                existing_alloc = next(
                    (a for a in self._state.allocations if a.quarter == pending.quarter),
                    None,
                )
                fill_value = avg_price * new_qty * 100
                if existing_alloc is None:
                    self._state.record_allocation(AllocationRecord(
                        quarter=pending.quarter,
                        allocated_date=fill_date.isoformat(),
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

            self._state.add_trade(TradeRecord(
                timestamp=fill_timestamp,
                order_id=pending.order_id,
                action="buy",
                intent=pending.intent,
                symbol=symbol,
                underlying=details.underlying,
                strike=details.strike,
                expiry=details.expiry.isoformat(),
                qty=new_qty,
                fill_price=avg_price,
                total_value=avg_price * new_qty * 100,
                underlying_price=underlying_price,
            ))

            summary["actions"].append({
                "type": "fill_buy",
                "symbol": symbol,
                "qty": new_qty,
                "intent": pending.intent,
                "price": avg_price,
            })

        elif pending.action == "sell":
            existing = self._state.get_position(symbol)
            entry_price = existing.avg_entry_price if existing else None
            purchase_date_str = existing.purchase_date if existing else None

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
                logger.warning("Sell fill for %s but no position record found", symbol)

            realized_pnl = None
            holding_days = None
            if entry_price is not None:
                realized_pnl = (avg_price - entry_price) * new_qty * 100
                if purchase_date_str:
                    try:
                        purchase_date = date.fromisoformat(purchase_date_str)
                        # Holding period must be measured from purchase to actual fill,
                        # not to today. Reconciliation can lag the fill by hours/days,
                        # which would otherwise inflate holding period and could flip
                        # short-vs-long-term tax classification near the 365-day boundary.
                        holding_days = (fill_date - purchase_date).days
                    except ValueError:
                        pass

            self._state.add_trade(TradeRecord(
                timestamp=fill_timestamp,
                order_id=pending.order_id,
                action="sell",
                intent=pending.intent,
                symbol=symbol,
                underlying=details.underlying,
                strike=details.strike,
                expiry=details.expiry.isoformat(),
                qty=new_qty,
                fill_price=avg_price,
                total_value=avg_price * new_qty * 100,
                underlying_price=underlying_price,
                avg_entry_price=entry_price,
                realized_pnl=realized_pnl,
                holding_days=holding_days,
            ))

            if realized_pnl is not None:
                logger.info(
                    "Sell fill: %s qty=%d, entry=$%.2f, exit=$%.2f, realized P&L=$%.2f",
                    symbol, new_qty, entry_price, avg_price, realized_pnl,
                )

            summary["actions"].append({
                "type": "fill_sell",
                "symbol": symbol,
                "qty": new_qty,
                "intent": pending.intent,
                "realized_pnl": realized_pnl,
            })

    def _safe_get_underlying_price(self, underlying: str) -> float | None:
        try:
            return self._client.get_underlying_price(underlying)
        except Exception as e:
            logger.warning("Could not fetch underlying price for %s: %s", underlying, e)
            return None

    @staticmethod
    def _extract_fill_timestamp(order) -> str:
        """Return the broker's actual fill time as ISO string.

        Prefers `filled_at` (set when the order is fully filled), falls back to
        `updated_at` (last change — typically the most recent partial fill),
        then `submitted_at`, then now() (UTC-aware) as last resort. This is what
        should be recorded on TradeRecord and used for holding-period math, NOT
        the reconciliation time, since reconciliation can lag the actual fill.
        """
        if order is None:
            return now_utc_iso()
        for attr in ("filled_at", "updated_at", "submitted_at"):
            ts = getattr(order, attr, None)
            if ts is None:
                continue
            if isinstance(ts, datetime):
                return ts.isoformat()
            return str(ts)
        return now_utc_iso()

    @staticmethod
    def _parse_iso_date(iso_timestamp: str) -> date:
        """Extract the date portion from an ISO datetime string."""
        try:
            # Strip timezone suffix and microseconds before parsing
            return datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00")).date()
        except (ValueError, AttributeError):
            try:
                return date.fromisoformat(iso_timestamp.split("T")[0])
            except (ValueError, AttributeError, IndexError):
                return date.today()

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
                submitted_at=now_utc_iso(),
                underlying=details.underlying,
            ))
            summary["real_trades_today"] = True

    def _process_pending_followups(self, summary: dict) -> None:
        """Drain the queue of followup actions left by prior monitor passes.

        Each FollowupAction was queued when reconciliation observed a relevant
        fill but couldn't act (because monitor mustn't originate orders). The
        morning trading run is the only place these are processed.

        Successfully executed followups are removed. Failed ones STAY QUEUED
        so the next morning's run retries them — a transient failure (no
        contract found, API blip) shouldn't permanently drop a roll that the
        strategy needs for position continuity.
        """
        if not self._state.pending_followups:
            return

        remaining: list[FollowupAction] = []
        for followup in self._state.pending_followups:
            if followup.action_type == "roll":
                logger.info(
                    "Processing queued roll: buy %d %s LEAPs (queued at %s, sourced from %s)",
                    followup.qty, followup.underlying, followup.queued_at,
                    followup.sourced_from_order_id,
                )
                success = self._submit_roll_buy(followup.underlying, followup.qty, summary)
                if not success:
                    logger.warning(
                        "Roll followup failed — keeping queued for retry on next run "
                        "(underlying=%s, qty=%d, sourced_from=%s)",
                        followup.underlying, followup.qty, followup.sourced_from_order_id,
                    )
                    remaining.append(followup)
            else:
                logger.warning(
                    "Unknown followup action_type %r — leaving in queue for manual review",
                    followup.action_type,
                )
                remaining.append(followup)

        self._state.pending_followups = remaining

    def _submit_roll_buy(self, underlying: str, qty: int, summary: dict) -> bool:
        """Attempt to find and buy replacement LEAPs. Returns True if an order
        was successfully accepted by the broker (or simulated in dry-run)."""
        candidate = self._finder.find_best_leaps_call(underlying)
        if candidate is None:
            logger.error("Roll forward failed: no LEAPs contract found for %s", underlying)
            summary["errors"].append(f"Roll forward: no contract for {underlying}")
            return False

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
                intent="open",
                option_symbol=candidate.symbol,
                qty=qty,
                submitted_at=now_utc_iso(),
                underlying=underlying,
            ))
            summary["real_trades_today"] = True

        return result.success

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
                    submitted_at=now_utc_iso(),
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
