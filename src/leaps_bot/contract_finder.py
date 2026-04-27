from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from leaps_bot.alpaca_client import AlpacaClient, _safe_float
from leaps_bot.config import AppConfig
from leaps_bot.models import ContractCandidate
from leaps_bot.pricing import RateFetcher, black_scholes_call, intrinsic_value

logger = logging.getLogger(__name__)


class ContractFinder:
    def __init__(self, client: AlpacaClient, config: AppConfig, rate_fetcher: RateFetcher):
        self._client = client
        self._config = config
        self._rates = rate_fetcher

    def find_best_leaps_call(self, underlying: str | None = None) -> ContractCandidate | None:
        if underlying is None:
            underlying = self._determine_underlying()
            if underlying is None:
                return None

        spot = self._client.get_underlying_price(underlying)
        logger.info("Current %s price: $%.2f", underlying, spot)

        target_strike = spot * (1 - self._config.strategy.itm_depth_pct)
        expiry_gte, expiry_lte = self._expiry_window()
        logger.info(
            "Searching: strike ≤ $%.2f, expiry %s to %s",
            target_strike, expiry_gte, expiry_lte,
        )

        candidates = self._get_candidates(underlying, spot, target_strike, expiry_gte, expiry_lte)
        if not candidates:
            logger.warning("No candidates found, widening expiry window by ±1 month")
            expiry_gte -= timedelta(days=30)
            expiry_lte += timedelta(days=30)
            candidates = self._get_candidates(underlying, spot, target_strike, expiry_gte, expiry_lte)

        if not candidates:
            logger.error("No viable LEAPs contracts found for %s", underlying)
            return None

        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]
        logger.info(
            "Best candidate: %s (strike=$%.2f, expiry=%s, delta=%.3f, score=%.2f, theo=$%.2f, mid=$%.2f)",
            best.symbol, best.strike, best.expiry, best.delta, best.score,
            best.theoretical_price, best.mid,
        )
        return best

    def _determine_underlying(self) -> str | None:
        pref = self._config.strategy.underlying_preference
        if pref.upper() in ("SPY",):
            return pref.upper()

        # Auto: always use SPY. The actual affordability check happens downstream
        # in the allocator when it computes cost_per_contract from the real ask
        # price. We don't gate on a rough estimate here because intrinsic-based
        # guesses can be 20-30% off from the actual ask (time value on LEAPs is
        # substantial), which would block accounts that CAN afford the real price.
        return "SPY"

    def _expiry_window(self) -> tuple[date, date]:
        today = date.today()
        gte = today + timedelta(days=self._config.strategy.min_expiry_months * 30)
        lte = today + timedelta(days=self._config.strategy.max_expiry_months * 30)
        return gte, lte

    def _get_candidates(
        self,
        underlying: str,
        spot: float,
        target_strike: float,
        expiry_gte: date,
        expiry_lte: date,
    ) -> list[ContractCandidate]:
        # Strike range: target ± 5% of spot to capture nearby listed strikes
        strike_margin = spot * 0.05
        strike_gte = target_strike - strike_margin
        strike_lte = target_strike + strike_margin

        contracts = self._client.find_option_contracts(
            underlying=underlying,
            expiry_gte=expiry_gte,
            expiry_lte=expiry_lte,
            strike_lte=strike_lte,
            strike_gte=strike_gte,
        )
        if not contracts:
            return []

        # Build symbol -> contract metadata map (has open_interest)
        contract_meta: dict[str, Any] = {}
        for c in contracts:
            sym = c.symbol
            contract_meta[sym] = c

        # Get option chain snapshots for quotes + Greeks
        chain = self._client.get_option_chain(
            underlying=underlying,
            expiry_gte=expiry_gte,
            expiry_lte=expiry_lte,
            strike_lte=strike_lte,
            strike_gte=strike_gte,
        )

        candidates = []
        for sym, snapshot in chain.items():
            meta = contract_meta.get(sym)

            quote = snapshot.latest_quote if snapshot.latest_quote else None
            greeks = snapshot.greeks if hasattr(snapshot, "greeks") else None

            # Stale quote guard
            quote_ts = getattr(quote, "timestamp", None) if quote else None
            if quote_ts is not None:
                try:
                    if isinstance(quote_ts, datetime):
                        ts_dt = quote_ts
                    else:
                        ts_dt = datetime.fromisoformat(str(quote_ts).replace("Z", "+00:00"))
                    if ts_dt.tzinfo is None:
                        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                    age_min = (datetime.now(timezone.utc) - ts_dt).total_seconds() / 60.0
                    if age_min > self._config.safety.stale_quote_warn_minutes:
                        logger.warning(
                            "%s: quote is %.0f minutes old (> %dm threshold) — relying on Black-Scholes",
                            sym, age_min, self._config.safety.stale_quote_warn_minutes,
                        )
                except (ValueError, TypeError) as e:
                    logger.debug("Could not parse quote timestamp for %s: %s", sym, e)

            bid = _safe_float(getattr(quote, "bid_price", None)) if quote else 0.0
            ask = _safe_float(getattr(quote, "ask_price", None)) if quote else 0.0
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0

            delta = _safe_float(getattr(greeks, "delta", None)) if greeks else 0.0
            iv = _safe_float(getattr(snapshot, "implied_volatility", None))

            oi = int(_safe_float(getattr(meta, "open_interest", 0))) if meta else 0
            strike = _safe_float(getattr(meta, "strike_price", None)) if meta else 0.0
            expiry_raw = getattr(meta, "expiration_date", None)
            if expiry_raw is None:
                continue
            expiry = expiry_raw if isinstance(expiry_raw, date) else date.fromisoformat(str(expiry_raw))

            if strike <= 0 or mid <= 0:
                continue

            # Filters
            if delta < self._config.safety.min_delta:
                continue
            if oi < self._config.safety.min_open_interest:
                continue

            spread_pct = (ask - bid) / mid if mid > 0 else float("inf")
            if spread_pct > self._config.safety.max_bid_ask_spread_pct:
                logger.debug("Skipping %s: spread %.2f%% > max %.2f%%", sym, spread_pct * 100, self._config.safety.max_bid_ask_spread_pct * 100)
                continue

            # Black-Scholes theoretical price
            T = (expiry - date.today()).days / 365.0
            theo = black_scholes_call(
                S=spot,
                K=strike,
                T=T,
                r=self._rates.risk_free_rate,
                q=self._rates.dividend_yield,
                sigma=iv if iv > 0 else 0.20,
            )

            # Extrinsic check
            intrinsic = intrinsic_value(spot, strike)
            extrinsic = mid - intrinsic
            if theo > 0 and extrinsic / theo > self._config.pricing.max_extrinsic_pct:
                logger.debug("Skipping %s: extrinsic %.2f%% of theo > max %.2f%%", sym, (extrinsic / theo) * 100, self._config.pricing.max_extrinsic_pct * 100)
                continue

            # Divergence warning
            if theo > 0 and abs(theo - mid) / theo > self._config.pricing.price_divergence_warn_pct:
                logger.warning(
                    "%s: theoretical $%.2f vs market mid $%.2f (%.1f%% divergence)",
                    sym, theo, mid, abs(theo - mid) / theo * 100,
                )

            candidate = ContractCandidate(
                symbol=sym,
                underlying=underlying,
                strike=strike,
                expiry=expiry,
                bid=bid,
                ask=ask,
                mid=mid,
                delta=delta,
                iv=iv,
                open_interest=oi,
                theoretical_price=theo,
            )
            candidate.score = self._score(candidate, target_strike, spot)
            candidates.append(candidate)

        logger.info("Found %d viable candidates after filtering", len(candidates))
        return candidates

    def _score(self, c: ContractCandidate, target_strike: float, spot: float) -> float:
        # Proximity to target strike (closer = better)
        strike_diff = abs(c.strike - target_strike) / spot
        strike_score = max(0, 1.0 - strike_diff * 20)

        # Longer expiry preferred (within window)
        days_to_expiry = (c.expiry - date.today()).days
        expiry_score = min(1.0, days_to_expiry / (self._config.strategy.max_expiry_months * 30))

        # Tighter spread = better
        spread_score = max(0, 1.0 - c.spread_pct * 10)

        # Higher delta = more stock-like
        delta_score = min(1.0, c.delta / 0.95)

        # Higher OI = better liquidity
        oi_score = min(1.0, c.open_interest / 1000.0)

        return (
            strike_score * 0.30
            + expiry_score * 0.25
            + spread_score * 0.20
            + delta_score * 0.15
            + oi_score * 0.10
        )

    def calculate_limit_price(self, candidate: ContractCandidate, side: str) -> float:
        theo = candidate.theoretical_price
        mid = candidate.mid
        offset_pct = self._config.strategy.limit_offset_pct

        if side == "buy":
            base = min(theo, mid) if mid > 0 else theo
            return round(base * (1 + offset_pct), 2)
        else:  # sell
            base = max(theo, mid) if mid > 0 else theo
            return round(base * (1 - offset_pct), 2)
