from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum


def now_utc_iso() -> str:
    """Return the current time as a UTC-aware ISO string.

    All bot-internal timestamps go through this helper instead of
    `datetime.now().isoformat()` (which is naive — no tzinfo). Tz-aware
    output means timestamps are correct regardless of the host machine's
    local time, including weird cases like running locally from a non-UTC
    laptop or a runner with TZ misconfigured. `parse_timestamp` then
    decodes them deterministically.
    """
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(ts: str | None) -> datetime:
    """Parse an ISO timestamp string to a timezone-aware UTC datetime.

    Necessary because timestamps in this system come from multiple sources:
    - Modern internal writes: `now_utc_iso()` → tz-aware UTC
    - Legacy internal writes: `datetime.now().isoformat()` → naive (no tzinfo)
    - Broker: Alpaca `filled_at` → tz-aware (often `+00:00` suffix)

    Lexical string comparison across naive and aware ISO strings can misorder
    events that happened at the same instant. This helper normalizes them all
    to UTC datetime so comparisons and sorting are correct.

    Naive strings are interpreted as UTC. This is correct for new writes
    (which use `now_utc_iso`) and approximately correct for legacy writes
    from GitHub Actions runners (which default to UTC). Legacy writes from
    non-UTC local machines would have a small offset, but that's an
    unavoidable cost of pre-fix data — fixed going forward.

    Returns `datetime.min` (UTC) for None or unparseable input — keeps unknown
    timestamps at the very beginning of any sort, which is safe.
    """
    if not ts:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


class Action(Enum):
    HOLD = "HOLD"
    SELL = "SELL"
    EMERGENCY_SELL = "EMERGENCY_SELL"


class OrderSideIntent(Enum):
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_CLOSE = "sell_to_close"


@dataclass
class OptionDetails:
    underlying: str
    expiry: date
    option_type: str  # "C" or "P"
    strike: float

    @staticmethod
    def from_occ_symbol(symbol: str) -> OptionDetails:
        underlying = symbol[:-15]
        date_str = symbol[-15:-9]
        option_type = symbol[-9]
        strike_raw = symbol[-8:]
        return OptionDetails(
            underlying=underlying,
            expiry=datetime.strptime(date_str, "%y%m%d").date(),
            option_type=option_type,
            strike=int(strike_raw) / 1000.0,
        )


@dataclass
class ContractCandidate:
    symbol: str
    underlying: str
    strike: float
    expiry: date
    bid: float
    ask: float
    mid: float
    delta: float
    iv: float
    open_interest: int
    theoretical_price: float
    score: float = 0.0

    @property
    def spread_pct(self) -> float:
        if self.mid <= 0:
            return float("inf")
        return (self.ask - self.bid) / self.mid

    @property
    def extrinsic(self) -> float:
        return max(0, self.mid - self.intrinsic)

    @property
    def intrinsic(self) -> float:
        return 0.0  # set externally based on underlying price


@dataclass
class PositionAction:
    action: Action
    option_symbol: str
    qty: int
    days_remaining: int
    original_dte: int
    sell_threshold_days: int
    reason: str


@dataclass
class OrderResult:
    success: bool
    order_id: str | None
    symbol: str
    side: str
    qty: int
    price: float | None
    message: str
    dry_run: bool = False


@dataclass
class PositionRecord:
    option_symbol: str
    underlying: str
    strike: float
    expiry_date: str
    purchase_date: str
    original_dte: int
    qty: int
    avg_entry_price: float
    order_id: str


@dataclass
class AllocationRecord:
    quarter: str
    allocated_date: str
    amount: float
    contracts_bought: list[str] = field(default_factory=list)


@dataclass
class PendingOrderRecord:
    order_id: str
    action: str  # "buy" or "sell"
    intent: str  # "open", "close", "roll", "allocate"
    option_symbol: str
    qty: int
    submitted_at: str
    underlying: str = ""  # for roll intent: which underlying to find replacement on
    quarter: str = ""     # for allocate intent: which quarter this allocation belongs to
    recorded_qty: int = 0  # how much of the order has already been reflected in state


@dataclass
class FollowupAction:
    """An action queued by reconciliation that the next trading run will execute.

    Used so monitor passes can reconcile fills (record trades, remove positions,
    keep state accurate) without violating the safety invariant that monitor
    never originates orders. The morning workflow drains this queue before
    running its normal trade evaluation.

    Currently the only kind is "roll" — a replacement-LEAPs buy queued after
    a sell-with-roll-intent fills.
    """
    action_type: str  # "roll" for now; future kinds would extend this
    underlying: str
    qty: int
    sourced_from_order_id: str   # the sell order whose fill triggered this
    queued_at: str               # ISO timestamp when queued


@dataclass
class TradeRecord:
    """A single fill (or fill increment) — the canonical record of a transaction.

    Multiple TradeRecords per order are possible if the order fills incrementally;
    each represents a discrete fill with its own qty and price.
    """
    timestamp: str        # ISO datetime when fill was recorded
    order_id: str
    action: str           # "buy" or "sell"
    intent: str           # "open", "close", "roll", "allocate"
    symbol: str           # OCC option symbol
    underlying: str
    strike: float
    expiry: str           # ISO date
    qty: int              # contracts in this fill increment
    fill_price: float     # avg fill price for this increment
    total_value: float    # qty * fill_price * 100 (option multiplier)
    underlying_price: float | None = None  # spot at time of fill (for benchmark)
    # P&L info, populated only for sells:
    avg_entry_price: float | None = None    # avg entry price of the closed contracts
    realized_pnl: float | None = None       # (fill_price - entry) * qty * 100
    holding_days: int | None = None         # days from purchase to this sell


@dataclass
class PositionSnapshot:
    symbol: str
    underlying: str
    strike: float
    expiry: str
    qty: int
    avg_entry_price: float
    current_price: float       # mark price per contract (per-share, x100 for total)
    market_value: float        # current market value
    cost_basis: float          # total entry cost
    unrealized_pl: float
    unrealized_plpc: float     # as decimal (0.10 = +10%)
    days_remaining: int


@dataclass
class DailySnapshot:
    """End-of-run portfolio snapshot for time-series reporting."""
    date: str                  # YYYY-MM-DD
    timestamp: str             # ISO datetime
    cash: float
    options_buying_power: float
    portfolio_value: float     # account.portfolio_value (cash + market value)
    positions_market_value: float
    num_positions: int
    underlying_prices: dict[str, float] = field(default_factory=dict)  # {"SPY": 550.12}
    positions: list[PositionSnapshot] = field(default_factory=list)


@dataclass
class RunRecord:
    """Activity record for one bot invocation, for run history/debugging."""
    timestamp: str
    duration_seconds: float
    skipped: bool
    skip_reason: str | None = None
    actions: list[str] = field(default_factory=list)  # short summary strings
    errors: list[str] = field(default_factory=list)
    real_trades_today: bool = False
    portfolio_value: float | None = None
    run_type: str = "trade"    # "trade" (morning run) or "monitor" (mid-day/EOD)
