from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


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
