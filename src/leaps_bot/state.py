from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Type, TypeVar

from leaps_bot.models import AllocationRecord, PendingOrderRecord, PositionRecord

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("data/state.json")

T = TypeVar("T")


def _build_record(cls: Type[T], data: dict[str, Any]) -> T:
    """Construct a dataclass from a dict, tolerating missing or extra fields.

    Missing fields fall back to the dataclass default. Extra fields are silently
    dropped. This makes state files forward- and backward-compatible across
    schema migrations (e.g., the `intent` field added to PendingOrderRecord).
    """
    field_defaults = {}
    for f in fields(cls):
        if f.default is not f.default_factory:
            # has a default value
            if f.default is not __import__("dataclasses").MISSING:
                field_defaults[f.name] = f.default
            elif f.default_factory is not __import__("dataclasses").MISSING:  # type: ignore[misc]
                field_defaults[f.name] = f.default_factory()  # type: ignore[misc]

    valid = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in valid}

    # Special-case migration: pre-hardening PendingOrderRecord lacked `intent`.
    # Infer a sensible value so old in-flight orders don't crash on load.
    if cls is PendingOrderRecord and "intent" not in kwargs:
        action = kwargs.get("action", "")
        kwargs["intent"] = "open" if action == "buy" else "close"
        logger.info(
            "Migrating legacy pending order %s: inferred intent=%s",
            kwargs.get("order_id", "?"), kwargs["intent"],
        )

    return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class BotState:
    positions: list[PositionRecord] = field(default_factory=list)
    allocations: list[AllocationRecord] = field(default_factory=list)
    pending_orders: list[PendingOrderRecord] = field(default_factory=list)
    last_run: str | None = None
    last_trade_date: str | None = None

    def save(self, path: Path = DEFAULT_STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("State saved to %s", path)

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_PATH) -> BotState:
        if not path.exists():
            logger.info("No state file found at %s, starting fresh", path)
            return cls()

        with open(path) as f:
            data = json.load(f)

        positions = [_build_record(PositionRecord, p) for p in data.get("positions", [])]
        allocations = [_build_record(AllocationRecord, a) for a in data.get("allocations", [])]
        pending_orders = [_build_record(PendingOrderRecord, o) for o in data.get("pending_orders", [])]

        return cls(
            positions=positions,
            allocations=allocations,
            pending_orders=pending_orders,
            last_run=data.get("last_run"),
            last_trade_date=data.get("last_trade_date"),
        )

    def get_position(self, option_symbol: str) -> PositionRecord | None:
        for p in self.positions:
            if p.option_symbol == option_symbol:
                return p
        return None

    def add_position(self, record: PositionRecord) -> None:
        self.positions.append(record)

    def remove_position(self, option_symbol: str) -> None:
        self.positions = [p for p in self.positions if p.option_symbol != option_symbol]

    def current_quarter_key(self, today_iso: str) -> str:
        year = today_iso[:4]
        month = int(today_iso[5:7])
        q = (month - 1) // 3 + 1
        return f"{year}-Q{q}"

    def has_allocated_this_quarter(self, today_iso: str) -> bool:
        key = self.current_quarter_key(today_iso)
        if any(a.quarter == key for a in self.allocations):
            return True
        # Also block if there's a pending allocation order for this quarter
        return any(o.intent == "allocate" and o.quarter == key for o in self.pending_orders)

    def has_pending_roll(self, option_symbol: str) -> bool:
        return any(
            o.intent == "roll" and o.option_symbol == option_symbol and o.action == "sell"
            for o in self.pending_orders
        )

    def record_allocation(self, record: AllocationRecord) -> None:
        self.allocations.append(record)

    def add_pending_order(self, record: PendingOrderRecord) -> None:
        self.pending_orders.append(record)

    def remove_pending_order(self, order_id: str) -> None:
        self.pending_orders = [o for o in self.pending_orders if o.order_id != order_id]
