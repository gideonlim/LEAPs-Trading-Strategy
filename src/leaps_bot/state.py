from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Type, TypeVar

from leaps_bot.models import (
    AllocationRecord,
    DailySnapshot,
    FollowupAction,
    PendingOrderRecord,
    PositionRecord,
    PositionSnapshot,
    RunRecord,
    TradeRecord,
)

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
        quarter = kwargs.get("quarter", "") or ""
        if action == "buy":
            # If quarter is set, this was an allocation buy — preserve that
            # intent so the quarter remains marked allocated (otherwise the
            # bot might re-deploy this quarter after upgrade).
            kwargs["intent"] = "allocate" if quarter else "open"
        else:
            kwargs["intent"] = "close"
        logger.info(
            "Migrating legacy pending order %s: inferred intent=%s (action=%s, quarter=%r)",
            kwargs.get("order_id", "?"), kwargs["intent"], action, quarter,
        )

    return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class BotState:
    positions: list[PositionRecord] = field(default_factory=list)
    allocations: list[AllocationRecord] = field(default_factory=list)
    pending_orders: list[PendingOrderRecord] = field(default_factory=list)
    last_run: str | None = None
    last_trade_date: str | None = None
    # Reporting history (append-only):
    trades: list[TradeRecord] = field(default_factory=list)
    snapshots: list[DailySnapshot] = field(default_factory=list)
    runs: list[RunRecord] = field(default_factory=list)
    # Actions queued by monitor reconciliation, drained by next trading run.
    # Keeps the monitor invariant clean: monitor reconciles, run() acts.
    pending_followups: list[FollowupAction] = field(default_factory=list)

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
        trades = [_build_record(TradeRecord, t) for t in data.get("trades", [])]
        runs = [_build_record(RunRecord, r) for r in data.get("runs", [])]
        pending_followups = [_build_record(FollowupAction, f) for f in data.get("pending_followups", [])]

        # DailySnapshot has nested PositionSnapshot list — rebuild explicitly
        snapshots = []
        for s in data.get("snapshots", []):
            nested_positions = s.get("positions", [])
            s_copy = {k: v for k, v in s.items() if k != "positions"}
            snap = _build_record(DailySnapshot, s_copy)
            snap.positions = [_build_record(PositionSnapshot, p) for p in nested_positions]
            snapshots.append(snap)

        return cls(
            positions=positions,
            allocations=allocations,
            pending_orders=pending_orders,
            last_run=data.get("last_run"),
            last_trade_date=data.get("last_trade_date"),
            trades=trades,
            snapshots=snapshots,
            runs=runs,
            pending_followups=pending_followups,
        )

    # -- Reporting log accessors --

    def add_trade(self, trade: TradeRecord) -> None:
        self.trades.append(trade)

    def add_snapshot(self, snapshot: DailySnapshot) -> None:
        self.snapshots.append(snapshot)

    def add_run(self, run: RunRecord) -> None:
        self.runs.append(run)

    def get_trades_by_symbol(self, symbol: str) -> list[TradeRecord]:
        return [t for t in self.trades if t.symbol == symbol]

    def get_realized_pnl(self) -> float:
        return sum(t.realized_pnl or 0.0 for t in self.trades if t.action == "sell")

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
