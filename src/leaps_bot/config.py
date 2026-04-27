from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class StrategyConfig:
    underlying_preference: str = "auto"
    itm_depth_pct: float = 0.20
    min_expiry_months: int = 12
    max_expiry_months: int = 18
    sell_threshold_fraction: float = 0.333
    order_type: str = "limit"
    limit_offset_pct: float = 0.02


@dataclass
class PricingConfig:
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.013
    max_extrinsic_pct: float = 0.25
    price_divergence_warn_pct: float = 0.10


@dataclass
class AllocationConfig:
    quarterly_months: list[int] = field(default_factory=lambda: [3, 6, 9, 12])
    max_cash_deploy_pct: float = 0.90
    min_cash_reserve: float = 500.0
    allocation_window_days: int = 7  # Only allocate within the first N calendar days of a quarter month


@dataclass
class SafetyConfig:
    no_trade_minutes_after_open: int = 60
    max_bid_ask_spread_pct: float = 0.10
    min_open_interest: int = 100
    min_delta: float = 0.80
    emergency_sell_days: int = 30
    stale_quote_warn_minutes: int = 15


@dataclass
class DataConfig:
    feed: str = "indicative"


@dataclass
class AppConfig:
    paper_trading: bool = True
    dry_run: bool = False
    log_level: str = "INFO"
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api_key: str = ""
    secret_key: str = ""


def _build_section(cls, raw: dict | None):
    if raw is None:
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(config_path: str | Path = "config.yaml", env_path: str | Path | None = None) -> AppConfig:
    load_dotenv(env_path or ".env")

    raw: dict = {}
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    cfg = AppConfig(
        paper_trading=raw.get("paper_trading", True),
        dry_run=raw.get("dry_run", False),
        log_level=raw.get("log_level", "INFO"),
        strategy=_build_section(StrategyConfig, raw.get("strategy")),
        pricing=_build_section(PricingConfig, raw.get("pricing")),
        allocation=_build_section(AllocationConfig, raw.get("allocation")),
        safety=_build_section(SafetyConfig, raw.get("safety")),
        data=_build_section(DataConfig, raw.get("data")),
        api_key=os.environ.get("ALPACA_API_KEY", ""),
        secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
    )
    return cfg
