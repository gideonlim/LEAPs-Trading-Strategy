"""AUD/USD exchange rate provider.

Sources daily rates from the Frankfurter public API (https://www.frankfurter.app),
which is free and unauthenticated. Frankfurter's data comes from the ECB —
this differs from the RBA's official 4 PM rate by ~0.1% on most days. For
strict ATO-compliant filing on material amounts, verify against RBA's
published rate; this module is a reasonable approximation that's fine for
typical retail option positions.

Falls back to the closest preceding business day's rate if a specific date
has no published rate (weekends, holidays). Caches results on disk between
runs so repeated exports don't re-fetch the same data.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_FX_CACHE_PATH = Path("data/fx_cache.json")


class FXProvider(ABC):
    """Abstracts AUD/USD rate lookup so callers don't care about the source."""

    @abstractmethod
    def get_rate(self, d: date) -> float | None:
        """AUD per 1 USD on date d. None if unavailable for any reason."""
        ...

    def describe(self) -> str:
        """Short description for logging/CSV headers."""
        return self.__class__.__name__


class FlatFXProvider(FXProvider):
    """Returns the same rate for any date. Use for offline approximation."""

    def __init__(self, rate: float):
        self._rate = rate

    def get_rate(self, d: date) -> float | None:
        return self._rate

    def describe(self) -> str:
        return f"flat rate {self._rate:.4f} AUD/USD"


class FrankfurterFXProvider(FXProvider):
    """Fetches per-date AUD/USD rates from Frankfurter (ECB source).

    Strategy:
    - On first call, fetch a date range that covers all needed dates in one
      HTTP request (Frankfurter supports `start..end` ranges).
    - Cache results on disk so subsequent runs are fast and offline-safe.
    - For weekends/holidays where ECB published no rate, fall back to the
      most recent preceding business day's rate.
    """

    BASE_URL = "https://api.frankfurter.app"

    def __init__(self, cache_path: Path | None = None):
        self._cache_path = cache_path or DEFAULT_FX_CACHE_PATH
        self._cache: dict[str, float] = {}  # ISO date → rate
        self._range_loaded: tuple[date, date] | None = None
        self._load_cache()

    def get_rate(self, d: date) -> float | None:
        # Try the exact date first; if missing, fall back to preceding days
        # (weekends/holidays). Cap the search at 7 days back to avoid
        # silently returning a wildly stale rate.
        for offset in range(0, 8):
            lookup = d - timedelta(days=offset)
            iso = lookup.isoformat()
            if iso in self._cache:
                if offset > 0:
                    logger.debug(
                        "FX: no rate for %s, using %s (%d day(s) earlier)",
                        d, lookup, offset,
                    )
                return self._cache[iso]
        return None

    def prefetch_range(self, start: date, end: date) -> None:
        """Fetch a contiguous date range in a single HTTP request and cache it.

        Idempotent: doesn't refetch dates already cached. Expands the
        requested window slightly to ensure weekend/holiday fallback works
        for dates near the start of the window.
        """
        # Pad start by 7 days so weekend fallbacks at the start of the FY work
        fetch_start = start - timedelta(days=7)
        fetch_end = end

        # Skip if the cache already covers this range
        if self._range_loaded and self._range_loaded[0] <= fetch_start and self._range_loaded[1] >= fetch_end:
            return

        # Don't request future dates — Frankfurter returns the latest available
        today = date.today()
        if fetch_end > today:
            fetch_end = today
        if fetch_start > fetch_end:
            return

        url = f"{self.BASE_URL}/{fetch_start.isoformat()}..{fetch_end.isoformat()}?from=USD&to=AUD"
        try:
            req = Request(url, headers={"User-Agent": "leaps-bot-fx/1.0"})
            with urlopen(req, timeout=15) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError) as e:
            logger.warning("FX: failed to fetch %s: %s", url, e)
            return

        rates = payload.get("rates", {})
        if not rates:
            logger.warning("FX: empty response for %s", url)
            return

        for iso, fx in rates.items():
            aud = fx.get("AUD")
            if aud is not None:
                self._cache[iso] = float(aud)

        self._range_loaded = (fetch_start, fetch_end)
        self._save_cache()
        logger.info(
            "FX: loaded %d daily rates for %s to %s",
            len(rates), fetch_start, fetch_end,
        )

    def describe(self) -> str:
        return "Frankfurter daily AUD/USD (ECB)"

    # -- Cache I/O --

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            self._cache = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
            logger.debug("FX: loaded %d cached rates from %s", len(self._cache), self._cache_path)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("FX: could not load cache from %s: %s", self._cache_path, e)
            self._cache = {}

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=2, sort_keys=True)
        except OSError as e:
            logger.warning("FX: could not save cache to %s: %s", self._cache_path, e)
