"""Unified price store for CEX prices and Polymarket odds."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field

from popoly.types import MarketOdds


@dataclass
class _PriceEntry:
    """Internal wrapper that pairs a price with its observation timestamp."""

    price: float
    timestamp: float  # monotonic seconds


@dataclass
class _OddsEntry:
    """Internal wrapper that pairs MarketOdds with its observation timestamp."""

    odds: MarketOdds
    timestamp: float  # monotonic seconds


class PriceCache:
    """Thread-safe, async-safe store for the latest prices and market odds.

    All public mutators and accessors acquire an ``asyncio.Lock`` so the
    cache can be shared safely across concurrent tasks.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._prices: dict[str, _PriceEntry] = {}
        self._odds: dict[str, _OddsEntry] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    async def update_price(self, asset: str, price: float) -> None:
        """Store (or overwrite) the latest CEX price for *asset*.

        Args:
            asset: Asset identifier, e.g. ``"BTC"`` or ``"ETH"``.
            price: Latest trade price in USD.
        """
        async with self._lock:
            self._prices[asset] = _PriceEntry(
                price=price,
                timestamp=time.monotonic(),
            )

    async def update_odds(self, key: str, odds: MarketOdds) -> None:
        """Store (or overwrite) the latest Polymarket odds under *key*.

        Args:
            key: A descriptive key, e.g. ``"BTC_UP_5m"``.
            odds: The :class:`~popoly.types.MarketOdds` snapshot.
        """
        async with self._lock:
            self._odds[key] = _OddsEntry(
                odds=odds,
                timestamp=time.monotonic(),
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    async def get_price(self, asset: str) -> tuple[float, float] | None:
        """Return ``(price, timestamp)`` for *asset*, or ``None`` if absent.

        The timestamp is in :func:`time.monotonic` seconds.
        """
        async with self._lock:
            entry = self._prices.get(asset)
            if entry is None:
                return None
            return entry.price, entry.timestamp

    async def get_odds(self, key: str) -> MarketOdds | None:
        """Return the latest :class:`MarketOdds` for *key*, or ``None``."""
        async with self._lock:
            entry = self._odds.get(key)
            if entry is None:
                return None
            return entry.odds

    async def snapshot(self) -> dict:
        """Return an immutable deep copy of all cached data.

        The returned dict has two top-level keys:

        * ``"prices"`` -- ``{asset: {"price": float, "timestamp": float}}``
        * ``"odds"`` -- ``{key: MarketOdds}``
        """
        async with self._lock:
            prices_snapshot = {
                asset: {"price": e.price, "timestamp": e.timestamp}
                for asset, e in self._prices.items()
            }
            odds_snapshot = {
                key: copy.deepcopy(e.odds) for key, e in self._odds.items()
            }
        return {"prices": prices_snapshot, "odds": odds_snapshot}

    async def is_stale(self, key: str, max_age: float = 10.0) -> bool:
        """Check whether the entry identified by *key* is older than *max_age* seconds.

        The key is looked up first in the prices dict, then in the odds
        dict.  If the key is not found at all the data is considered stale.

        Args:
            key: Asset name (for prices) or odds key.
            max_age: Maximum acceptable age in seconds (default ``10.0``).
        """
        now = time.monotonic()
        async with self._lock:
            if key in self._prices:
                return (now - self._prices[key].timestamp) > max_age
            if key in self._odds:
                return (now - self._odds[key].timestamp) > max_age
        # Key not found -- treat as stale.
        return True
