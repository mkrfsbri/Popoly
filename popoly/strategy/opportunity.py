"""Edge detection between CEX prices and Polymarket implied probabilities."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from popoly.types import (
    Asset,
    Direction,
    MarketOdds,
    Opportunity,
    PriceSnapshot,
    Side,
    Timeframe,
)

if TYPE_CHECKING:
    from popoly.config import Config


class OpportunityDetector:
    """Detects mispricings by comparing CEX momentum to Polymarket odds."""

    # Rolling window duration in seconds.
    _WINDOW_SECS: float = 60.0

    def __init__(self, config: Config, *, sensitivity: float = 100.0) -> None:
        self._config = config
        self._sensitivity = sensitivity
        # Per-asset rolling price buffer: asset -> deque[(timestamp, price)]
        self._price_history: dict[Asset, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def record_price(self, snap: PriceSnapshot) -> None:
        """Append a price observation to the rolling window."""
        ts = snap.timestamp.timestamp()
        self._price_history[snap.asset].append((ts, snap.price))

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, snapshot: dict) -> list[Opportunity]:
        """Return opportunities sorted by edge descending.

        *snapshot* is expected to carry:
          - ``"prices"``: ``dict[Asset, PriceSnapshot]`` – latest CEX prices
          - ``"markets"``: ``list[MarketOdds]`` – current Polymarket books
        """
        prices: dict[Asset, PriceSnapshot] = snapshot.get("prices", {})
        markets: list[MarketOdds] = snapshot.get("markets", [])

        # Record latest prices into the rolling window.
        for snap in prices.values():
            self.record_price(snap)

        lag_threshold = self._config.lag_threshold_pp / 100.0
        now = datetime.now(timezone.utc)
        opportunities: list[Opportunity] = []

        for odds in markets:
            momentum = self._momentum(odds.asset)
            if momentum is None:
                continue

            fair_up = 1.0 / (1.0 + math.exp(-self._sensitivity * momentum))
            fair_down = 1.0 - fair_up

            fair_prob = fair_up if odds.direction is Direction.UP else fair_down
            market_prob = odds.yes_prob

            edge = abs(fair_prob - market_prob)
            if edge <= lag_threshold:
                continue

            side = Side.YES if fair_prob > market_prob else Side.NO

            opportunities.append(
                Opportunity(
                    asset=odds.asset,
                    direction=odds.direction,
                    timeframe=odds.timeframe,
                    fair_prob=fair_prob,
                    market_prob=market_prob,
                    edge=edge,
                    side=side,
                    condition_id=odds.condition_id,
                    token_id=odds.token_id,
                    timestamp=now,
                )
            )

        opportunities.sort(key=lambda o: o.edge, reverse=True)
        return opportunities

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _momentum(self, asset: Asset) -> float | None:
        """Compute % price change over the rolling window.

        Returns ``None`` when there is insufficient data.
        """
        buf = self._price_history.get(asset)
        if not buf or len(buf) < 2:
            return None

        now_ts = buf[-1][0]
        cutoff = now_ts - self._WINDOW_SECS

        # Find earliest entry within the window.
        oldest_price: float | None = None
        for ts, price in buf:
            if ts >= cutoff:
                oldest_price = price
                break

        if oldest_price is None or oldest_price == 0.0:
            return None

        latest_price = buf[-1][1]
        return (latest_price - oldest_price) / oldest_price
