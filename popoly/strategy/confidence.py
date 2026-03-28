"""Confidence scoring for detected opportunities."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from popoly.types import ConfidenceResult, MarketOdds, Opportunity, PriceSnapshot


class ConfidenceScorer:
    """Produces a 0..1 confidence score from weighted factors."""

    # Factor weights (must sum to 1.0).
    _W_FRESHNESS: float = 0.25
    _W_EDGE: float = 0.30
    _W_DEPTH: float = 0.20
    _W_VOLATILITY: float = 0.25

    # Thresholds.
    _FRESHNESS_LIMIT: float = 5.0  # seconds – full score if both feeds < this
    _EDGE_CAP: float = 0.20  # edge at which score saturates to 1.0
    _IDEAL_MOMENTUM: float = 0.002  # centre of volatility bell curve
    _VOL_SIGMA: float = 0.005  # width of volatility bell curve

    def score(self, opportunity: Opportunity, snapshot: dict) -> ConfidenceResult:
        """Score an *opportunity* given the current data *snapshot*.

        *snapshot* is expected to carry:
          - ``"prices"``: ``dict[Asset, PriceSnapshot]``
          - ``"markets"``: ``list[MarketOdds]``
          - ``"momentum"``: ``dict[Asset, float]`` (optional, % change)
        """
        now = datetime.now(timezone.utc)

        freshness = self._score_freshness(opportunity, snapshot, now)
        edge_mag = self._score_edge(opportunity)
        depth = self._score_depth(opportunity, snapshot)
        vol = self._score_volatility(snapshot, opportunity)

        total = (
            self._W_FRESHNESS * freshness
            + self._W_EDGE * edge_mag
            + self._W_DEPTH * depth
            + self._W_VOLATILITY * vol
        )

        breakdown = {
            "data_freshness": round(freshness, 4),
            "edge_magnitude": round(edge_mag, 4),
            "orderbook_depth": round(depth, 4),
            "volatility": round(vol, 4),
        }

        return ConfidenceResult(
            score=round(min(max(total, 0.0), 1.0), 4),
            breakdown=breakdown,
            timestamp=now,
        )

    # ------------------------------------------------------------------
    # Individual factor scorers
    # ------------------------------------------------------------------

    def _score_freshness(
        self, opp: Opportunity, snapshot: dict, now: datetime
    ) -> float:
        """1.0 if both feeds are < 5 s old, linear decay to 0."""
        prices: dict = snapshot.get("prices", {})
        markets: list[MarketOdds] = snapshot.get("markets", [])

        price_snap: PriceSnapshot | None = prices.get(opp.asset)
        market_snap: MarketOdds | None = None
        for m in markets:
            if (
                m.asset == opp.asset
                and m.direction == opp.direction
                and m.timeframe == opp.timeframe
            ):
                market_snap = m
                break

        ages: list[float] = []
        if price_snap is not None:
            ages.append(abs((now - price_snap.timestamp).total_seconds()))
        if market_snap is not None:
            ages.append(abs((now - market_snap.timestamp).total_seconds()))

        if not ages:
            return 0.0

        worst_age = max(ages)
        if worst_age <= self._FRESHNESS_LIMIT:
            return 1.0
        # Linear decay from 5 s to staleness threshold (default 10 s).
        staleness = max(self._FRESHNESS_LIMIT * 2, 10.0)
        if worst_age >= staleness:
            return 0.0
        return 1.0 - (worst_age - self._FRESHNESS_LIMIT) / (
            staleness - self._FRESHNESS_LIMIT
        )

    def _score_edge(self, opp: Opportunity) -> float:
        """Higher edge -> higher score, capped at ``_EDGE_CAP``."""
        return min(opp.edge / self._EDGE_CAP, 1.0)

    def _score_depth(self, opp: Opportunity, snapshot: dict) -> float:
        """Tighter bid/ask spread -> higher score (proxy for depth)."""
        markets: list[MarketOdds] = snapshot.get("markets", [])
        for m in markets:
            if (
                m.asset == opp.asset
                and m.direction == opp.direction
                and m.timeframe == opp.timeframe
            ):
                spread = abs(m.best_ask - m.best_bid)
                # Spread of 0 -> 1.0, spread of 0.10 -> 0.0.
                return max(1.0 - spread / 0.10, 0.0)
        return 0.0

    def _score_volatility(self, snapshot: dict, opp: Opportunity) -> float:
        """Bell curve: moderate momentum is ideal, extremes score low."""
        momentum_map: dict = snapshot.get("momentum", {})
        momentum = momentum_map.get(opp.asset)
        if momentum is None:
            return 0.5  # neutral when data is unavailable

        abs_momentum = abs(momentum)
        # Gaussian bell curve centred on _IDEAL_MOMENTUM.
        exponent = -((abs_momentum - self._IDEAL_MOMENTUM) ** 2) / (
            2 * self._VOL_SIGMA**2
        )
        return math.exp(exponent)
