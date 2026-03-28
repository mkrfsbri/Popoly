"""Shared domain types for Popoly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


# ======================================================================
# Enums
# ======================================================================


class Asset(StrEnum):
    """Tracked crypto assets."""

    BTC = "BTC"
    ETH = "ETH"


class Direction(StrEnum):
    """Price-movement direction."""

    UP = "UP"
    DOWN = "DOWN"


class Timeframe(StrEnum):
    """Prediction-market timeframe buckets."""

    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"


class Side(StrEnum):
    """Market side (YES / NO token)."""

    YES = "YES"
    NO = "NO"


class TradeStatus(StrEnum):
    """Lifecycle status of a trade."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


# ======================================================================
# Value objects
# ======================================================================


@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    """A single price observation from an exchange feed."""

    asset: Asset
    price: float
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class MarketOdds:
    """Current probability / order-book snapshot for a Polymarket market."""

    asset: Asset
    direction: Direction
    timeframe: Timeframe
    yes_prob: float
    no_prob: float
    best_bid: float
    best_ask: float
    condition_id: str
    token_id: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class Opportunity:
    """A detected edge between the fair probability and market price."""

    asset: Asset
    direction: Direction
    timeframe: Timeframe
    fair_prob: float
    market_prob: float
    edge: float
    side: Side
    condition_id: str
    token_id: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """Aggregated confidence score with per-factor breakdown."""

    score: float  # 0.0 .. 1.0
    breakdown: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """A sized, confidence-scored trade ready for execution."""

    opportunity: Opportunity
    size_usd: float
    confidence: float


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Persisted record of an executed (or paper) trade."""

    id: str
    timestamp: datetime
    asset: Asset
    direction: Direction
    timeframe: Timeframe
    side: Side
    size_usd: float
    price: float
    edge: float
    confidence: float
    paper: bool
    pnl: float | None = None
    status: TradeStatus = TradeStatus.OPEN
