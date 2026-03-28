"""Position tracking with auto-merge support for Popoly."""

from __future__ import annotations

import logging
from dataclasses import replace

from popoly.types import TradeRecord, TradeStatus

logger = logging.getLogger(__name__)


class PositionManager:
    """In-memory ledger of all open (and recently closed) positions.

    Supports **auto-merge**: when a new trade targets the same market
    (same ``market_key``) as an existing open position on the same side,
    the two are merged into a single position with a weighted-average
    entry price.
    """

    def __init__(self) -> None:
        # trade_id -> TradeRecord
        self._positions: dict[str, TradeRecord] = {}
        # market_key -> trade_id  (index for fast merge lookups)
        self._market_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_position(self, trade: TradeRecord) -> TradeRecord:
        """Register a new position, merging with an existing one if possible.

        Returns the resulting :class:`TradeRecord` (either the original
        trade, or the merged position).
        """
        existing_id = self._market_index.get(trade.market_key)
        existing = self._positions.get(existing_id) if existing_id else None

        if existing is not None and existing.status == TradeStatus.OPEN:
            merged = self._merge(existing, trade)
            self._positions[merged.id] = merged
            logger.info(
                "Merged position %s + %s -> %s | %s size=$%.2f avg_price=%.4f",
                existing.id[:8],
                trade.id[:8],
                merged.id[:8],
                merged.market_key,
                merged.size_usd,
                merged.price,
            )
            return merged

        # No merge — register as new.
        self._positions[trade.id] = trade
        self._market_index[trade.market_key] = trade.id
        logger.info(
            "Tracking position %s  %s %s $%.2f",
            trade.id[:8],
            trade.asset,
            trade.side,
            trade.size_usd,
        )
        return trade

    def close_position(
        self,
        trade_id: str,
        pnl: float,
        *,
        status: TradeStatus = TradeStatus.CLOSED,
    ) -> TradeRecord | None:
        """Mark a position as closed/claimed and attach its realised P&L.

        Returns the updated :class:`TradeRecord`, or ``None`` if the
        *trade_id* was not found.
        """
        trade = self._positions.pop(trade_id, None)
        if trade is None:
            logger.warning("Cannot close unknown position %s", trade_id)
            return None

        # Remove market index entry.
        if self._market_index.get(trade.market_key) == trade_id:
            del self._market_index[trade.market_key]

        closed = replace(trade, status=status, pnl=pnl)
        logger.info(
            "Closed position %s [%s] — pnl $%.2f",
            trade_id[:8],
            status.value,
            pnl,
        )
        return closed

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[TradeRecord]:
        """Return a list of all currently open positions."""
        return [
            t for t in self._positions.values()
            if t.status == TradeStatus.OPEN
        ]

    def total_exposure(self) -> float:
        """Sum of ``size_usd`` across all open positions."""
        return sum(t.size_usd for t in self.get_open_positions())

    def has_position(self, condition_id: str) -> bool:
        """Return ``True`` if there is an open position on *condition_id*.

        The condition id is reconstructed from the trade's asset,
        direction, and timeframe fields (matching the key scheme used by
        the portfolio module).
        """
        for trade in self._positions.values():
            key = f"{trade.asset}_{trade.direction}_{trade.timeframe}"
            if key == condition_id:
                return True
        return False

    def has_position_for_market(self, market_key: str) -> bool:
        """Return ``True`` if an open position exists for *market_key*."""
        tid = self._market_index.get(market_key)
        if tid is None:
            return False
        pos = self._positions.get(tid)
        return pos is not None and pos.status == TradeStatus.OPEN

    def get_position(self, trade_id: str) -> TradeRecord | None:
        """Look up a single position by trade id."""
        return self._positions.get(trade_id)

    def get_position_by_market(self, market_key: str) -> TradeRecord | None:
        """Look up the open position for a given *market_key*."""
        tid = self._market_index.get(market_key)
        if tid is None:
            return None
        return self._positions.get(tid)

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(existing: TradeRecord, incoming: TradeRecord) -> TradeRecord:
        """Combine two positions on the same market into one.

        The resulting position keeps the *existing* trade's id and uses a
        weighted-average entry price.  Edge and confidence are averaged
        weighted by size.
        """
        total_size = existing.size_usd + incoming.size_usd
        if total_size == 0:
            avg_price = existing.price
            avg_edge = existing.edge
            avg_conf = existing.confidence
        else:
            avg_price = (
                existing.price * existing.size_usd
                + incoming.price * incoming.size_usd
            ) / total_size
            avg_edge = (
                existing.edge * existing.size_usd
                + incoming.edge * incoming.size_usd
            ) / total_size
            avg_conf = (
                existing.confidence * existing.size_usd
                + incoming.confidence * incoming.size_usd
            ) / total_size

        return replace(
            existing,
            size_usd=total_size,
            price=avg_price,
            edge=avg_edge,
            confidence=avg_conf,
            # Keep the most recent timestamp
            timestamp=max(existing.timestamp, incoming.timestamp),
        )
