"""Position tracking for Popoly."""

from __future__ import annotations

import logging
from dataclasses import replace

from popoly.types import TradeRecord, TradeStatus

logger = logging.getLogger(__name__)


class PositionManager:
    """In-memory ledger of all open (and recently closed) positions.

    Positions are keyed by their ``trade.id`` and can be queried by
    condition id for duplicate-detection.
    """

    def __init__(self) -> None:
        self._positions: dict[str, TradeRecord] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_position(self, trade: TradeRecord) -> None:
        """Register a new open position."""
        if trade.id in self._positions:
            logger.warning("Position %s already tracked — skipping", trade.id)
            return
        self._positions[trade.id] = trade
        logger.info(
            "Tracking position %s  %s %s $%.2f",
            trade.id[:8],
            trade.asset,
            trade.side,
            trade.size_usd,
        )

    def close_position(
        self,
        trade_id: str,
        pnl: float,
    ) -> TradeRecord | None:
        """Mark a position as closed and attach its realised P&L.

        Returns the updated :class:`TradeRecord`, or ``None`` if the
        *trade_id* was not found.
        """
        trade = self._positions.pop(trade_id, None)
        if trade is None:
            logger.warning("Cannot close unknown position %s", trade_id)
            return None

        closed = replace(trade, status=TradeStatus.CLOSED, pnl=pnl)
        logger.info(
            "Closed position %s — pnl $%.2f",
            trade_id[:8],
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

    def get_position(self, trade_id: str) -> TradeRecord | None:
        """Look up a single position by trade id."""
        return self._positions.get(trade_id)
