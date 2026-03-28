"""Paper-trading simulator for Popoly."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from popoly.types import (
    MarketOdds,
    Side,
    TradeIntent,
    TradeRecord,
    TradeStatus,
)

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates order fills against current market prices.

    Every trade produced by this class has ``paper=True`` and is filled
    at the prevailing market price taken from the snapshot.
    """

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        intent: TradeIntent,
        snapshot: dict[str, float],
    ) -> TradeRecord:
        """Simulate filling *intent* at the current market price.

        Parameters
        ----------
        intent:
            Sized, confidence-scored trade ready for execution.
        snapshot:
            Mapping that must contain at least ``"best_bid"`` and
            ``"best_ask"`` keys (floats).  The midpoint is used as the
            paper fill price.

        Returns
        -------
        TradeRecord
            A record with ``paper=True`` and a generated UUID trade id.
        """
        opp = intent.opportunity

        # Use mid-price from snapshot as the paper fill price.
        best_bid = snapshot.get("best_bid", opp.market_prob)
        best_ask = snapshot.get("best_ask", opp.market_prob)
        fill_price = (best_bid + best_ask) / 2.0

        trade = TradeRecord(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            asset=opp.asset,
            direction=opp.direction,
            timeframe=opp.timeframe,
            side=opp.side,
            size_usd=intent.size_usd,
            price=fill_price,
            edge=opp.edge,
            confidence=intent.confidence,
            paper=True,
            status=TradeStatus.OPEN,
        )

        logger.info(
            "[paper] Filled %s %s %s %s — $%.2f @ %.4f (edge %.2f%%)",
            trade.asset,
            trade.direction,
            trade.timeframe,
            trade.side,
            trade.size_usd,
            trade.price,
            trade.edge * 100,
        )
        return trade

    # ------------------------------------------------------------------
    # P&L resolution
    # ------------------------------------------------------------------

    async def resolve_position(
        self,
        trade: TradeRecord,
        current_odds: MarketOdds,
    ) -> float:
        """Compute the paper P&L for a position given current market odds.

        The position is treated as a simple token purchase at
        ``trade.price`` and is marked-to-market at the current best
        bid/ask midpoint on the relevant side.

        Parameters
        ----------
        trade:
            The open paper trade to resolve.
        current_odds:
            Current market odds snapshot used for exit pricing.

        Returns
        -------
        float
            Realised P&L in USD.
        """
        exit_price = (current_odds.best_bid + current_odds.best_ask) / 2.0

        # Tokens held = size_usd / entry_price.
        qty = trade.size_usd / trade.price if trade.price != 0 else 0.0
        proceeds = qty * exit_price
        pnl = proceeds - trade.size_usd

        logger.info(
            "[paper] Resolved %s — entry %.4f  exit %.4f  pnl $%.2f",
            trade.id,
            trade.price,
            exit_price,
            pnl,
        )
        return pnl
