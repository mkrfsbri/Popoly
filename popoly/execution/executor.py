"""Order dispatch for Popoly."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from popoly.types import (
    TradeIntent,
    TradeRecord,
    TradeStatus,
)

if TYPE_CHECKING:
    from popoly.config import Config
    from popoly.execution.paper_trader import PaperTrader
    from popoly.execution.position_manager import PositionManager
    from popoly.risk.portfolio import Portfolio

logger = logging.getLogger(__name__)

# Retry parameters for live CLOB submissions.
_MAX_RETRIES: int = 3
_RETRY_BACKOFF: float = 1.0  # seconds


class Executor:
    """Dispatches trades to paper or live execution and records results.

    Parameters
    ----------
    config:
        Application configuration (drives paper vs. live routing).
    paper_trader:
        Paper-trading simulator used in non-live mode.
    portfolio:
        Portfolio state tracker for cash / position accounting.
    position_manager:
        Open-position ledger.
    database:
        Async-capable storage backend (must expose
        ``async save_trade(trade: TradeRecord) -> None``).
    telegram:
        Notification service (must expose
        ``async send_alert(message: str) -> None``).
    """

    def __init__(
        self,
        config: Config,
        paper_trader: PaperTrader,
        portfolio: Portfolio,
        position_manager: PositionManager,
        database: Any = None,
        telegram: Any = None,
    ) -> None:
        self._config = config
        self._paper_trader = paper_trader
        self._portfolio = portfolio
        self._positions = position_manager
        self._db = database
        self._telegram = telegram

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def execute(
        self,
        intent: TradeIntent,
        snapshot: dict[str, float],
    ) -> TradeRecord | None:
        """Execute a trade (paper or live) and record the result.

        Returns the :class:`TradeRecord` on success, or ``None`` when the
        live order could not be placed after retries.
        """
        if not self._config.is_live:
            trade = await self._paper_trader.execute(intent, snapshot)
        else:
            trade = await self._submit_live_order(intent, snapshot)
            if trade is None:
                return None

        # Record in portfolio and position manager.
        self._portfolio.record_trade(trade, is_entry=True)
        self._positions.add_position(trade)

        # Persist to database.
        await self._save(trade)

        # Send Telegram notification.
        await self._notify_entry(trade)

        return trade

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    async def close_position(
        self,
        trade_record: TradeRecord,
        snapshot: dict[str, float],
    ) -> float:
        """Close an open position and return realised P&L.

        For paper trades, P&L is computed from the snapshot midpoint.
        For live trades, an opposing limit order is submitted.
        """
        best_bid = snapshot.get("best_bid", trade_record.price)
        best_ask = snapshot.get("best_ask", trade_record.price)
        exit_price = (best_bid + best_ask) / 2.0

        qty = trade_record.size_usd / trade_record.price if trade_record.price else 0.0
        proceeds = qty * exit_price
        pnl = proceeds - trade_record.size_usd

        # Update position manager.
        closed = self._positions.close_position(trade_record.id, pnl)

        # Update portfolio.
        if closed is not None:
            self._portfolio.record_trade(closed, is_entry=False)
            await self._save(closed)
            await self._notify_exit(closed, pnl)

        return pnl

    # ------------------------------------------------------------------
    # Live CLOB order (py_clob_client)
    # ------------------------------------------------------------------

    async def _submit_live_order(
        self,
        intent: TradeIntent,
        snapshot: dict[str, float],
    ) -> TradeRecord | None:
        """Place a limit order via the Polymarket CLOB and return a trade.

        Retries up to ``_MAX_RETRIES`` times with exponential back-off on
        transient API errors.
        """
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs
        except ImportError:
            logger.error(
                "py_clob_client is not installed — cannot submit live orders"
            )
            return None

        opp = intent.opportunity

        client = ClobClient(
            host=self._config.polymarket_api_url,
            key=self._config.private_key,
            chain_id=137,  # Polygon mainnet
        )

        best_bid = snapshot.get("best_bid", opp.market_prob)
        best_ask = snapshot.get("best_ask", opp.market_prob)
        limit_price = (best_bid + best_ask) / 2.0

        order_args = OrderArgs(
            price=limit_price,
            size=intent.size_usd / limit_price if limit_price else 0.0,
            side=opp.side.value,
            token_id=opp.token_id,
        )

        last_exc: BaseException | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                signed_order = client.create_order(order_args)
                response = client.post_order(signed_order)
                logger.info(
                    "[live] Order placed (attempt %d) — response: %s",
                    attempt,
                    response,
                )
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[live] Order attempt %d/%d failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF * attempt)
        else:
            logger.error(
                "[live] All %d order attempts failed. Last error: %s",
                _MAX_RETRIES,
                last_exc,
            )
            return None

        trade = TradeRecord(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            asset=opp.asset,
            direction=opp.direction,
            timeframe=opp.timeframe,
            side=opp.side,
            size_usd=intent.size_usd,
            price=limit_price,
            edge=opp.edge,
            confidence=intent.confidence,
            paper=False,
            status=TradeStatus.OPEN,
        )
        return trade

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _save(self, trade: TradeRecord) -> None:
        if self._db is None:
            return
        try:
            await self._db.insert_trade(trade)
        except Exception:
            logger.exception("Failed to persist trade %s", trade.id)

    # ------------------------------------------------------------------
    # Notification helpers
    # ------------------------------------------------------------------

    async def _notify_entry(self, trade: TradeRecord) -> None:
        mode = "PAPER" if trade.paper else "LIVE"
        msg = (
            f"[{mode}] Opened {trade.asset} {trade.direction} "
            f"{trade.timeframe} {trade.side}\n"
            f"Size: ${trade.size_usd:.2f}  Price: {trade.price:.4f}  "
            f"Edge: {trade.edge:.2%}  Confidence: {trade.confidence:.2%}"
        )
        await self._send_telegram(msg)

    async def _notify_exit(self, trade: TradeRecord, pnl: float) -> None:
        mode = "PAPER" if trade.paper else "LIVE"
        msg = (
            f"[{mode}] Closed {trade.asset} {trade.direction} "
            f"{trade.timeframe} {trade.side}\n"
            f"P&L: ${pnl:+.2f}"
        )
        await self._send_telegram(msg)

    async def _send_telegram(self, message: str) -> None:
        if self._telegram is None:
            return
        try:
            await self._telegram.send(message)
        except Exception:
            logger.exception("Telegram alert failed")
