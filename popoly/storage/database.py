"""SQLite persistence layer for Popoly."""

from __future__ import annotations

import logging
from typing import Any

import aiosqlite

from popoly.types import TradeRecord, TradeStatus

logger = logging.getLogger(__name__)

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id           TEXT PRIMARY KEY,
    timestamp    REAL    NOT NULL,
    asset        TEXT    NOT NULL,
    direction    TEXT    NOT NULL,
    timeframe    TEXT    NOT NULL,
    side         TEXT    NOT NULL,
    size_usd     REAL    NOT NULL,
    price        REAL    NOT NULL,
    edge         REAL    NOT NULL,
    confidence   REAL    NOT NULL,
    paper        INTEGER NOT NULL,
    condition_id TEXT    NOT NULL DEFAULT '',
    token_id     TEXT    NOT NULL DEFAULT '',
    pnl          REAL,
    status       TEXT    NOT NULL
)
"""

_CREATE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            REAL NOT NULL,
    portfolio_value      REAL NOT NULL,
    cash                 REAL NOT NULL,
    open_positions_value REAL NOT NULL,
    daily_pnl            REAL NOT NULL
)
"""


class Database:
    """Async SQLite store for trades and portfolio snapshots."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Open the database and create tables if they don't exist."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute(_CREATE_TRADES)
        await self._conn.execute(_CREATE_SNAPSHOTS)
        await self._conn.commit()
        logger.info("Database initialised at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialised – call init() first")
        return self._conn

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        return dict(row)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    async def insert_trade(self, trade: TradeRecord) -> None:
        """Insert a new trade record."""
        await self._db.execute(
            """
            INSERT INTO trades
                (id, timestamp, asset, direction, timeframe, side,
                 size_usd, price, edge, confidence, paper,
                 condition_id, token_id, pnl, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.id,
                trade.timestamp.timestamp(),
                str(trade.asset),
                str(trade.direction),
                str(trade.timeframe),
                str(trade.side),
                trade.size_usd,
                trade.price,
                trade.edge,
                trade.confidence,
                int(trade.paper),
                trade.condition_id,
                trade.token_id,
                trade.pnl,
                str(trade.status),
            ),
        )
        await self._db.commit()
        logger.debug("Inserted trade %s", trade.id)

    async def update_trade(self, trade_id: str, pnl: float, status: str) -> None:
        """Update pnl and status for an existing trade."""
        await self._db.execute(
            "UPDATE trades SET pnl = ?, status = ? WHERE id = ?",
            (pnl, status, trade_id),
        )
        await self._db.commit()
        logger.debug("Updated trade %s -> status=%s pnl=%.4f", trade_id, status, pnl)

    async def get_recent_trades(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent *limit* trades ordered by timestamp descending."""
        cursor = await self._db.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_all_trades(self) -> list[dict[str, Any]]:
        """Return every trade ordered by timestamp ascending."""
        cursor = await self._db.execute(
            "SELECT * FROM trades ORDER BY timestamp ASC",
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    async def insert_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Insert a portfolio snapshot.

        Expected keys: timestamp, portfolio_value, cash,
        open_positions_value, daily_pnl.
        """
        await self._db.execute(
            """
            INSERT INTO snapshots
                (timestamp, portfolio_value, cash, open_positions_value, daily_pnl)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot["timestamp"],
                snapshot["portfolio_value"],
                snapshot["cash"],
                snapshot["open_positions_value"],
                snapshot["daily_pnl"],
            ),
        )
        await self._db.commit()
        logger.debug("Inserted portfolio snapshot")

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    async def get_trade_stats(self) -> dict[str, Any]:
        """Return basic trade statistics.

        Returns a dict with keys: win_count, loss_count, total_pnl.
        Only closed trades with a non-null pnl are counted.
        """
        cursor = await self._db.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS win_count,
                COALESCE(SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END), 0) AS loss_count,
                COALESCE(SUM(pnl), 0.0) AS total_pnl
            FROM trades
            WHERE status IN (?, ?) AND pnl IS NOT NULL
            """,
            (str(TradeStatus.CLOSED), str(TradeStatus.CLAIMED)),
        )
        row = await cursor.fetchone()
        if row is None:
            return {"win_count": 0, "loss_count": 0, "total_pnl": 0.0}
        return self._row_to_dict(row)
