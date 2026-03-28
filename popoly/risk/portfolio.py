"""Portfolio state tracking for Popoly."""

from __future__ import annotations

import logging

from popoly.types import TradeRecord, TradeStatus

logger = logging.getLogger(__name__)


class Portfolio:
    """Tracks cash, positions, and portfolio-level metrics.

    Positions are keyed by ``(condition_id, token_id)`` and store the
    outstanding USD size at the entry price.
    """

    def __init__(self, initial_balance: float) -> None:
        self.cash: float = initial_balance
        self.initial_value: float = initial_balance
        self.daily_start_value: float = initial_balance
        # key -> (side, size_usd, entry_price)
        self.positions: dict[tuple[str, str], dict] = {}

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def total_value(self, current_prices: dict[str, float]) -> float:
        """Return cash plus mark-to-market value of all open positions.

        ``current_prices`` maps ``token_id`` to the current token price.
        """
        mtm = 0.0
        for (_cid, tid), pos in self.positions.items():
            token_price = current_prices.get(tid, pos["entry_price"])
            # Number of tokens held = size_usd / entry_price.
            qty = pos["size_usd"] / pos["entry_price"]
            mtm += qty * token_price
        return self.cash + mtm

    def daily_pnl(self, current_prices: dict[str, float]) -> float:
        """Absolute P&L since the start of the trading day."""
        return self.total_value(current_prices) - self.daily_start_value

    def drawdown_pct(self, current_prices: dict[str, float]) -> float:
        """Daily P&L as a fraction of daily start value.

        A negative return value indicates a drawdown.
        """
        if self.daily_start_value == 0:
            return 0.0
        return self.daily_pnl(current_prices) / self.daily_start_value

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def position_pct(
        self, size_usd: float, current_prices: dict[str, float]
    ) -> float:
        """Return *size_usd* as a fraction of total portfolio value."""
        tv = self.total_value(current_prices)
        if tv == 0:
            return 0.0
        return size_usd / tv

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradeRecord, is_entry: bool) -> None:
        """Update cash and positions after a trade execution.

        Parameters
        ----------
        trade:
            The executed trade record.
        is_entry:
            ``True`` when opening a new position, ``False`` when closing.
        """
        key = (trade.market_key, trade.token_id or trade.side)

        if is_entry:
            self.cash -= trade.size_usd
            self.positions[key] = {
                "side": trade.side,
                "size_usd": trade.size_usd,
                "entry_price": trade.price,
                "asset": trade.asset,
                "direction": trade.direction,
                "timeframe": trade.timeframe,
            }
            logger.info(
                "Opened position %s — size $%.2f @ %.4f",
                key,
                trade.size_usd,
                trade.price,
            )
        else:
            pos = self.positions.pop(key, None)
            if pos is not None:
                qty = pos["size_usd"] / pos["entry_price"]
                proceeds = qty * trade.price
                self.cash += proceeds
                pnl = proceeds - pos["size_usd"]
                logger.info(
                    "Closed position %s — proceeds $%.2f  pnl $%.2f",
                    key,
                    proceeds,
                    pnl,
                )
            else:
                # Fallback: just credit the size back.
                self.cash += trade.size_usd
                logger.warning("Closed unknown position %s", key)

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Reset daily tracking values.  Call at the start of each trading day."""
        # Use last known value (positions valued at entry price as fallback).
        fallback_prices: dict[str, float] = {
            tid: pos["entry_price"]
            for (_cid, tid), pos in self.positions.items()
        }
        self.daily_start_value = self.total_value(fallback_prices)
        logger.info("Daily reset — start value $%.2f", self.daily_start_value)
