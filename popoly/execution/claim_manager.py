"""Auto-merge and auto-claim for Popoly.

Two distinct operations run in a single background loop:

1. **CTF Merge** — when the bot holds both YES and NO tokens on the
   same condition, it calls ``mergePositions()`` on the CTF contract to
   burn the pair and reclaim USDC.e collateral.  For every 1 YES + 1 NO
   token merged, $1 USDC.e is returned.

2. **Claim Winning Positions** — when a market resolves, winning tokens
   are redeemed via ``redeemPositions()`` at $1.00 each.  Losing tokens
   are worth $0 and the position is settled as a loss.

In paper mode both operations are simulated locally.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from popoly.types import (
    MarketOdds,
    MarketOutcome,
    Side,
    TradeRecord,
    TradeStatus,
)

if TYPE_CHECKING:
    from popoly.config import Config
    from popoly.execution.ctf import CTFOperator
    from popoly.execution.position_manager import PositionManager
    from popoly.feeds.price_cache import PriceCache
    from popoly.risk.portfolio import Portfolio

logger = logging.getLogger(__name__)

# Background poll interval.
_POLL_INTERVAL: float = 5.0


class ClaimManager:
    """Background service for CTF merges and winning-position claims.

    Parameters
    ----------
    config:
        Application config (drives paper vs. live routing).
    position_manager:
        Open-position ledger.
    portfolio:
        Portfolio state tracker.
    price_cache:
        Shared price/odds cache.
    ctf:
        On-chain CTF operator (only needed in live mode).
    database:
        Async storage backend.
    telegram:
        Notification service.
    """

    def __init__(
        self,
        config: Config,
        position_manager: PositionManager,
        portfolio: Portfolio,
        price_cache: PriceCache,
        ctf: CTFOperator | None = None,
        database: Any = None,
        telegram: Any = None,
    ) -> None:
        self._config = config
        self._positions = position_manager
        self._portfolio = portfolio
        self._price_cache = price_cache
        self._ctf = ctf
        self._db = database
        self._telegram = telegram
        # Prevent double-processing.
        self._processed_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Poll continuously for merge opportunities and resolved markets."""
        logger.info("ClaimManager started (poll every %.0fs)", _POLL_INTERVAL)
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                logger.info("ClaimManager cancelled")
                raise
            except Exception:
                logger.exception("Error in ClaimManager loop")
            await asyncio.sleep(_POLL_INTERVAL)

    async def _tick(self) -> None:
        """Single iteration: merge pairs first, then claim resolved."""
        await self._auto_merge()
        await self._auto_claim()

    # ==================================================================
    # 1. CTF Merge — YES + NO → USDC.e
    # ==================================================================

    async def _auto_merge(self) -> None:
        """Detect YES+NO pairs on the same condition and merge them."""
        pairs = self._positions.find_mergeable_pairs()
        if not pairs:
            return

        for yes_pos, no_pos in pairs:
            if yes_pos.id in self._processed_ids or no_pos.id in self._processed_ids:
                continue
            await self._merge_pair(yes_pos, no_pos)

    async def _merge_pair(
        self,
        yes_pos: TradeRecord,
        no_pos: TradeRecord,
    ) -> None:
        """Merge one YES+NO pair back into USDC.e collateral.

        The mergeable amount is the *minimum* token quantity held on
        each side.  After the merge:
        - The smaller position is fully closed.
        - The larger position is reduced by the merged quantity.
        - The reclaimed USDC.e is credited back to cash.
        """
        yes_qty = yes_pos.size_usd / yes_pos.price if yes_pos.price > 0 else 0.0
        no_qty = no_pos.size_usd / no_pos.price if no_pos.price > 0 else 0.0
        merge_qty = min(yes_qty, no_qty)

        if merge_qty <= 0:
            return

        # The USDC.e collateral reclaimed (1 full set = $1).
        collateral_returned = merge_qty  # in USDC.e

        # Resolve which condition_id to use.
        condition_id = yes_pos.condition_id or no_pos.condition_id

        logger.info(
            "CTF merge: %s %s %s — %.2f YES + %.2f NO → $%.2f USDC.e",
            yes_pos.asset,
            yes_pos.direction,
            yes_pos.timeframe,
            merge_qty,
            merge_qty,
            collateral_returned,
        )

        # Execute on-chain (live) or simulate (paper).
        if self._config.is_live:
            if self._ctf is None:
                logger.error("CTFOperator not configured — cannot merge on-chain")
                return
            try:
                await self._ctf.merge_positions(
                    condition_id=condition_id,
                    amount=int(merge_qty * 1e6),  # USDC.e has 6 decimals
                )
            except Exception:
                logger.exception("On-chain merge failed — will retry")
                return  # Retry next cycle.
        else:
            logger.info("[paper] Simulated CTF merge of %.2f full sets", merge_qty)

        # Cost basis of the merged portion on each side.
        yes_merged_cost = merge_qty * yes_pos.price
        no_merged_cost = merge_qty * no_pos.price
        total_cost = yes_merged_cost + no_merged_cost

        # P&L: collateral back minus what we paid for both sides.
        merge_pnl = collateral_returned - total_cost

        # Update positions.
        # Close the side that was fully consumed; reduce the other.
        if yes_qty <= no_qty:
            # YES fully consumed.
            self._positions.close_position(
                yes_pos.id, pnl=merge_pnl / 2, status=TradeStatus.MERGED,
            )
            self._reduce_position(no_pos, merge_qty)
        else:
            # NO fully consumed.
            self._positions.close_position(
                no_pos.id, pnl=merge_pnl / 2, status=TradeStatus.MERGED,
            )
            self._reduce_position(yes_pos, merge_qty)

        # Credit the collateral back to cash.
        self._portfolio.cash += collateral_returned

        # Persist.
        await self._persist_update(yes_pos.id, merge_pnl / 2, TradeStatus.MERGED)
        await self._persist_update(no_pos.id, merge_pnl / 2, TradeStatus.MERGED)

        self._processed_ids.add(yes_pos.id)
        self._processed_ids.add(no_pos.id)

        # Notify.
        await self._notify_merge(yes_pos, no_pos, merge_qty, collateral_returned, merge_pnl)

    def _reduce_position(self, pos: TradeRecord, qty_removed: float) -> None:
        """Shrink an open position by *qty_removed* tokens.

        If the remaining quantity is zero (or near-zero), closes it.
        """
        from dataclasses import replace

        total_qty = pos.size_usd / pos.price if pos.price > 0 else 0.0
        remaining_qty = total_qty - qty_removed

        if remaining_qty <= 1e-8:
            self._positions.close_position(pos.id, pnl=0.0, status=TradeStatus.MERGED)
            return

        new_size = remaining_qty * pos.price
        updated = replace(pos, size_usd=new_size)
        self._positions._positions[pos.id] = updated  # noqa: SLF001

    # ==================================================================
    # 2. Claim Winning Positions — resolved market → redeem
    # ==================================================================

    async def _auto_claim(self) -> None:
        """Check open positions against market resolution and claim."""
        open_positions = self._positions.get_open_positions()
        if not open_positions:
            return

        snapshot = await self._price_cache.snapshot()
        odds_map: dict[str, MarketOdds] = snapshot.get("odds", {})

        for pos in open_positions:
            if pos.id in self._processed_ids:
                continue

            outcome = self._detect_resolution(pos, odds_map)
            if outcome is MarketOutcome.UNRESOLVED:
                continue

            pnl = self._compute_settlement_pnl(pos, outcome)
            await self._settle_position(pos, outcome, pnl)

    def _detect_resolution(
        self,
        pos: TradeRecord,
        odds_map: dict[str, MarketOdds],
    ) -> MarketOutcome:
        """Determine whether the market for *pos* has resolved.

        A market is considered resolved when the YES probability is
        pinned at >= 0.99 (YES won) or <= 0.01 (NO won).
        """
        odds_key = f"{pos.asset}_{pos.direction}_{pos.timeframe}"
        odds = odds_map.get(odds_key)

        if odds is None:
            return MarketOutcome.UNRESOLVED

        if odds.yes_prob >= 0.99 or odds.best_bid >= 0.98:
            return MarketOutcome.YES
        if odds.yes_prob <= 0.01 or odds.best_ask <= 0.02:
            return MarketOutcome.NO

        return MarketOutcome.UNRESOLVED

    @staticmethod
    def _compute_settlement_pnl(
        pos: TradeRecord,
        outcome: MarketOutcome,
    ) -> float:
        """Calculate realised P&L based on the market outcome.

        Winning tokens redeem at $1.00 each.  Losing tokens are worth $0.
        """
        qty = pos.size_usd / pos.price if pos.price > 0 else 0.0
        cost = pos.size_usd

        won = (
            (pos.side is Side.YES and outcome is MarketOutcome.YES)
            or (pos.side is Side.NO and outcome is MarketOutcome.NO)
        )

        if won:
            return qty * 1.0 - cost  # payout - cost
        return -cost  # total loss

    async def _settle_position(
        self,
        pos: TradeRecord,
        outcome: MarketOutcome,
        pnl: float,
    ) -> None:
        """Settle a resolved position via CTF redeem or paper simulation."""
        won = pnl > 0
        label = "WON" if won else "LOST"

        logger.info(
            "Market resolved [%s]: %s %s %s %s — %s pnl=$%.2f",
            outcome.value,
            pos.asset,
            pos.direction,
            pos.timeframe,
            pos.side,
            label,
            pnl,
        )

        # Live: call redeemPositions on-chain for winners.
        if self._config.is_live and won:
            if self._ctf is None:
                logger.error("CTFOperator not configured — cannot redeem")
                return
            try:
                condition_id = pos.condition_id
                await self._ctf.redeem_positions(condition_id=condition_id)
            except Exception:
                logger.exception(
                    "On-chain redeem failed for %s — will retry", pos.id[:8]
                )
                return
        elif not self._config.is_live:
            logger.info("[paper] Simulated redeem for %s", pos.id[:8])

        # Close position.
        closed = self._positions.close_position(
            pos.id, pnl, status=TradeStatus.CLAIMED,
        )
        if closed is not None:
            self._portfolio.record_trade(closed, is_entry=False)

        await self._persist_update(pos.id, pnl, TradeStatus.CLAIMED)
        await self._notify_claim(pos, outcome, pnl)
        self._processed_ids.add(pos.id)

    # ==================================================================
    # Persistence
    # ==================================================================

    async def _persist_update(
        self,
        trade_id: str,
        pnl: float,
        status: TradeStatus,
    ) -> None:
        if self._db is None:
            return
        try:
            await self._db.update_trade(trade_id, pnl=pnl, status=status.value)
        except Exception:
            logger.exception("Failed to persist update for %s", trade_id[:8])

    # ==================================================================
    # Notifications
    # ==================================================================

    async def _notify_merge(
        self,
        yes_pos: TradeRecord,
        no_pos: TradeRecord,
        merge_qty: float,
        collateral: float,
        pnl: float,
    ) -> None:
        if self._telegram is None:
            return
        mode = "PAPER" if yes_pos.paper else "LIVE"
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"\U0001f501 <b>{mode} CTF Merge</b>\n"
            f"Market: {yes_pos.asset} {yes_pos.direction} {yes_pos.timeframe}\n"
            f"Merged: {merge_qty:.2f} YES + {merge_qty:.2f} NO\n"
            f"Collateral returned: <b>${collateral:.2f} USDC.e</b>\n"
            f"YES cost: ${yes_pos.size_usd:.2f} @ {yes_pos.price:.4f}\n"
            f"NO cost: ${no_pos.size_usd:.2f} @ {no_pos.price:.4f}\n"
            f"Merge P&L: <b>{sign}${pnl:.2f}</b>"
        )
        try:
            await self._telegram.send(msg)
        except Exception:
            logger.exception("Telegram merge alert failed")

    async def _notify_claim(
        self,
        pos: TradeRecord,
        outcome: MarketOutcome,
        pnl: float,
    ) -> None:
        if self._telegram is None:
            return
        won = pnl > 0
        mode = "PAPER" if pos.paper else "LIVE"
        emoji = "\u2705" if won else "\u274c"
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"{emoji} <b>{mode} Position {'Claimed' if won else 'Settled'}</b>\n"
            f"Market: {pos.asset} {pos.direction} {pos.timeframe}\n"
            f"Side: {pos.side}  |  Outcome: {outcome.value}\n"
            f"Size: ${pos.size_usd:.2f}  |  Entry: {pos.price:.4f}\n"
            f"P&L: <b>{sign}${pnl:.2f}</b>\n"
            f"ID: <code>{pos.id}</code>"
        )
        try:
            await self._telegram.send(msg)
        except Exception:
            logger.exception("Telegram claim alert failed")
