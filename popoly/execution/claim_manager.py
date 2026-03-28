"""Auto-claim winning positions for Popoly.

Monitors open positions, detects when a Polymarket market has resolved,
and automatically claims winnings or marks losses.  For live trading
this calls the Polymarket contract to redeem tokens; for paper trading
it simulates the settlement.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
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
    from popoly.execution.position_manager import PositionManager
    from popoly.feeds.price_cache import PriceCache
    from popoly.risk.portfolio import Portfolio

logger = logging.getLogger(__name__)

# How often to check for resolved markets (seconds).
_CLAIM_POLL_INTERVAL: float = 5.0

# Retry parameters for on-chain redemption.
_MAX_CLAIM_RETRIES: int = 3
_CLAIM_RETRY_BACKOFF: float = 2.0


class ClaimManager:
    """Background service that auto-claims resolved positions.

    For each open position the manager periodically checks the
    Polymarket orderbook / market status.  When a market resolves:

    * **Winning position** — tokens are redeemed at 1.00 (full payout).
      P&L = payout - cost.
    * **Losing position** — tokens are worth 0.00.
      P&L = -cost (total loss of the wagered amount).

    In paper mode no on-chain transaction is needed; the claim is
    simulated locally.
    """

    def __init__(
        self,
        config: Config,
        position_manager: PositionManager,
        portfolio: Portfolio,
        price_cache: PriceCache,
        database: Any = None,
        telegram: Any = None,
    ) -> None:
        self._config = config
        self._positions = position_manager
        self._portfolio = portfolio
        self._price_cache = price_cache
        self._db = database
        self._telegram = telegram
        # Track markets that have already been claimed to avoid double-claims.
        self._claimed_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop — poll open positions and claim resolved ones."""
        logger.info("ClaimManager started (poll every %.0fs)", _CLAIM_POLL_INTERVAL)
        while True:
            try:
                await self._check_and_claim()
            except asyncio.CancelledError:
                logger.info("ClaimManager cancelled — shutting down")
                raise
            except Exception:
                logger.exception("Error in claim loop")
            await asyncio.sleep(_CLAIM_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _check_and_claim(self) -> None:
        """Iterate over open positions, detect resolved markets, claim."""
        open_positions = self._positions.get_open_positions()
        if not open_positions:
            return

        snapshot = await self._price_cache.snapshot()
        odds_map: dict[str, MarketOdds] = snapshot.get("odds", {})

        for pos in open_positions:
            if pos.id in self._claimed_ids:
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
        pinned at >= 0.99 (YES won) or <= 0.01 (NO won).  This handles
        both the case where Polymarket updates the orderbook after
        resolution and the case where the price naturally converges to
        0 or 1 at expiry.
        """
        # Build the odds cache key the same way PolymarketFeed does.
        odds_key = f"{pos.asset}_{pos.direction}_{pos.timeframe}"
        odds = odds_map.get(odds_key)

        if odds is None:
            return MarketOutcome.UNRESOLVED

        # Pinned prices indicate resolution.
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

        Winning YES tokens pay out 1.00 per token, winning NO tokens
        also pay out 1.00.  Losing tokens are worth 0.
        """
        qty = pos.size_usd / pos.price if pos.price > 0 else 0.0
        cost = pos.size_usd

        if pos.side is Side.YES:
            if outcome is MarketOutcome.YES:
                payout = qty * 1.0  # each YES token redeems at $1
                return payout - cost
            else:
                return -cost  # total loss
        else:
            # Side.NO
            if outcome is MarketOutcome.NO:
                payout = qty * 1.0
                return payout - cost
            else:
                return -cost

    async def _settle_position(
        self,
        pos: TradeRecord,
        outcome: MarketOutcome,
        pnl: float,
    ) -> None:
        """Settle a resolved position: claim on-chain (live) or locally (paper)."""
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

        # Live: attempt on-chain redemption for winning positions.
        if self._config.is_live and won:
            success = await self._claim_on_chain(pos)
            if not success:
                logger.error(
                    "On-chain claim failed for %s — will retry next cycle",
                    pos.id[:8],
                )
                return  # Don't mark as claimed; retry next iteration.

        # Close position in position manager.
        closed = self._positions.close_position(
            pos.id, pnl, status=TradeStatus.CLAIMED,
        )

        # Update portfolio.
        if closed is not None:
            self._portfolio.record_trade(closed, is_entry=False)

        # Persist to database.
        await self._persist_claim(pos, pnl)

        # Send notification.
        await self._notify_claim(pos, outcome, pnl)

        # Mark as claimed.
        self._claimed_ids.add(pos.id)

    async def _claim_on_chain(self, pos: TradeRecord) -> bool:
        """Redeem winning tokens via the Polymarket CTF contract.

        Uses py_clob_client to interact with the Conditional Token
        Framework for token redemption on Polygon.
        """
        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            logger.error("py_clob_client not installed — cannot claim on-chain")
            return False

        client = ClobClient(
            host=self._config.polymarket_api_url,
            key=self._config.private_key,
            chain_id=137,
        )

        last_exc: BaseException | None = None
        for attempt in range(1, _MAX_CLAIM_RETRIES + 1):
            try:
                loop = asyncio.get_running_loop()
                # ClobClient.redeem() is the standard method for claiming
                # resolved conditional tokens. Falls back to raw CTF
                # contract call if the client doesn't expose it directly.
                if hasattr(client, "redeem"):
                    result = await loop.run_in_executor(
                        None,
                        lambda: client.redeem(pos.condition_id),  # noqa: B023
                    )
                else:
                    # Fallback: use the lower-level create_and_post approach
                    # to sell the winning tokens at 1.00 on the book.
                    from py_clob_client.clob_types import OrderArgs

                    sell_args = OrderArgs(
                        price=0.99,
                        size=pos.size_usd / pos.price if pos.price > 0 else 0.0,
                        side="SELL",
                        token_id=pos.token_id,
                    )
                    signed = client.create_order(sell_args)
                    result = client.post_order(signed)

                logger.info(
                    "[live] Claim succeeded (attempt %d) for %s: %s",
                    attempt,
                    pos.id[:8],
                    result,
                )
                return True

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[live] Claim attempt %d/%d failed for %s: %s",
                    attempt,
                    _MAX_CLAIM_RETRIES,
                    pos.id[:8],
                    exc,
                )
                if attempt < _MAX_CLAIM_RETRIES:
                    await asyncio.sleep(_CLAIM_RETRY_BACKOFF * attempt)

        logger.error(
            "[live] All %d claim attempts failed for %s. Last error: %s",
            _MAX_CLAIM_RETRIES,
            pos.id[:8],
            last_exc,
        )
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist_claim(self, pos: TradeRecord, pnl: float) -> None:
        if self._db is None:
            return
        try:
            await self._db.update_trade(
                pos.id, pnl=pnl, status=TradeStatus.CLAIMED.value,
            )
        except Exception:
            logger.exception("Failed to persist claim for %s", pos.id[:8])

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

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
