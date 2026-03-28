"""Polymarket CLOB poller for BTC/ETH short-timeframe markets."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from py_clob_client.client import ClobClient

from popoly.config import Config
from popoly.feeds.price_cache import PriceCache
from popoly.types import Asset, Direction, MarketOdds, Timeframe
from popoly.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Keywords used to identify the relevant Polymarket markets.
_MARKET_KEYWORDS: list[dict[str, Any]] = [
    {"asset": Asset.BTC, "direction": Direction.UP, "timeframe": Timeframe.FIVE_MIN},
    {"asset": Asset.BTC, "direction": Direction.DOWN, "timeframe": Timeframe.FIVE_MIN},
    {"asset": Asset.BTC, "direction": Direction.UP, "timeframe": Timeframe.FIFTEEN_MIN},
    {"asset": Asset.BTC, "direction": Direction.DOWN, "timeframe": Timeframe.FIFTEEN_MIN},
    {"asset": Asset.ETH, "direction": Direction.UP, "timeframe": Timeframe.FIVE_MIN},
    {"asset": Asset.ETH, "direction": Direction.DOWN, "timeframe": Timeframe.FIVE_MIN},
    {"asset": Asset.ETH, "direction": Direction.UP, "timeframe": Timeframe.FIFTEEN_MIN},
    {"asset": Asset.ETH, "direction": Direction.DOWN, "timeframe": Timeframe.FIFTEEN_MIN},
]


def _market_key(asset: Asset, direction: Direction, timeframe: Timeframe) -> str:
    """Build the cache key for a given market, e.g. ``BTC_UP_5m``."""
    return f"{asset}_{direction}_{timeframe}"


class _MarketInfo:
    """Metadata for a single discovered Polymarket condition."""

    __slots__ = (
        "asset", "direction", "timeframe", "condition_id",
        "token_id", "yes_token_id", "no_token_id",
    )

    def __init__(
        self,
        asset: Asset,
        direction: Direction,
        timeframe: Timeframe,
        condition_id: str,
        token_id: str,
        yes_token_id: str = "",
        no_token_id: str = "",
    ) -> None:
        self.asset = asset
        self.direction = direction
        self.timeframe = timeframe
        self.condition_id = condition_id
        self.token_id = token_id
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id


class PolymarketFeed:
    """Polls Polymarket CLOB orderbooks for BTC/ETH short-timeframe markets
    and writes :class:`~popoly.types.MarketOdds` into a shared
    :class:`PriceCache`.

    Args:
        cache: The shared :class:`PriceCache` to write odds into.
        config: Application :class:`Config`.
    """

    def __init__(self, cache: PriceCache, config: Config) -> None:
        self._cache = cache
        self._config = config
        self._client = ClobClient(
            config.polymarket_api_url,
            key=config.api_key or None,
        )
        self._rate_limiter = RateLimiter(
            max_tokens=float(config.rate_limit_rps),
            refill_rate=float(config.rate_limit_rps),
        )
        self._markets: list[_MarketInfo] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main polling loop.  Discovers markets once, then repeatedly
        fetches orderbook snapshots at ``config.poll_interval``.
        """
        await self._discover_markets()

        if not self._markets:
            logger.error(
                "No Polymarket markets discovered -- PolymarketFeed will not poll."
            )
            return

        logger.info(
            "PolymarketFeed entering poll loop (%d markets, %.1fs interval).",
            len(self._markets),
            self._config.poll_interval,
        )

        while True:
            try:
                await self._poll_all()
            except asyncio.CancelledError:
                logger.info("PolymarketFeed cancelled -- shutting down.")
                raise
            except Exception:
                logger.exception("Error during Polymarket poll cycle.")

            await asyncio.sleep(self._config.poll_interval)

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def _discover_markets(self) -> None:
        """Query the Polymarket API to find condition/token IDs for the
        BTC/ETH 5min/15min up/down markets we care about.
        """
        logger.info("Discovering Polymarket markets ...")
        self._markets.clear()

        for spec in _MARKET_KEYWORDS:
            asset: Asset = spec["asset"]
            direction: Direction = spec["direction"]
            timeframe: Timeframe = spec["timeframe"]

            query = f"{asset} {direction.lower()} {timeframe}"
            try:
                await self._rate_limiter.acquire()
                # ClobClient is synchronous -- run in executor to avoid blocking.
                loop = asyncio.get_running_loop()
                markets_resp = await loop.run_in_executor(
                    None, self._client.get_markets
                )
            except Exception:
                logger.exception(
                    "Failed to query Polymarket API for %s.", query
                )
                continue

            # ``get_markets`` returns a list-like of market dicts.  Walk
            # through them to find one whose question/description matches.
            match = self._find_matching_market(
                markets_resp, asset, direction, timeframe
            )
            if match is not None:
                self._markets.append(match)
                key = _market_key(asset, direction, timeframe)
                logger.info(
                    "Discovered market %s  condition=%s  token=%s",
                    key,
                    match.condition_id,
                    match.token_id,
                )
            else:
                logger.warning(
                    "Could not find Polymarket market for %s.", query
                )

        logger.info("Market discovery complete: %d markets found.", len(self._markets))

    @staticmethod
    def _find_matching_market(
        markets: Any,
        asset: Asset,
        direction: Direction,
        timeframe: Timeframe,
    ) -> _MarketInfo | None:
        """Scan API response for the first market matching the given spec."""
        # Polymarket market list may be a list of dicts or objects.
        asset_lower = asset.lower()
        direction_lower = direction.lower()
        tf_str = str(timeframe)  # e.g. "5m" or "15m"

        for market in markets:
            # Support both dict-like and attribute access.
            question = (
                market.get("question", "") if isinstance(market, dict) else getattr(market, "question", "")
            )
            question_lower = question.lower()

            if (
                asset_lower in question_lower
                and direction_lower in question_lower
                and tf_str in question_lower
            ):
                condition_id = (
                    market.get("condition_id", "")
                    if isinstance(market, dict)
                    else getattr(market, "condition_id", "")
                )
                # Extract YES and NO token IDs from the tokens list.
                tokens = (
                    market.get("tokens", [])
                    if isinstance(market, dict)
                    else getattr(market, "tokens", [])
                )
                yes_token_id = ""
                no_token_id = ""
                for tok in tokens:
                    tok_id = (
                        tok.get("token_id", "")
                        if isinstance(tok, dict)
                        else getattr(tok, "token_id", "")
                    )
                    outcome = (
                        tok.get("outcome", "")
                        if isinstance(tok, dict)
                        else getattr(tok, "outcome", "")
                    )
                    if outcome.upper() == "YES":
                        yes_token_id = tok_id
                    elif outcome.upper() == "NO":
                        no_token_id = tok_id

                # Fallback: if outcomes not labelled, use positional.
                if not yes_token_id and tokens:
                    first_token = tokens[0]
                    yes_token_id = (
                        first_token.get("token_id", "")
                        if isinstance(first_token, dict)
                        else getattr(first_token, "token_id", "")
                    )
                if not no_token_id and len(tokens) > 1:
                    second_token = tokens[1]
                    no_token_id = (
                        second_token.get("token_id", "")
                        if isinstance(second_token, dict)
                        else getattr(second_token, "token_id", "")
                    )

                token_id = yes_token_id or no_token_id
                if condition_id and token_id:
                    return _MarketInfo(
                        asset=asset,
                        direction=direction,
                        timeframe=timeframe,
                        condition_id=condition_id,
                        token_id=token_id,
                        yes_token_id=yes_token_id,
                        no_token_id=no_token_id,
                    )

        return None

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_all(self) -> None:
        """Poll the orderbook for every discovered market."""
        for mkt in self._markets:
            try:
                await self._poll_orderbook(mkt.condition_id, mkt.token_id, mkt)
            except Exception:
                key = _market_key(mkt.asset, mkt.direction, mkt.timeframe)
                logger.exception("Failed to poll orderbook for %s.", key)

    async def _poll_orderbook(
        self,
        condition_id: str,
        token_id: str,
        market_info: _MarketInfo,
    ) -> None:
        """Fetch the current orderbook for a single market and update
        the :class:`PriceCache` with derived :class:`MarketOdds`.
        """
        await self._rate_limiter.acquire()

        loop = asyncio.get_running_loop()
        orderbook = await loop.run_in_executor(
            None, self._client.get_order_book, token_id
        )

        yes_prob, no_prob, best_bid, best_ask = self._compute_implied_prob(orderbook)

        odds = MarketOdds(
            asset=market_info.asset,
            direction=market_info.direction,
            timeframe=market_info.timeframe,
            yes_prob=yes_prob,
            no_prob=no_prob,
            best_bid=best_bid,
            best_ask=best_ask,
            condition_id=condition_id,
            token_id=token_id,
            timestamp=datetime.now(timezone.utc),
            yes_token_id=market_info.yes_token_id,
            no_token_id=market_info.no_token_id,
        )

        key = _market_key(market_info.asset, market_info.direction, market_info.timeframe)
        await self._cache.update_odds(key, odds)
        logger.debug(
            "Updated odds for %s: yes=%.3f  no=%.3f  bid=%.3f  ask=%.3f",
            key,
            yes_prob,
            no_prob,
            best_bid,
            best_ask,
        )

    @staticmethod
    def _compute_implied_prob(orderbook: Any) -> tuple[float, float, float, float]:
        """Derive implied probability from an orderbook snapshot.

        Returns:
            ``(yes_prob, no_prob, best_bid, best_ask)``

        The orderbook from ``py_clob_client`` exposes ``bids`` and ``asks``
        lists of ``{"price": str, "size": str}`` dicts (or similar).
        We take the top-of-book bid and ask, then compute the midpoint
        as the implied YES probability.
        """
        bids = (
            orderbook.get("bids", [])
            if isinstance(orderbook, dict)
            else getattr(orderbook, "bids", [])
        )
        asks = (
            orderbook.get("asks", [])
            if isinstance(orderbook, dict)
            else getattr(orderbook, "asks", [])
        )

        best_bid = 0.0
        best_ask = 1.0

        if bids:
            top_bid = bids[0]
            best_bid = float(
                top_bid.get("price", 0) if isinstance(top_bid, dict) else getattr(top_bid, "price", 0)
            )
        if asks:
            top_ask = asks[0]
            best_ask = float(
                top_ask.get("price", 1) if isinstance(top_ask, dict) else getattr(top_ask, "price", 1)
            )

        # Midpoint as implied YES probability.
        if best_bid > 0 and best_ask < 1:
            yes_prob = (best_bid + best_ask) / 2.0
        elif best_bid > 0:
            yes_prob = best_bid
        elif best_ask < 1:
            yes_prob = best_ask
        else:
            yes_prob = 0.5

        no_prob = 1.0 - yes_prob

        return yes_prob, no_prob, best_bid, best_ask
