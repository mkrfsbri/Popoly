"""Binance WebSocket feed for real-time BTC/ETH trade prices."""

from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets
import websockets.exceptions

from popoly.config import Config
from popoly.feeds.price_cache import PriceCache
from popoly.utils.retry import async_retry

logger = logging.getLogger(__name__)

# Symbols arriving from Binance are uppercased, e.g. "BTCUSDT".
# Map them to the short asset names used across Popoly.
_SYMBOL_MAP: dict[str, str] = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
}

_STALE_TIMEOUT: float = 30.0  # seconds without a message before we reconnect


class BinanceFeed:
    """Streams ``btcusdt@trade`` and ``ethusdt@trade`` from Binance and
    pushes every price tick into a shared :class:`PriceCache`.

    Args:
        cache: The shared :class:`PriceCache` to write prices into.
        config: Application :class:`Config` (used for the WS URL).
    """

    def __init__(self, cache: PriceCache, config: Config) -> None:
        self._cache = cache
        self._config = config
        self._ws_url = (
            f"{config.binance_ws_url}/btcusdt@trade/ethusdt@trade"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect to Binance and stream trades indefinitely.

        Automatically reconnects on any disconnect using exponential
        backoff (delegated to :func:`~popoly.utils.retry.async_retry`).
        """
        backoff = 1.0
        max_backoff = 60.0

        while True:
            try:
                await self._stream()
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError,
            ) as exc:
                logger.warning(
                    "Binance WS disconnected (%s). Reconnecting in %.1fs ...",
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except asyncio.CancelledError:
                logger.info("BinanceFeed cancelled -- shutting down.")
                raise
            except Exception:
                logger.exception(
                    "Unexpected error in BinanceFeed. Reconnecting in %.1fs ...",
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _stream(self) -> None:
        """Open a single WS session and consume messages until failure."""
        logger.info("Connecting to Binance WS: %s", self._ws_url)

        async with websockets.connect(  # type: ignore[attr-defined]
            self._ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            logger.info("Binance WS connected.")
            last_msg_time = time.monotonic()

            while True:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=_STALE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - last_msg_time
                    logger.warning(
                        "No Binance message for %.0fs -- reconnecting.",
                        elapsed,
                    )
                    return  # break out so the outer loop reconnects

                last_msg_time = time.monotonic()
                await self._handle_message(raw)

    async def _handle_message(self, raw: str | bytes) -> None:
        """Parse a single trade message and update the cache."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Non-JSON message from Binance: %s", raw[:120])
            return

        symbol: str | None = data.get("s")
        price_str: str | None = data.get("p")

        if symbol is None or price_str is None:
            # Not a trade message (could be a subscription confirmation).
            return

        asset = _SYMBOL_MAP.get(symbol.upper())
        if asset is None:
            return

        try:
            price = float(price_str)
        except (ValueError, TypeError):
            logger.debug("Invalid price value from Binance: %s", price_str)
            return

        await self._cache.update_price(asset, price)
