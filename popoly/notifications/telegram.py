"""Telegram notification service for Popoly."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from popoly.types import TradeRecord

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


class TelegramNotifier:
    """Send alerts to a Telegram chat via the Bot API.

    If *bot_token* or *chat_id* are empty the notifier is disabled and all
    calls silently log instead of sending HTTP requests.  Messages are rate-
    limited to at most one per second.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        self._client: httpx.AsyncClient | None = None
        self._last_send: float = 0.0
        self._lock = asyncio.Lock()

        if not self._enabled:
            logger.warning(
                "Telegram notifier disabled – bot_token or chat_id not set"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def _rate_limit(self) -> None:
        """Ensure at least 1 second between consecutive sends."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_send
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            self._last_send = time.monotonic()

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    async def send(self, message: str) -> None:
        """Send *message* to the configured Telegram chat.

        Errors are logged but never propagated so the trading bot keeps
        running even when Telegram is unreachable.
        """
        if not self._enabled:
            logger.info("[TG disabled] %s", message)
            return

        await self._rate_limit()

        try:
            client = await self._get_client()
            url = f"{_TELEGRAM_API}/bot{self._bot_token}/sendMessage"
            resp = await client.post(
                url,
                json={
                    "chat_id": self._chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
            )
            if resp.status_code != 200:
                logger.error(
                    "Telegram API returned %s: %s", resp.status_code, resp.text
                )
        except Exception:
            logger.exception("Failed to send Telegram message")

    # ------------------------------------------------------------------
    # Formatted alerts
    # ------------------------------------------------------------------

    async def trade_alert(self, trade: TradeRecord) -> None:
        """Format and send a trade notification."""
        mode = "PAPER" if trade.paper else "LIVE"
        text = (
            f"<b>{mode} Trade Executed</b>\n"
            f"Asset: {trade.asset}\n"
            f"Direction: {trade.direction}\n"
            f"Side: {trade.side}\n"
            f"Size: ${trade.size_usd:.2f}\n"
            f"Price: {trade.price:.4f}\n"
            f"Edge: {trade.edge:.2%}\n"
            f"Confidence: {trade.confidence:.2%}\n"
            f"ID: <code>{trade.id}</code>"
        )
        await self.send(text)

    async def drawdown_alert(
        self, drawdown_pct: float, portfolio_value: float
    ) -> None:
        """Send a drawdown warning."""
        text = (
            f"\u26a0\ufe0f <b>Drawdown Warning</b>\n"
            f"Drawdown: {drawdown_pct:.2%}\n"
            f"Portfolio value: ${portfolio_value:.2f}"
        )
        await self.send(text)

    async def kill_switch_alert(self, reason: str) -> None:
        """Send a critical kill-switch notification."""
        text = (
            f"\U0001f6a8 <b>KILL SWITCH ENGAGED</b>\n"
            f"Reason: {reason}\n"
            f"All trading has been halted."
        )
        await self.send(text)

    async def startup_alert(self, config_summary: str) -> None:
        """Send a bot-started notification."""
        text = (
            f"\u2705 <b>Popoly Bot Started</b>\n"
            f"<pre>{config_summary}</pre>"
        )
        await self.send(text)

    async def shutdown_alert(self) -> None:
        """Send a bot-stopped notification and close the HTTP client."""
        await self.send("\u274c <b>Popoly Bot Stopped</b>")
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
