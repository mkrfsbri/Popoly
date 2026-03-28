"""Emergency kill switch for Popoly."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from popoly.config import Config
from popoly.risk.portfolio import Portfolio

logger = logging.getLogger(__name__)


class KillSwitch:
    """Monitors portfolio drawdown and halts trading when limits are breached.

    The switch uses an :class:`asyncio.Event` internally: when the event is
    *set*, trading is halted.  A manual :meth:`reset` is required to resume.
    """

    def __init__(
        self,
        config: Config,
        portfolio: Portfolio,
        *,
        on_engage: Callable[[str], Any] | None = None,
    ) -> None:
        self._config = config
        self._portfolio = portfolio
        self._event = asyncio.Event()
        self._on_engage = on_engage

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_engaged(self) -> bool:
        """Return ``True`` when the kill switch is engaged (trading halted)."""
        return self._event.is_set()

    def engage(self, reason: str) -> None:
        """Engage the kill switch, halting all trading.

        Parameters
        ----------
        reason:
            Human-readable explanation logged at CRITICAL level.
        """
        if not self._event.is_set():
            self._event.set()
            logger.critical("KILL SWITCH ENGAGED: %s", reason)
            if self._on_engage is not None:
                try:
                    self._on_engage(reason)
                except Exception:
                    logger.exception("on_engage callback failed")

    def reset(self) -> None:
        """Manually reset the kill switch to allow trading to resume."""
        self._event.clear()
        logger.warning("Kill switch reset — trading may resume")

    # ------------------------------------------------------------------
    # Background monitor
    # ------------------------------------------------------------------

    async def monitor(
        self,
        get_prices: Callable[[], Coroutine[Any, Any, dict[str, float]]],
    ) -> None:
        """Continuously check drawdown and engage if limit is exceeded.

        Parameters
        ----------
        get_prices:
            An async callable that returns the current price map
            (``token_id -> price``).
        """
        logger.info(
            "Kill-switch monitor started (max drawdown %.1f%%)",
            self._config.max_daily_drawdown * 100,
        )
        while True:
            try:
                prices = await get_prices()
                dd = self._portfolio.drawdown_pct(prices)
                # drawdown_pct returns negative when losing money.
                if dd < 0 and abs(dd) >= self._config.max_daily_drawdown:
                    self.engage(
                        f"daily drawdown {dd:.2%} exceeds "
                        f"limit -{self._config.max_daily_drawdown:.2%}"
                    )
            except asyncio.CancelledError:
                logger.info("Kill-switch monitor cancelled")
                raise
            except Exception:
                logger.exception("Error in kill-switch monitor loop")

            await asyncio.sleep(1.0)
