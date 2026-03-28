"""Configuration module for Popoly."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Self

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable application configuration."""

    # --- Polymarket API credentials ---
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    private_key: str = ""

    # --- Telegram ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- Live-trading guards (ALL three must be True for live mode) ---
    live: bool = False
    live_confirm: bool = False
    live_ack_risk: bool = False

    # --- Trading thresholds ---
    edge_threshold: float = 0.05
    confidence_threshold: float = 0.85
    max_position_pct: float = 0.08
    lag_threshold_pp: float = 3.0

    # --- Risk ---
    max_daily_drawdown: float = 0.20
    kelly_fraction: float = 0.5

    # --- Balance ---
    initial_balance: float = 1000.0

    # --- Storage ---
    db_path: str = "data/popoly.db"

    # --- External endpoints ---
    binance_ws_url: str = "wss://stream.binance.com:9443/ws"
    polymarket_api_url: str = "https://clob.polymarket.com"

    # --- Polling / rate-limiting ---
    poll_interval: float = 2.0
    staleness_threshold: float = 10.0
    rate_limit_rps: int = 10

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def is_live(self) -> bool:
        """Return ``True`` only when all three live-mode flags are set."""
        return self.live and self.live_confirm and self.live_ack_risk

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> Self:
        """Build a ``Config`` from environment variables (+ .env file)."""
        load_dotenv()

        def _bool(key: str) -> bool:
            return os.getenv(key, "").lower() in {"1", "true", "yes"}

        def _float(key: str, default: float) -> float:
            raw = os.getenv(key)
            return float(raw) if raw is not None else default

        def _int(key: str, default: int) -> int:
            raw = os.getenv(key)
            return int(raw) if raw is not None else default

        return cls(
            # Polymarket
            api_key=os.getenv("POLYMARKET_API_KEY", ""),
            api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
            api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
            private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
            # Telegram
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            # Live-mode flags
            live=_bool("POPOLY_LIVE"),
            live_confirm=_bool("POPOLY_LIVE_CONFIRM"),
            live_ack_risk=_bool("POPOLY_LIVE_ACK_RISK"),
            # Thresholds
            edge_threshold=_float("POPOLY_EDGE_THRESHOLD", 0.05),
            confidence_threshold=_float("POPOLY_CONFIDENCE_THRESHOLD", 0.85),
            max_position_pct=_float("POPOLY_MAX_POSITION_PCT", 0.08),
            lag_threshold_pp=_float("POPOLY_LAG_THRESHOLD_PP", 3.0),
            # Risk
            max_daily_drawdown=_float("POPOLY_MAX_DAILY_DRAWDOWN", 0.20),
            kelly_fraction=_float("POPOLY_KELLY_FRACTION", 0.5),
            # Balance
            initial_balance=_float("POPOLY_INITIAL_BALANCE", 1000.0),
            # Storage
            db_path=os.getenv("POPOLY_DB_PATH", "data/popoly.db"),
            # Endpoints
            binance_ws_url=os.getenv(
                "POPOLY_BINANCE_WS_URL",
                "wss://stream.binance.com:9443/ws",
            ),
            polymarket_api_url=os.getenv(
                "POPOLY_POLYMARKET_API_URL",
                "https://clob.polymarket.com",
            ),
            # Polling / rate-limiting
            poll_interval=_float("POPOLY_POLL_INTERVAL", 2.0),
            staleness_threshold=_float("POPOLY_STALENESS_THRESHOLD", 10.0),
            rate_limit_rps=_int("POPOLY_RATE_LIMIT_RPS", 10),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if the config is invalid for live trading.

        In paper mode only basic sanity checks are applied.  When
        ``is_live`` is ``True``, Polymarket credentials must be present.
        """
        if self.edge_threshold <= 0:
            raise ValueError("edge_threshold must be positive")
        if not 0 < self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        if not 0 < self.max_daily_drawdown <= 1:
            raise ValueError("max_daily_drawdown must be in (0, 1]")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if self.rate_limit_rps <= 0:
            raise ValueError("rate_limit_rps must be positive")

        if self.is_live:
            missing = [
                name
                for name, value in (
                    ("api_key", self.api_key),
                    ("api_secret", self.api_secret),
                    ("api_passphrase", self.api_passphrase),
                    ("private_key", self.private_key),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"Live mode requires Polymarket credentials: {', '.join(missing)}"
                )
