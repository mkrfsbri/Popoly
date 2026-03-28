"""Popoly entry point: python -m popoly"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time

from popoly.config import Config
from popoly.dashboard.display import Dashboard
from popoly.execution.claim_manager import ClaimManager
from popoly.execution.executor import Executor
from popoly.execution.paper_trader import PaperTrader
from popoly.execution.position_manager import PositionManager
from popoly.feeds.binance_ws import BinanceFeed
from popoly.feeds.polymarket_clob import PolymarketFeed
from popoly.feeds.price_cache import PriceCache
from popoly.notifications.telegram import TelegramNotifier
from popoly.risk.kill_switch import KillSwitch
from popoly.risk.portfolio import Portfolio
from popoly.risk.risk_gate import RiskGate
from popoly.storage.database import Database
from popoly.strategy.confidence import ConfidenceScorer
from popoly.strategy.kelly import compute_position_size
from popoly.strategy.opportunity import OpportunityDetector
from popoly.utils.logging_config import setup_logging

logger = logging.getLogger("popoly.main")


class Bot:
    """Main bot orchestrator."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.price_cache = PriceCache()
        self.portfolio = Portfolio(config.initial_balance)
        self.kill_switch = KillSwitch(config, self.portfolio)
        self.position_manager = PositionManager()
        self.paper_trader = PaperTrader()
        self.telegram = TelegramNotifier(config.telegram_bot_token, config.telegram_chat_id)
        self.db = Database(config.db_path)
        self.executor = Executor(
            config=config,
            paper_trader=self.paper_trader,
            portfolio=self.portfolio,
            database=self.db,
            telegram=self.telegram,
            position_manager=self.position_manager,
        )
        self.claim_manager = ClaimManager(
            config=config,
            position_manager=self.position_manager,
            portfolio=self.portfolio,
            price_cache=self.price_cache,
            database=self.db,
            telegram=self.telegram,
        )
        self.opportunity_detector = OpportunityDetector(config)
        self.confidence_scorer = ConfidenceScorer()
        self.risk_gate = RiskGate(config, self.portfolio, self.kill_switch)
        self.binance_feed = BinanceFeed(self.price_cache, config)
        self.polymarket_feed = PolymarketFeed(self.price_cache, config)
        self.dashboard = Dashboard(
            portfolio=self.portfolio,
            position_manager=self.position_manager,
            database=self.db,
            config=config,
            price_cache=self.price_cache,
        )
        self._running = True
        self._start_time = time.time()

    def _get_current_prices(self) -> dict:
        """Build current prices dict from price cache for portfolio valuation."""
        prices = {}
        for asset in ("BTC", "ETH"):
            data = self.price_cache._prices.get(asset)
            if data:
                prices[asset] = data[0]
        return prices

    async def _get_current_prices_async(self) -> dict:
        return self._get_current_prices()

    async def _strategy_loop(self) -> None:
        """Main strategy loop: detect opportunities, check risk, execute."""
        from popoly.types import Asset, PriceSnapshot

        logger.info("Strategy loop started")
        while self._running:
            try:
                await asyncio.sleep(self.config.poll_interval)

                raw_snapshot = await self.price_cache.snapshot()
                if not raw_snapshot.get("prices"):
                    continue

                # Convert raw snapshot to the format OpportunityDetector expects:
                #   "prices": dict[Asset, PriceSnapshot]
                #   "markets": list[MarketOdds]
                from datetime import datetime, timezone

                prices_for_detector: dict[Asset, PriceSnapshot] = {}
                for asset_str, pdata in raw_snapshot["prices"].items():
                    asset = Asset(asset_str)
                    prices_for_detector[asset] = PriceSnapshot(
                        asset=asset,
                        price=pdata["price"],
                        timestamp=datetime.now(timezone.utc),
                    )

                markets = list(raw_snapshot.get("odds", {}).values())

                detector_snapshot = {
                    "prices": prices_for_detector,
                    "markets": markets,
                }

                # Detect opportunities
                opportunities = self.opportunity_detector.detect(detector_snapshot)
                if not opportunities:
                    continue

                current_prices = self._get_current_prices()

                for opp in opportunities:
                    # Score confidence
                    confidence = self.confidence_scorer.score(opp, snapshot)

                    # Size the position
                    size_usd = compute_position_size(
                        edge=opp.edge,
                        market_prob=opp.market_prob,
                        portfolio_value=self.portfolio.total_value(current_prices),
                        config=self.config,
                    )
                    if size_usd <= 0:
                        continue

                    # Risk gate
                    approved, reason = self.risk_gate.check(
                        opportunity=opp,
                        confidence=confidence,
                        position_size_usd=size_usd,
                        current_prices=current_prices,
                    )
                    if not approved:
                        logger.debug("Trade rejected: %s", reason)
                        continue

                    # Execute
                    from popoly.types import TradeIntent

                    intent = TradeIntent(
                        opportunity=opp,
                        size_usd=size_usd,
                        confidence=confidence.score,
                    )
                    trade = await self.executor.execute(intent, snapshot)
                    if trade:
                        logger.info(
                            "Trade executed: %s %s %s | size=$%.2f edge=%.1f%% conf=%.1f%%",
                            trade.asset,
                            trade.direction,
                            trade.side,
                            trade.size_usd,
                            trade.edge * 100,
                            trade.confidence * 100,
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in strategy loop")
                await asyncio.sleep(5)

    def _kill_switch_engage_callback(self, reason: str) -> None:
        """Called synchronously when kill switch engages."""
        self.dashboard.set_kill_switch(True)
        # Schedule async Telegram alert without blocking.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.telegram.kill_switch_alert(reason))
        except RuntimeError:
            pass

    async def run(self) -> None:
        """Start all components and run until interrupted."""
        setup_logging()

        # Initialize database
        await self.db.init()

        # Set kill switch callback
        self.kill_switch._on_engage = self._kill_switch_engage_callback

        mode = "LIVE" if self.config.is_live else "PAPER"
        logger.info("Popoly starting in %s mode", mode)

        config_summary = (
            f"Mode: {mode}\n"
            f"Initial balance: ${self.config.initial_balance:.2f}\n"
            f"Edge threshold: {self.config.edge_threshold:.0%}\n"
            f"Confidence threshold: {self.config.confidence_threshold:.0%}\n"
            f"Max position: {self.config.max_position_pct:.0%}\n"
            f"Max drawdown: {self.config.max_daily_drawdown:.0%}\n"
            f"Kelly fraction: {self.config.kelly_fraction}"
        )
        await self.telegram.startup_alert(config_summary)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.binance_feed.run(), name="binance_feed")
                tg.create_task(self.polymarket_feed.run(), name="polymarket_feed")
                tg.create_task(
                    self.kill_switch.monitor(self._get_current_prices_async),
                    name="kill_switch",
                )
                tg.create_task(self._strategy_loop(), name="strategy")
                tg.create_task(self.claim_manager.run(), name="claim_manager")
                tg.create_task(
                    self.dashboard.run(self._get_current_prices_async),
                    name="dashboard",
                )
        except* KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except* Exception as eg:
            for exc in eg.exceptions:
                logger.exception("Task failed: %s", exc)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False
        await self.telegram.shutdown_alert()
        await self.db.close()
        logger.info("Shutdown complete")


def main() -> None:
    config = Config.from_env()
    config.validate()

    bot = Bot(config)

    loop = asyncio.new_event_loop()

    def _signal_handler() -> None:
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(bot.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
