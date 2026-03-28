"""Terminal dashboard using the Rich library."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from popoly.config import Config
    from popoly.feeds.price_cache import PriceCache
    from popoly.risk.portfolio import Portfolio
    from popoly.types import TradeRecord

logger = logging.getLogger(__name__)


class Dashboard:
    """Rich-powered terminal dashboard for Popoly.

    Displays portfolio metrics, open positions, recent trades, and feed
    health information.  Designed to run as a background ``asyncio`` task
    alongside the main trading loop.
    """

    _REFRESH_INTERVAL: float = 1.0

    def __init__(
        self,
        portfolio: Portfolio,
        position_manager: Any,
        database: Any,
        config: Config,
        price_cache: PriceCache,
    ) -> None:
        self._portfolio = portfolio
        self._position_manager = position_manager
        self._database = database
        self._config = config
        self._price_cache = price_cache
        self._start_time = time.monotonic()
        self._kill_switch: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_kill_switch(self, active: bool) -> None:
        """Toggle the kill-switch indicator on the dashboard."""
        self._kill_switch = active

    async def run(self, get_prices: Callable[[], dict[str, float]]) -> None:
        """Continuously refresh the dashboard inside a ``rich.live.Live`` context.

        Parameters
        ----------
        get_prices:
            A callable (sync or async) that returns a ``{token_id: price}``
            mapping used for mark-to-market valuation.
        """
        with Live(self.build_layout({}, []), refresh_per_second=2, screen=True) as live:
            while True:
                try:
                    prices = get_prices() if not asyncio.iscoroutinefunction(get_prices) else await get_prices()
                    trades = await self._fetch_recent_trades()
                    live.update(self.build_layout(prices, trades))
                except Exception:
                    logger.debug("Dashboard render error", exc_info=True)
                await asyncio.sleep(self._REFRESH_INTERVAL)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def build_layout(
        self,
        current_prices: dict[str, float],
        recent_trades: list[TradeRecord],
    ) -> Layout:
        """Build the full terminal layout.

        Returns a :class:`rich.layout.Layout` tree with header, stats,
        positions, recent trades, and feed panels.
        """
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="upper", size=8),
            Layout(name="positions", ratio=2),
            Layout(name="lower", ratio=2),
        )

        layout["upper"].split_row(
            Layout(name="pnl", ratio=1),
            Layout(name="stats", ratio=1),
            Layout(name="feed", ratio=1),
        )

        layout["lower"].split_row(
            Layout(name="trades", ratio=1),
        )

        # Populate each section.
        layout["header"].update(self._build_header())
        layout["pnl"].update(self._build_pnl_panel(current_prices))
        layout["stats"].update(self._build_stats_panel(recent_trades))
        layout["feed"].update(self._build_feed_panel())
        layout["positions"].update(self._build_positions_panel(current_prices))
        layout["lower"].update(self._build_trades_panel(recent_trades))

        return layout

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_header(self) -> Panel:
        """Header bar with title, mode, uptime, and kill-switch status."""
        mode = self._config.is_live
        mode_text = Text("LIVE", style="bold red") if mode else Text("PAPER", style="bold green")

        uptime_secs = time.monotonic() - self._start_time
        hours, remainder = divmod(int(uptime_secs), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        kill_text = Text(" KILL SWITCH ACTIVE ", style="bold white on red") if self._kill_switch else Text("OK", style="green")

        header = Text.assemble(
            Text("  POPOLY  ", style="bold white on blue"),
            "  Mode: ",
            mode_text,
            f"  |  Uptime: {uptime_str}  |  Kill Switch: ",
            kill_text,
        )
        return Panel(header, style="bold")

    def _build_pnl_panel(self, current_prices: dict[str, float]) -> Panel:
        """P&L panel: daily, cumulative, and max drawdown."""
        daily_abs = self._portfolio.daily_pnl(current_prices)
        daily_pct = self._portfolio.drawdown_pct(current_prices)
        total_value = self._portfolio.total_value(current_prices)
        cumulative = total_value - self._portfolio.initial_value
        cumulative_pct = (cumulative / self._portfolio.initial_value * 100) if self._portfolio.initial_value else 0.0

        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", justify="right")
        table.add_column()

        table.add_row("Daily P&L", self._fmt_pnl(daily_abs, pct=daily_pct * 100))
        table.add_row("Cumulative", self._fmt_pnl(cumulative, pct=cumulative_pct))
        table.add_row("Total Value", f"${total_value:,.2f}")
        table.add_row("Max DD", self._fmt_pct(daily_pct * 100, invert=True))

        return Panel(table, title="P&L", border_style="cyan")

    def _build_stats_panel(self, trades: list[TradeRecord]) -> Panel:
        """Trade statistics panel."""
        closed = [t for t in trades if t.status.value == "CLOSED"]
        wins = [t for t in closed if (t.pnl or 0) > 0]
        losses = [t for t in closed if (t.pnl or 0) <= 0]
        total = len(closed)
        win_rate = (len(wins) / total * 100) if total > 0 else 0.0

        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", justify="right")
        table.add_column()

        table.add_row("Total Trades", str(total))
        table.add_row("Wins", Text(str(len(wins)), style="green"))
        table.add_row("Losses", Text(str(len(losses)), style="red"))
        table.add_row("Win Rate", self._fmt_pct(win_rate))

        return Panel(table, title="Stats", border_style="cyan")

    def _build_positions_panel(self, current_prices: dict[str, float]) -> Panel:
        """Open positions table."""
        table = Table(
            title="Open Positions",
            expand=True,
            show_lines=False,
            pad_edge=True,
        )
        table.add_column("Asset", style="bold")
        table.add_column("Direction")
        table.add_column("Timeframe")
        table.add_column("Side")
        table.add_column("Size", justify="right")
        table.add_column("Entry Price", justify="right")
        table.add_column("Cur. Prob", justify="right")
        table.add_column("Unreal. P&L", justify="right")

        for (_cid, tid), pos in self._portfolio.positions.items():
            cur_price = current_prices.get(tid, pos["entry_price"])
            qty = pos["size_usd"] / pos["entry_price"]
            unrealized = (cur_price - pos["entry_price"]) * qty

            table.add_row(
                str(pos.get("asset", "?")),
                str(pos.get("direction", "?")),
                str(pos.get("timeframe", "?")),
                str(pos.get("side", "?")),
                f"${pos['size_usd']:,.2f}",
                f"{pos['entry_price']:.4f}",
                f"{cur_price:.4f}",
                self._fmt_pnl_text(unrealized),
            )

        if not self._portfolio.positions:
            table.add_row(
                Text("No open positions", style="dim"),
                "", "", "", "", "", "", "",
            )

        return Panel(table, border_style="blue")

    def _build_trades_panel(self, trades: list[TradeRecord]) -> Panel:
        """Last 10 trades table."""
        table = Table(
            title="Last 10 Trades",
            expand=True,
            show_lines=False,
            pad_edge=True,
        )
        table.add_column("Time")
        table.add_column("Asset", style="bold")
        table.add_column("Dir")
        table.add_column("Side")
        table.add_column("Size", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Edge", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Status")

        last_10 = trades[-10:] if len(trades) > 10 else trades
        for t in reversed(last_10):
            pnl_text = self._fmt_pnl_text(t.pnl) if t.pnl is not None else Text("-", style="dim")
            status_style = {
                "OPEN": "yellow",
                "CLOSED": "green",
                "CANCELLED": "red",
                "CLAIMED": "bold green",
                "MERGED": "cyan",
            }.get(t.status.value, "white")

            table.add_row(
                t.timestamp.strftime("%H:%M:%S"),
                str(t.asset),
                str(t.direction),
                str(t.side),
                f"${t.size_usd:,.2f}",
                f"{t.price:.4f}",
                f"{t.edge:.2%}",
                pnl_text,
                Text(t.status.value, style=status_style),
            )

        if not trades:
            table.add_row(
                Text("No trades yet", style="dim"),
                "", "", "", "", "", "", "", "",
            )

        return Panel(table, border_style="magenta")

    def _build_feed_panel(self) -> Panel:
        """Feed health indicators for Binance and Polymarket."""
        now = time.monotonic()
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", justify="right")
        table.add_column()

        # Binance price feed -- read from the internal cache dict directly
        # to avoid awaiting inside a sync render method.
        binance_info = self._extract_price_info("BTC", now)
        table.add_row("Binance BTC", binance_info)

        binance_eth = self._extract_price_info("ETH", now)
        table.add_row("Binance ETH", binance_eth)

        # Polymarket -- check any odds key to gauge freshness.
        poly_info = self._extract_odds_info(now)
        table.add_row("Polymarket", poly_info)

        return Panel(table, title="Feed Status", border_style="cyan")

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    async def _fetch_recent_trades(self) -> list[TradeRecord]:
        """Pull recent trades from the database or position manager.

        Falls back to an empty list when the backing store is unavailable.
        """
        try:
            if hasattr(self._database, "get_recent_trades"):
                result = self._database.get_recent_trades(limit=50)
                if asyncio.iscoroutine(result):
                    result = await result
                return list(result)  # type: ignore[arg-type]
            if hasattr(self._position_manager, "trades"):
                return list(self._position_manager.trades)
        except Exception:
            logger.debug("Could not fetch recent trades", exc_info=True)
        return []

    def _extract_price_info(self, asset: str, now: float) -> Text:
        """Read price cache internals synchronously for display."""
        entry = self._price_cache._prices.get(asset)  # noqa: SLF001
        if entry is None:
            return Text("no data", style="dim red")
        age = now - entry.timestamp
        price_str = f"${entry.price:,.2f}"
        age_str = f"{age:.0f}s ago"
        style = "green" if age < self._config.staleness_threshold else "red"
        return Text(f"{price_str}  ({age_str})", style=style)

    def _extract_odds_info(self, now: float) -> Text:
        """Read odds cache internals synchronously for display."""
        if not self._price_cache._odds:  # noqa: SLF001
            return Text("no data", style="dim red")
        # Use the most recently updated odds entry.
        latest = max(self._price_cache._odds.values(), key=lambda e: e.timestamp)  # noqa: SLF001
        age = now - latest.timestamp
        age_str = f"{age:.0f}s ago"
        style = "green" if age < self._config.staleness_threshold else "red"
        return Text(f"last poll {age_str}", style=style)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_pnl(value: float, *, pct: float | None = None) -> Text:
        """Format a dollar P&L value, optionally with a percentage."""
        style = "green" if value >= 0 else "red"
        sign = "+" if value >= 0 else ""
        text = f"{sign}${value:,.2f}"
        if pct is not None:
            text += f"  ({sign}{pct:.2f}%)"
        return Text(text, style=style)

    @staticmethod
    def _fmt_pnl_text(value: float | None) -> Text:
        """Format a single P&L number with color."""
        if value is None:
            return Text("-", style="dim")
        style = "green" if value >= 0 else "red"
        sign = "+" if value >= 0 else ""
        return Text(f"{sign}${value:,.2f}", style=style)

    @staticmethod
    def _fmt_pct(value: float, *, invert: bool = False) -> Text:
        """Format a percentage with green/red coloring.

        When *invert* is True negative values are shown green (useful
        for drawdown where a smaller magnitude is better).
        """
        positive_is_good = not invert
        if positive_is_good:
            style = "green" if value >= 0 else "red"
        else:
            style = "green" if value <= 0 else "red"
        sign = "+" if value >= 0 else ""
        return Text(f"{sign}{value:.2f}%", style=style)
