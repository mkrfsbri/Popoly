"""Pre-trade risk validation gate for Popoly."""

from __future__ import annotations

import logging

from popoly.config import Config
from popoly.risk.kill_switch import KillSwitch
from popoly.risk.portfolio import Portfolio
from popoly.types import ConfidenceResult, Opportunity

logger = logging.getLogger(__name__)


class RiskGate:
    """Validates a prospective trade against portfolio risk limits.

    Every check must pass for the trade to be approved.  The method
    :meth:`check` returns a ``(bool, str)`` tuple indicating approval
    status and a human-readable reason.
    """

    def __init__(
        self,
        config: Config,
        portfolio: Portfolio,
        kill_switch: KillSwitch,
    ) -> None:
        self._config = config
        self._portfolio = portfolio
        self._kill_switch = kill_switch

    def check(
        self,
        opportunity: Opportunity,
        confidence: ConfidenceResult,
        position_size_usd: float,
        current_prices: dict[str, float],
    ) -> tuple[bool, str]:
        """Run all pre-trade risk checks.

        Returns
        -------
        tuple[bool, str]
            ``(True, "approved")`` when all checks pass, otherwise
            ``(False, "<reason>")`` with a description of the first
            failing check.
        """
        # 1. Kill switch must not be engaged.
        if self._kill_switch.is_engaged:
            return False, "kill switch is engaged"

        # 2. Edge must exceed threshold.
        edge = opportunity.edge
        threshold = self._config.edge_threshold
        if edge <= threshold:
            return False, (
                f"edge {edge:.1%} below threshold {threshold:.1%}"
            )

        # 3. Confidence must exceed threshold.
        score = confidence.score
        if score <= self._config.confidence_threshold:
            return False, (
                f"confidence {score:.1%} below threshold"
            )

        # 4. Position size must not exceed max percentage of portfolio.
        pos_pct = self._portfolio.position_pct(
            position_size_usd, current_prices
        )
        if pos_pct >= self._config.max_position_pct:
            return False, "position too large"

        logger.debug(
            "Trade approved — edge=%.1f%% confidence=%.1f%% size=%.1f%%",
            edge * 100,
            score * 100,
            pos_pct * 100,
        )
        return True, "approved"
