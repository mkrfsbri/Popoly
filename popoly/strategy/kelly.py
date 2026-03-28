"""Kelly Criterion position sizing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from popoly.config import Config


def half_kelly(
    edge: float,
    odds: float,
    kelly_fraction: float = 0.5,
) -> float:
    """Return the fractional Kelly stake (0.0 .. 1.0).

    Uses the standard Kelly formula::

        f* = (b * p - q) / b

    where *b* = ``odds`` (net payout per dollar risked), *p* is the
    estimated win probability derived from *edge*
    (``p = 0.5 + edge / 2``), and *q = 1 - p*.

    The result is scaled by *kelly_fraction* (default 0.5 for half-Kelly)
    and clamped to ``[0.0, 1.0]``.
    """
    if odds <= 0:
        return 0.0

    p = 0.5 + edge / 2.0
    q = 1.0 - p
    f_star = (odds * p - q) / odds
    sized = f_star * kelly_fraction
    return max(0.0, min(sized, 1.0))


def compute_position_size(
    edge: float,
    market_prob: float,
    portfolio_value: float,
    config: Config,
) -> float:
    """Return the dollar amount to wager.

    Steps:
      1. Derive implied odds from *market_prob*.
      2. Compute the half-Kelly fraction.
      3. Multiply by *portfolio_value*.
      4. Cap at ``config.max_position_pct * portfolio_value``.
    """
    if market_prob <= 0 or market_prob >= 1:
        return 0.0

    # Implied decimal odds: payout per dollar risked (net of stake).
    odds = (1.0 / market_prob) - 1.0

    fraction = half_kelly(edge, odds, kelly_fraction=config.kelly_fraction)
    amount = fraction * portfolio_value
    cap = config.max_position_pct * portfolio_value
    return min(amount, cap)
