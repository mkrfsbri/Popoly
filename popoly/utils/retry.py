"""Async retry decorator with exponential backoff and jitter."""

import asyncio
import functools
import logging
import random
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that retries an async function with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of attempts (including the first call).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper bound on the delay between retries.
        exceptions: Tuple of exception types that trigger a retry.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc]
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_attempts,
                            func.__qualname__,
                            exc,
                        )
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    jitter = random.uniform(0, delay * 0.5)  # noqa: S311
                    total_delay = delay + jitter
                    logger.warning(
                        "Attempt %d/%d for %s failed (%s). Retrying in %.2fs.",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        exc,
                        total_delay,
                    )
                    await asyncio.sleep(total_delay)

            # Should be unreachable, but keeps the type-checker happy.
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
