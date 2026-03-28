"""Token-bucket rate limiter for async contexts."""

import asyncio
import time


class RateLimiter:
    """A token-bucket rate limiter that is safe to use from async code.

    Args:
        max_tokens: Maximum number of tokens the bucket can hold.
        refill_rate: Tokens added per second.
    """

    def __init__(self, max_tokens: float, refill_rate: float) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")

        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self._tokens = max_tokens
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Add tokens based on elapsed time since the last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until the requested number of tokens is available, then consume them.

        Args:
            tokens: Number of tokens to consume (default 1).
        """
        if tokens > self.max_tokens:
            raise ValueError(
                f"Requested {tokens} tokens but bucket capacity is {self.max_tokens}"
            )

        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Calculate how long until enough tokens are available.
                deficit = tokens - self._tokens
                wait_time = deficit / self.refill_rate

            await asyncio.sleep(wait_time)
