"""Conditional Token Framework (CTF) on-chain operations for Polymarket.

Provides helpers for:
- **mergePositions**: burn equal amounts of YES + NO tokens to reclaim
  USDC.e collateral (1 YES + 1 NO → $1 USDC.e).
- **redeemPositions**: after a market resolves, redeem winning tokens
  for USDC.e collateral.

On Polygon mainnet the relevant contracts are:
- CTF:     0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
- USDC.e:  0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Polygon mainnet addresses.
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
PARENT_COLLECTION_ID = "0x" + "00" * 32  # 32 zero bytes
BINARY_PARTITION = [1, 2]  # index sets for a binary (YES/NO) market

# Retry settings.
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds


class CTFOperator:
    """Interacts with the Polymarket CTF contract on Polygon.

    Uses ``py_clob_client`` for signing and broadcasting transactions.
    All public methods are async-safe (sync calls run in the default
    executor).
    """

    def __init__(self, polymarket_api_url: str, private_key: str) -> None:
        self._api_url = polymarket_api_url
        self._private_key = private_key
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the ClobClient."""
        if self._client is None:
            from py_clob_client.client import ClobClient

            self._client = ClobClient(
                host=self._api_url,
                key=self._private_key,
                chain_id=137,
            )
        return self._client

    # ------------------------------------------------------------------
    # Merge: YES + NO → USDC.e
    # ------------------------------------------------------------------

    async def merge_positions(
        self,
        condition_id: str,
        amount: int,
    ) -> dict[str, Any]:
        """Call ``mergePositions`` on the CTF contract.

        Burns *amount* of each YES and NO token for the given
        *condition_id* and receives *amount* USDC.e in return.

        Parameters
        ----------
        condition_id:
            The market's condition ID (bytes32 hex string).
        amount:
            Number of full sets (YES+NO pairs) to merge.  Also the
            amount of USDC.e collateral received.

        Returns
        -------
        dict
            Transaction receipt or API response.
        """
        return await self._retry(
            "mergePositions",
            self._do_merge,
            condition_id,
            amount,
        )

    def _do_merge(self, condition_id: str, amount: int) -> dict[str, Any]:
        """Synchronous merge call."""
        client = self._get_client()

        # py_clob_client >= 0.40 exposes merge_positions directly.
        if hasattr(client, "merge_positions"):
            return client.merge_positions(
                condition_id=condition_id,
                amount=amount,
            )

        # Fallback: call the CTF contract via the low-level web3 helper
        # that py_clob_client wraps.  This uses the raw ABI.
        if hasattr(client, "contract") and hasattr(client.contract, "functions"):
            tx = client.contract.functions.mergePositions(
                USDC_E_ADDRESS,
                bytes.fromhex(PARENT_COLLECTION_ID[2:]),
                bytes.fromhex(condition_id[2:]) if condition_id.startswith("0x") else bytes.fromhex(condition_id),
                BINARY_PARTITION,
                amount,
            ).build_transaction(client._build_tx_params())  # noqa: SLF001
            signed = client._sign_and_send(tx)  # noqa: SLF001
            return {"tx_hash": signed.hex() if isinstance(signed, bytes) else str(signed)}

        raise RuntimeError(
            "py_clob_client does not expose merge_positions or raw contract access"
        )

    # ------------------------------------------------------------------
    # Redeem: resolved tokens → USDC.e
    # ------------------------------------------------------------------

    async def redeem_positions(
        self,
        condition_id: str,
        amounts: list[int] | None = None,
    ) -> dict[str, Any]:
        """Call ``redeemPositions`` on the CTF contract.

        After a market resolves, the winning outcome tokens can be
        redeemed 1:1 for USDC.e.  Losing tokens are worth zero.

        Parameters
        ----------
        condition_id:
            The resolved market's condition ID.
        amounts:
            Optional explicit amounts per index set.  If ``None`` the
            client redeems all available tokens.

        Returns
        -------
        dict
            Transaction receipt or API response.
        """
        return await self._retry(
            "redeemPositions",
            self._do_redeem,
            condition_id,
            amounts,
        )

    def _do_redeem(
        self,
        condition_id: str,
        amounts: list[int] | None,
    ) -> dict[str, Any]:
        """Synchronous redeem call."""
        client = self._get_client()

        if hasattr(client, "redeem_positions"):
            kwargs: dict[str, Any] = {"condition_id": condition_id}
            if amounts is not None:
                kwargs["amounts"] = amounts
            return client.redeem_positions(**kwargs)

        # Older py_clob_client — try the simpler "redeem" alias.
        if hasattr(client, "redeem"):
            return client.redeem(condition_id)

        # Raw contract fallback.
        if hasattr(client, "contract") and hasattr(client.contract, "functions"):
            tx = client.contract.functions.redeemPositions(
                USDC_E_ADDRESS,
                bytes.fromhex(PARENT_COLLECTION_ID[2:]),
                bytes.fromhex(condition_id[2:]) if condition_id.startswith("0x") else bytes.fromhex(condition_id),
                BINARY_PARTITION,
            ).build_transaction(client._build_tx_params())  # noqa: SLF001
            signed = client._sign_and_send(tx)  # noqa: SLF001
            return {"tx_hash": signed.hex() if isinstance(signed, bytes) else str(signed)}

        raise RuntimeError(
            "py_clob_client does not expose redeem_positions or raw contract access"
        )

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    async def _retry(
        self,
        label: str,
        fn: Any,
        *args: Any,
    ) -> dict[str, Any]:
        """Run *fn* in an executor with retries and exponential backoff."""
        loop = asyncio.get_running_loop()
        last_exc: BaseException | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = await loop.run_in_executor(None, fn, *args)
                logger.info(
                    "[CTF] %s succeeded (attempt %d): %s",
                    label,
                    attempt,
                    result,
                )
                return result  # type: ignore[return-value]
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[CTF] %s attempt %d/%d failed: %s",
                    label,
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF * attempt)

        msg = f"[CTF] All {_MAX_RETRIES} {label} attempts failed. Last: {last_exc}"
        logger.error(msg)
        raise RuntimeError(msg) from last_exc
