#!/usr/bin/env python3
"""
Profit transfer executor for moving USDC from hot wallet to cold storage.

Handles:
- Transaction building and signing
- Gas estimation
- Exponential backoff retry
- Receipt monitoring
- Database logging
"""
import asyncio
import os
import logging
from typing import Optional
from datetime import datetime, timezone

from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from eth_account.signers.local import LocalAccount

logger = logging.getLogger(__name__)

# USDC contract on Polygon
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_DECIMALS = 6

# ERC-20 ABI (minimal)
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]


class ProfitTransferExecutor:
    """
    Execute USDC transfers to cold wallet with retry logic.

    Usage:
        executor = ProfitTransferExecutor(rpc_url, private_key, hot_wallet, cold_wallet, db, discord)
        tx_hash = await executor.transfer(amount_dollars=150.0, session_id=session_id)
    """

    def __init__(
        self,
        rpc_url: str,
        private_key: str,
        hot_wallet_address: str,
        cold_wallet_address: str,
        db,  # Database instance
        discord=None  # Optional DiscordWebhook instance
    ):
        """
        Initialize profit transfer executor.

        Args:
            rpc_url: Polygon RPC endpoint
            private_key: Hot wallet private key (for signing)
            hot_wallet_address: Source wallet address
            cold_wallet_address: Destination wallet address
            db: Database instance for logging
            discord: Optional DiscordWebhook for alerts
        """
        self.rpc_url = rpc_url
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.db = db
        self.discord = discord

        # Validate connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Polygon RPC: {rpc_url}")

        # Create account from private key
        try:
            self.account: LocalAccount = Account.from_key(private_key)
        except Exception as e:
            raise ValueError(f"Invalid private key: {e}")

        # Checksum addresses
        try:
            self.hot_wallet = Web3.to_checksum_address(hot_wallet_address)
            self.cold_wallet = Web3.to_checksum_address(cold_wallet_address)
        except ValueError as e:
            raise ValueError(f"Invalid wallet address: {e}")

        # Verify account matches hot wallet
        if self.account.address != self.hot_wallet:
            raise ValueError(
                f"Private key mismatch: expected {self.hot_wallet}, "
                f"got {self.account.address}"
            )

        # Initialize USDC contract
        self.usdc_address = Web3.to_checksum_address(USDC_ADDRESS)
        self.usdc_contract: Contract = self.w3.eth.contract(
            address=self.usdc_address,
            abi=ERC20_ABI
        )

        # Config from environment
        self.max_retries = int(os.getenv("MAX_TRANSFER_RETRIES", "3"))
        self.base_backoff = int(os.getenv("TRANSFER_RETRY_BACKOFF_SECONDS", "60"))
        self.min_amount = float(os.getenv("MINIMUM_TRANSFER_AMOUNT", "10"))
        self.max_amount = float(os.getenv("MAXIMUM_SINGLE_TRANSFER", "10000"))
        self.dry_run = os.getenv("PROFIT_TRANSFER_DRY_RUN", "false").lower() == "true"

        logger.info(
            f"ProfitTransferExecutor initialized: "
            f"{self.hot_wallet[:10]}... → {self.cold_wallet[:10]}... "
            f"(dry_run={self.dry_run})"
        )

    def _preflight_checks(self, amount_dollars: float) -> None:
        """
        Validate transfer parameters.

        Args:
            amount_dollars: Transfer amount in dollars

        Raises:
            ValueError: If validation fails
        """
        if amount_dollars < self.min_amount:
            raise ValueError(
                f"Amount ${amount_dollars:.2f} below minimum ${self.min_amount}"
            )

        if amount_dollars > self.max_amount:
            raise ValueError(
                f"Amount ${amount_dollars:.2f} exceeds maximum ${self.max_amount}"
            )

        if amount_dollars <= 0:
            raise ValueError(f"Amount must be positive, got ${amount_dollars:.2f}")

    async def transfer(
        self,
        amount_dollars: float,
        session_id: str,
        trigger_reason: str = "threshold"
    ) -> Optional[str]:
        """
        Transfer USDC to cold wallet.

        Args:
            amount_dollars: Amount to transfer in dollars
            session_id: Trading session ID
            trigger_reason: Why transfer triggered ('threshold' or 'time_interval')

        Returns:
            Transaction hash (hex string) if successful, None if failed

        Raises:
            ValueError: If pre-flight checks fail
        """
        # Pre-flight validation
        self._preflight_checks(amount_dollars)

        # Convert to raw USDC amount (6 decimals)
        amount_raw = int(amount_dollars * (10 ** USDC_DECIMALS))

        logger.info(
            f"Initiating transfer: ${amount_dollars:.2f} USDC "
            f"({amount_raw} raw) to {self.cold_wallet[:10]}... "
            f"(trigger: {trigger_reason})"
        )

        # Get nonce
        nonce = await asyncio.to_thread(
            self.w3.eth.get_transaction_count, self.hot_wallet
        )

        # Build transfer function call
        transfer_function = self.usdc_contract.functions.transfer(
            self.cold_wallet,
            amount_raw
        )

        # Estimate gas (with 20% buffer)
        try:
            gas_estimate = await asyncio.to_thread(
                transfer_function.estimate_gas,
                {"from": self.hot_wallet}
            )
            gas_limit = int(gas_estimate * 1.2)
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            gas_limit = 100000  # Fallback gas limit

        # Get current gas price
        gas_price = await asyncio.to_thread(self.w3.eth.gas_price)

        # Build transaction
        transaction = transfer_function.build_transaction({
            "from": self.hot_wallet,
            "gas": gas_limit,
            "gasPrice": gas_price,
            "nonce": nonce,
            "chainId": 137  # Polygon mainnet
        })

        # Calculate gas cost in MATIC
        gas_cost_wei = gas_limit * gas_price
        gas_cost_matic = gas_cost_wei / 1e18

        logger.info(
            f"Transaction built: gas_limit={gas_limit}, "
            f"gas_price={gas_price / 1e9:.2f} gwei, "
            f"cost=~{gas_cost_matic:.4f} MATIC"
        )

        # DRY RUN: Log but don't send
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would transfer ${amount_dollars:.2f} USDC "
                f"(gas: ~{gas_cost_matic:.4f} MATIC)"
            )
            return None

        # Record to database BEFORE sending
        transfer_id = await self.db.record_transfer(
            session_id=session_id,
            amount=amount_dollars,
            tx_hash=None,
            status="pending",
            trigger_reason=trigger_reason
        )

        # Sign transaction
        signed_txn = self.account.sign_transaction(transaction)

        # Send with retry logic
        tx_hash = await self._send_with_retry(
            signed_txn,
            transfer_id,
            amount_dollars
        )

        if tx_hash:
            # Monitor receipt in background (async task)
            asyncio.create_task(
                self._monitor_receipt(tx_hash, transfer_id, amount_dollars)
            )

        return tx_hash

    async def _send_with_retry(
        self,
        signed_txn,
        transfer_id: int,
        amount: float
    ) -> Optional[str]:
        """
        Send transaction with exponential backoff retry.

        Args:
            signed_txn: Signed transaction
            transfer_id: Database transfer ID
            amount: Transfer amount (for logging)

        Returns:
            Transaction hash if successful, None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                # Send transaction
                tx_hash_bytes = await asyncio.to_thread(
                    self.w3.eth.send_raw_transaction,
                    signed_txn.rawTransaction
                )
                tx_hash = tx_hash_bytes.hex()

                logger.info(
                    f"Transaction sent: {tx_hash} (attempt {attempt + 1})"
                )

                # Update database
                await self.db.update_transfer_status(
                    transfer_id=transfer_id,
                    status="sent",
                    tx_hash=tx_hash
                )

                return tx_hash

            except Exception as e:
                logger.error(
                    f"Send failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                # Update retry count
                await self.db.update_transfer_status(
                    transfer_id=transfer_id,
                    status="pending",
                    error_message=str(e)
                )

                # Retry with exponential backoff
                if attempt < self.max_retries - 1:
                    backoff = self.base_backoff * (2 ** attempt)
                    logger.info(f"Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                else:
                    # Final failure
                    logger.error(
                        f"Transfer failed after {self.max_retries} attempts"
                    )
                    await self.db.update_transfer_status(
                        transfer_id=transfer_id,
                        status="failed",
                        error_message=f"Failed after {self.max_retries} retries: {e}"
                    )

                    # Alert on Discord
                    if self.discord:
                        await self.discord.send_error(
                            message=f"❌ Profit transfer failed after {self.max_retries} attempts",
                            details=f"Amount: ${amount:.2f}\nError: {e}"
                        )

        return None

    async def _monitor_receipt(
        self,
        tx_hash: str,
        transfer_id: int,
        amount: float
    ) -> None:
        """
        Monitor transaction confirmation (background task).

        Args:
            tx_hash: Transaction hash
            transfer_id: Database transfer ID
            amount: Transfer amount
        """
        timeout = 300  # 5 minutes
        poll_interval = 5
        elapsed = 0

        logger.info(f"Monitoring receipt for {tx_hash}...")

        while elapsed < timeout:
            try:
                receipt = await asyncio.to_thread(
                    self.w3.eth.get_transaction_receipt, tx_hash
                )

                if receipt:
                    status = receipt.get("status")
                    gas_used = receipt.get("gasUsed", 0)

                    if status == 1:
                        # Success
                        logger.info(
                            f"✅ Transfer confirmed: ${amount:.2f} USDC "
                            f"(gas: {gas_used})"
                        )

                        await self.db.update_transfer_status(
                            transfer_id=transfer_id,
                            status="confirmed",
                            tx_hash=tx_hash,
                            gas_used=gas_used
                        )

                        # Alert on Discord
                        if self.discord:
                            await self.discord.send_profit_transfer(
                                amount=amount,
                                tx_hash=tx_hash,
                                gas_used=gas_used
                            )

                    else:
                        # Reverted
                        logger.error(f"❌ Transfer reverted: {tx_hash}")
                        await self.db.update_transfer_status(
                            transfer_id=transfer_id,
                            status="failed",
                            error_message="Transaction reverted"
                        )

                    return

            except Exception as e:
                logger.warning(f"Receipt query failed: {e}")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout
        logger.warning(f"⏱ Receipt timeout for {tx_hash}")
        await self.db.update_transfer_status(
            transfer_id=transfer_id,
            status="timeout",
            error_message=f"Receipt not found after {timeout}s"
        )
