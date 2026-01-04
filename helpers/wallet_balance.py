#!/usr/bin/env python3
"""
Wallet balance tracker for Polygon blockchain.

Queries real-time USDC balance from Polygon using Web3.py.
"""
import asyncio
import logging
from typing import Optional

from web3 import Web3
from web3.contract import Contract

logger = logging.getLogger(__name__)

# USDC token contract on Polygon
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_DECIMALS = 6

# Minimal ERC-20 ABI (only what we need)
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]


class WalletBalanceTracker:
    """
    Query USDC balance from Polygon blockchain.

    Usage:
        tracker = WalletBalanceTracker(rpc_url, wallet_address)
        balance = await tracker.get_balance()  # Returns float dollars
    """

    def __init__(self, rpc_url: str, wallet_address: str):
        """
        Initialize balance tracker.

        Args:
            rpc_url: Polygon RPC endpoint (e.g., Alchemy, Infura)
            wallet_address: Wallet address to track
        """
        self.rpc_url = rpc_url
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))

        # Validate RPC connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Polygon RPC: {rpc_url}")

        # Checksum address (validates format)
        try:
            self.wallet_address = Web3.to_checksum_address(wallet_address)
        except ValueError as e:
            raise ValueError(f"Invalid wallet address: {wallet_address}") from e

        # Initialize USDC contract
        self.usdc_address = Web3.to_checksum_address(USDC_ADDRESS)
        self.usdc_contract: Contract = self.w3.eth.contract(
            address=self.usdc_address,
            abi=ERC20_ABI
        )

        logger.info(f"WalletBalanceTracker initialized: {self.wallet_address[:10]}...")

    async def get_balance(self) -> float:
        """
        Get current USDC balance in dollars.

        Returns:
            Balance as float (e.g., 1234.56 = $1,234.56)

        Raises:
            Exception: If RPC call fails
        """
        try:
            # Call balanceOf in thread (web3.py is synchronous)
            balance_raw = await asyncio.to_thread(
                self.usdc_contract.functions.balanceOf(self.wallet_address).call
            )

            # Convert from 6 decimals to float dollars
            balance_dollars = balance_raw / (10 ** USDC_DECIMALS)

            return float(balance_dollars)

        except Exception as e:
            logger.error(f"Failed to query USDC balance: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Test RPC connectivity.

        Returns:
            True if RPC is responding
        """
        try:
            block_number = await asyncio.to_thread(
                lambda: self.w3.eth.block_number
            )
            is_healthy = block_number > 0

            if is_healthy:
                logger.info(f"RPC health check OK (block: {block_number})")
            else:
                logger.warning("RPC health check failed: block number is 0")

            return is_healthy

        except Exception as e:
            logger.error(f"RPC health check failed: {e}")
            return False

    def get_latest_block(self) -> int:
        """Get latest block number (synchronous)."""
        return self.w3.eth.block_number

    def is_connected(self) -> bool:
        """Check if Web3 provider is connected."""
        return self.w3.is_connected()
