#!/usr/bin/env python3
"""
Autonomy Manager - Full 24/7 autonomous trading support.

Handles:
1. Auto-redemption of resolved positions (winning markets)
2. POL gas balance monitoring and alerts
3. Health checks for all critical components

Polymarket uses Gnosis CTF (Conditional Tokens Framework):
- CTF Contract: 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045 (Polygon)
- Positions are ERC1155 tokens that can be redeemed after market resolution
"""
import asyncio
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from web3 import Web3
from web3.contract import Contract

logger = logging.getLogger(__name__)

# Contract addresses on Polygon
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # Gnosis CTF
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # Polymarket Exchange
NEG_RISK_ADAPTER = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # NegRisk Exchange
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e (bridged)

# Minimum POL for gas (in POL tokens)
MIN_POL_BALANCE = 1.0  # Alert if below 1 POL
CRITICAL_POL_BALANCE = 0.1  # Critical if below 0.1 POL

# CTF ABI (minimal for redemption)
CTF_ABI = [
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"}
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "name": "payoutDenominator",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"}
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]


class GasStatus(Enum):
    OK = "ok"
    LOW = "low"
    CRITICAL = "critical"


@dataclass
class HealthStatus:
    """System health status."""
    timestamp: datetime
    rpc_connected: bool
    gas_status: GasStatus
    pol_balance: float
    usdc_balance: float
    pending_redemptions: int
    websocket_connected: bool
    last_trade_time: Optional[datetime]
    errors: List[str]

    @property
    def is_healthy(self) -> bool:
        return (
            self.rpc_connected and
            self.gas_status != GasStatus.CRITICAL and
            len(self.errors) == 0
        )


class AutonomyManager:
    """
    Manages autonomous operations for 24/7 trading.

    Features:
    - Auto-redeem resolved positions
    - Monitor gas balance
    - Health checks
    - Discord alerts for critical issues
    """

    def __init__(
        self,
        rpc_url: str,
        wallet_address: str,
        private_key: Optional[str] = None,
        discord_webhook: Optional[str] = None
    ):
        """
        Initialize autonomy manager.

        Args:
            rpc_url: Polygon RPC endpoint
            wallet_address: Hot wallet address
            private_key: Private key for signing redemption txs (optional)
            discord_webhook: Discord webhook for alerts (optional)
        """
        self.rpc_url = rpc_url
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.wallet_address = Web3.to_checksum_address(wallet_address)
        self.private_key = private_key
        self.discord_webhook = discord_webhook

        # Initialize contracts
        self.ctf_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(CTF_CONTRACT),
            abi=CTF_ABI
        )

        # State tracking
        self.last_health_check: Optional[HealthStatus] = None
        self.pending_redemptions: List[Dict[str, Any]] = []
        self.running = False

        logger.info(f"AutonomyManager initialized for wallet: {wallet_address[:10]}...")

    async def get_pol_balance(self) -> float:
        """Get native POL balance for gas."""
        try:
            balance_wei = await asyncio.to_thread(
                self.w3.eth.get_balance,
                self.wallet_address
            )
            return float(self.w3.from_wei(balance_wei, 'ether'))
        except Exception as e:
            logger.error(f"Failed to get POL balance: {e}")
            return 0.0

    async def get_usdc_balance(self) -> float:
        """Get USDC.e balance."""
        try:
            usdc_abi = [{
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            }]
            usdc_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(USDC_E),
                abi=usdc_abi
            )
            balance = await asyncio.to_thread(
                usdc_contract.functions.balanceOf(self.wallet_address).call
            )
            return balance / 10**6
        except Exception as e:
            logger.error(f"Failed to get USDC balance: {e}")
            return 0.0

    def check_gas_status(self, pol_balance: float) -> GasStatus:
        """Determine gas status based on POL balance."""
        if pol_balance < CRITICAL_POL_BALANCE:
            return GasStatus.CRITICAL
        elif pol_balance < MIN_POL_BALANCE:
            return GasStatus.LOW
        return GasStatus.OK

    async def check_health(self) -> HealthStatus:
        """
        Perform comprehensive health check.

        Returns:
            HealthStatus with all component statuses
        """
        errors = []

        # Check RPC connection
        try:
            rpc_connected = await asyncio.to_thread(self.w3.is_connected)
        except Exception as e:
            rpc_connected = False
            errors.append(f"RPC connection failed: {e}")

        # Get balances
        pol_balance = await self.get_pol_balance()
        usdc_balance = await self.get_usdc_balance()

        # Check gas status
        gas_status = self.check_gas_status(pol_balance)
        if gas_status == GasStatus.CRITICAL:
            errors.append(f"CRITICAL: POL balance too low ({pol_balance:.4f} POL)")
        elif gas_status == GasStatus.LOW:
            logger.warning(f"Low POL balance: {pol_balance:.4f} POL")

        status = HealthStatus(
            timestamp=datetime.now(timezone.utc),
            rpc_connected=rpc_connected,
            gas_status=gas_status,
            pol_balance=pol_balance,
            usdc_balance=usdc_balance,
            pending_redemptions=len(self.pending_redemptions),
            websocket_connected=True,  # TODO: Check actual WSS status
            last_trade_time=None,  # TODO: Get from DB
            errors=errors
        )

        self.last_health_check = status
        return status

    async def find_redeemable_positions(self) -> List[Dict[str, Any]]:
        """
        Find positions that have resolved and can be redeemed.

        This queries the Polymarket API for resolved markets where
        we hold winning positions.

        Returns:
            List of redeemable position dicts with condition_id, token_id, etc.
        """
        # TODO: Implement by querying Polymarket API for user's positions
        # and checking which markets have resolved
        #
        # For now, return empty list - this needs integration with
        # the trading database to know what positions we hold
        logger.debug("Checking for redeemable positions...")
        return []

    async def redeem_position(
        self,
        condition_id: str,
        index_sets: List[int]
    ) -> Optional[str]:
        """
        Redeem a resolved position.

        Args:
            condition_id: The condition ID (market ID)
            index_sets: Which outcomes to redeem [1, 2] for both

        Returns:
            Transaction hash if successful, None otherwise
        """
        if not self.private_key:
            logger.warning("Cannot redeem: no private key configured")
            return None

        try:
            # Build transaction
            parent_collection_id = bytes(32)  # Null for Polymarket

            tx = self.ctf_contract.functions.redeemPositions(
                USDC_E,
                parent_collection_id,
                bytes.fromhex(condition_id[2:] if condition_id.startswith('0x') else condition_id),
                index_sets
            ).build_transaction({
                'from': self.wallet_address,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address),
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            })

            # Sign and send
            signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)

            logger.info(f"Redemption tx sent: {tx_hash.hex()}")
            return tx_hash.hex()

        except Exception as e:
            logger.error(f"Redemption failed: {e}")
            return None

    async def send_alert(self, message: str, level: str = "warning"):
        """Send alert to Discord webhook."""
        if not self.discord_webhook:
            logger.warning(f"Alert (no webhook): {message}")
            return

        try:
            import aiohttp

            color = {
                "info": 0x3498db,
                "warning": 0xf39c12,
                "error": 0xe74c3c,
                "critical": 0x8e44ad
            }.get(level, 0x95a5a6)

            payload = {
                "embeds": [{
                    "title": f"Autonomy Alert ({level.upper()})",
                    "description": message,
                    "color": color,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }]
            }

            async with aiohttp.ClientSession() as session:
                await session.post(self.discord_webhook, json=payload)

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def run_maintenance_loop(self, interval: int = 300):
        """
        Run periodic maintenance tasks.

        Args:
            interval: Seconds between maintenance runs (default: 5 min)
        """
        self.running = True
        logger.info(f"Starting autonomy maintenance loop (interval: {interval}s)")

        while self.running:
            try:
                # Health check
                health = await self.check_health()

                if not health.is_healthy:
                    await self.send_alert(
                        f"System unhealthy:\n" + "\n".join(health.errors),
                        level="error"
                    )
                elif health.gas_status == GasStatus.LOW:
                    await self.send_alert(
                        f"Low gas: {health.pol_balance:.4f} POL remaining",
                        level="warning"
                    )

                # Check for redeemable positions
                redeemable = await self.find_redeemable_positions()
                if redeemable:
                    logger.info(f"Found {len(redeemable)} redeemable positions")
                    for pos in redeemable:
                        tx_hash = await self.redeem_position(
                            pos['condition_id'],
                            pos['index_sets']
                        )
                        if tx_hash:
                            await self.send_alert(
                                f"Redeemed position: {tx_hash}",
                                level="info"
                            )

                # Log status
                logger.info(
                    f"Autonomy status: POL={health.pol_balance:.4f}, "
                    f"USDC=${health.usdc_balance:.2f}, "
                    f"gas_status={health.gas_status.value}"
                )

            except Exception as e:
                logger.error(f"Maintenance loop error: {e}", exc_info=True)

            await asyncio.sleep(interval)

    def stop(self):
        """Stop the maintenance loop."""
        self.running = False


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        rpc_url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
        wallet = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")

        if not wallet:
            print("Set POLYMARKET_FUNDER_ADDRESS env var")
            return

        manager = AutonomyManager(rpc_url, wallet)
        health = await manager.check_health()

        print(f"\n=== Health Check ===")
        print(f"RPC Connected: {health.rpc_connected}")
        print(f"POL Balance: {health.pol_balance:.4f}")
        print(f"USDC Balance: ${health.usdc_balance:.2f}")
        print(f"Gas Status: {health.gas_status.value}")
        print(f"Healthy: {health.is_healthy}")
        if health.errors:
            print(f"Errors: {health.errors}")

    asyncio.run(test())
