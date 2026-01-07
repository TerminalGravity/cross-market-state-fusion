#!/usr/bin/env python3
"""
Position Redeemer - Automatically redeem resolved Polymarket positions.

Queries Polymarket Data API for redeemable positions and executes
redemption transactions via the Gnosis CTF contract.

Key addresses (Polygon):
- CTF Contract: 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
- USDC.e Collateral: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
"""
import asyncio
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass
import aiohttp

from web3 import Web3
from eth_account import Account

logger = logging.getLogger(__name__)

# Contract addresses on Polygon
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# Polymarket Data API
DATA_API = "https://data-api.polymarket.com"

# CTF ABI for redemption
CTF_REDEEM_ABI = [
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
            {"name": "conditionId", "type": "bytes32"},
            {"name": "outcomeSlotCount", "type": "uint256"}
        ],
        "name": "payoutNumerators",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]


@dataclass
class RedeemablePosition:
    """A position that can be redeemed."""
    condition_id: str
    asset_id: str
    title: str
    outcome: str
    outcome_index: int
    size: float  # Number of shares
    current_value: float  # USD value if redeemed
    is_winning: bool
    negative_risk: bool


@dataclass
class RedemptionResult:
    """Result of a redemption attempt."""
    condition_id: str
    success: bool
    tx_hash: Optional[str]
    amount_redeemed: float
    error: Optional[str]


class PositionRedeemer:
    """
    Automatically redeem resolved Polymarket positions.

    Usage:
        redeemer = PositionRedeemer(rpc_url, wallet_address, private_key)
        results = await redeemer.redeem_all_winning()
    """

    def __init__(
        self,
        rpc_url: str,
        wallet_address: str,
        private_key: str,
        discord_webhook: Optional[str] = None
    ):
        """
        Initialize position redeemer.

        Args:
            rpc_url: Polygon RPC endpoint
            wallet_address: Wallet address holding positions
            private_key: Private key for signing transactions
            discord_webhook: Optional webhook for notifications
        """
        self.rpc_url = rpc_url
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.wallet_address = Web3.to_checksum_address(wallet_address)
        self.private_key = private_key
        self.discord_webhook = discord_webhook
        self.account = Account.from_key(private_key)
        self.running = False

        # Initialize CTF contract
        self.ctf_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(CTF_CONTRACT),
            abi=CTF_REDEEM_ABI
        )

        logger.info(f"PositionRedeemer initialized for wallet: {wallet_address[:10]}...")

    async def get_redeemable_positions(self) -> List[RedeemablePosition]:
        """
        Query Polymarket Data API for redeemable positions.

        Returns:
            List of positions that can be redeemed
        """
        url = f"{DATA_API}/positions"
        params = {
            "user": self.wallet_address,
            "redeemable": "true",
            "limit": 100
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Data API error: {resp.status}")
                        return []

                    data = await resp.json()

            positions = []
            for pos in data:
                # Only include positions with value (winning positions)
                current_value = float(pos.get('currentValue', 0))
                cur_price = float(pos.get('curPrice', 0))

                position = RedeemablePosition(
                    condition_id=pos['conditionId'],
                    asset_id=pos['asset'],
                    title=pos['title'],
                    outcome=pos['outcome'],
                    outcome_index=pos['outcomeIndex'],
                    size=float(pos['size']),
                    current_value=current_value,
                    is_winning=(cur_price == 1),
                    negative_risk=pos.get('negativeRisk', False)
                )
                positions.append(position)

            logger.info(f"Found {len(positions)} redeemable positions")
            return positions

        except Exception as e:
            logger.error(f"Failed to query positions: {e}")
            return []

    async def check_payout_reported(self, condition_id: str) -> bool:
        """
        Check if payouts have been reported for a condition.

        Args:
            condition_id: The condition ID (hex string with 0x prefix)

        Returns:
            True if payouts are reported and ready to redeem
        """
        try:
            condition_bytes = bytes.fromhex(
                condition_id[2:] if condition_id.startswith('0x') else condition_id
            )

            denominator = await asyncio.to_thread(
                self.ctf_contract.functions.payoutDenominator(condition_bytes).call
            )

            # Denominator > 0 means payouts are reported
            return denominator > 0

        except Exception as e:
            logger.error(f"Failed to check payout for {condition_id[:10]}...: {e}")
            return False

    async def redeem_position(self, position: RedeemablePosition) -> RedemptionResult:
        """
        Redeem a single position.

        Args:
            position: The position to redeem

        Returns:
            RedemptionResult with success status and tx hash
        """
        logger.info(f"Redeeming: {position.title} ({position.outcome}) - ${position.current_value:.2f}")

        try:
            # Check if payout is reported
            if not await self.check_payout_reported(position.condition_id):
                return RedemptionResult(
                    condition_id=position.condition_id,
                    success=False,
                    tx_hash=None,
                    amount_redeemed=0,
                    error="Payout not yet reported"
                )

            # Prepare transaction
            condition_bytes = bytes.fromhex(
                position.condition_id[2:] if position.condition_id.startswith('0x') else position.condition_id
            )

            # Parent collection ID is 0 for Polymarket
            parent_collection_id = bytes(32)

            # Index sets: [1, 2] for both outcomes in a binary market
            # This redeems whichever outcome(s) we hold
            index_sets = [1, 2]

            # Build transaction
            nonce = await asyncio.to_thread(
                self.w3.eth.get_transaction_count,
                self.wallet_address
            )

            gas_price = await asyncio.to_thread(
                lambda: self.w3.eth.gas_price
            )

            # Estimate gas
            try:
                gas_estimate = await asyncio.to_thread(
                    self.ctf_contract.functions.redeemPositions(
                        Web3.to_checksum_address(USDC_E),
                        parent_collection_id,
                        condition_bytes,
                        index_sets
                    ).estimate_gas,
                    {'from': self.wallet_address}
                )
                gas_limit = int(gas_estimate * 1.3)  # 30% buffer
            except Exception as e:
                logger.warning(f"Gas estimation failed, using default: {e}")
                gas_limit = 300000

            tx = self.ctf_contract.functions.redeemPositions(
                Web3.to_checksum_address(USDC_E),
                parent_collection_id,
                condition_bytes,
                index_sets
            ).build_transaction({
                'from': self.wallet_address,
                'nonce': nonce,
                'gas': gas_limit,
                'gasPrice': gas_price,
                'chainId': 137  # Polygon
            })

            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = await asyncio.to_thread(
                self.w3.eth.send_raw_transaction,
                signed_tx.raw_transaction
            )

            tx_hash_hex = tx_hash.hex()
            logger.info(f"Redemption tx sent: {tx_hash_hex}")

            # Wait for confirmation
            receipt = await asyncio.to_thread(
                self.w3.eth.wait_for_transaction_receipt,
                tx_hash,
                timeout=120
            )

            if receipt['status'] == 1:
                logger.info(f"Redemption confirmed: {tx_hash_hex}")
                return RedemptionResult(
                    condition_id=position.condition_id,
                    success=True,
                    tx_hash=tx_hash_hex,
                    amount_redeemed=position.current_value,
                    error=None
                )
            else:
                logger.error(f"Redemption failed (reverted): {tx_hash_hex}")
                return RedemptionResult(
                    condition_id=position.condition_id,
                    success=False,
                    tx_hash=tx_hash_hex,
                    amount_redeemed=0,
                    error="Transaction reverted"
                )

        except Exception as e:
            logger.error(f"Redemption error for {position.condition_id[:10]}...: {e}")
            return RedemptionResult(
                condition_id=position.condition_id,
                success=False,
                tx_hash=None,
                amount_redeemed=0,
                error=str(e)
            )

    async def cleanup_dead_positions(self) -> List[RedemptionResult]:
        """
        Clean up dead (losing) positions worth $0.

        These positions clutter the account but can be cleared by calling
        redeemPositions (returns $0 but removes from account).

        Returns:
            List of redemption results
        """
        positions = await self.get_redeemable_positions()

        # Filter to losing positions (value = 0, not winning)
        dead = [p for p in positions if p.current_value == 0 or not p.is_winning]

        if not dead:
            logger.info("No dead positions to clean up")
            return []

        logger.info(f"Found {len(dead)} dead positions to clean up")

        results = []
        for position in dead:
            # Check if payout is reported before attempting cleanup
            if not await self.check_payout_reported(position.condition_id):
                logger.debug(f"Skipping {position.condition_id[:10]}... - payout not reported")
                continue

            result = await self.redeem_position(position)
            results.append(result)

            # Small delay between redemptions
            await asyncio.sleep(2)

        successful = sum(1 for r in results if r.success)
        logger.info(f"Dead position cleanup: {successful}/{len(results)} cleaned")

        return results

    async def redeem_all_winning(self) -> List[RedemptionResult]:
        """
        Find and redeem all winning positions.

        Returns:
            List of redemption results
        """
        positions = await self.get_redeemable_positions()

        # Filter to only winning positions (value > 0)
        winning = [p for p in positions if p.current_value > 0 and p.is_winning]

        if not winning:
            logger.info("No winning positions to redeem")
            return []

        total_value = sum(p.current_value for p in winning)
        logger.info(f"Found {len(winning)} winning positions worth ${total_value:.2f}")

        results = []
        for position in winning:
            result = await self.redeem_position(position)
            results.append(result)

            # Small delay between redemptions to avoid nonce issues
            await asyncio.sleep(2)

        # Summary
        successful = [r for r in results if r.success]
        total_redeemed = sum(r.amount_redeemed for r in successful)

        logger.info(
            f"Redemption complete: {len(successful)}/{len(results)} successful, "
            f"${total_redeemed:.2f} redeemed"
        )

        # Send Discord notification if configured
        if self.discord_webhook and total_redeemed > 0:
            await self._send_discord_notification(results, total_redeemed)

        return results

    async def _send_discord_notification(
        self,
        results: List[RedemptionResult],
        total_redeemed: float
    ):
        """Send Discord notification about redemptions."""
        try:
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]

            description = f"**${total_redeemed:.2f}** redeemed from {len(successful)} positions"

            if failed:
                description += f"\n\n{len(failed)} redemptions failed"

            # Add transaction links
            if successful:
                tx_links = "\n".join([
                    f"[View Tx](https://polygonscan.com/tx/{r.tx_hash})"
                    for r in successful[:3]  # Limit to first 3
                ])
                description += f"\n\n{tx_links}"

            payload = {
                "embeds": [{
                    "title": "Position Redemption Complete",
                    "description": description,
                    "color": 0x00ff00 if not failed else 0xffaa00,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }]
            }

            async with aiohttp.ClientSession() as session:
                await session.post(self.discord_webhook, json=payload)

        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

    async def run_periodic_redemption(self, interval_minutes: int = 30):
        """
        Run periodic redemption check.

        Args:
            interval_minutes: Minutes between redemption checks
        """
        self.running = True
        logger.info(f"Starting periodic redemption (every {interval_minutes} min)")

        while self.running:
            try:
                results = await self.redeem_all_winning()

                if results:
                    successful = sum(1 for r in results if r.success)
                    total = sum(r.amount_redeemed for r in results if r.success)
                    logger.info(f"Periodic redemption: {successful} positions, ${total:.2f}")

            except Exception as e:
                logger.error(f"Periodic redemption error: {e}", exc_info=True)

            await asyncio.sleep(interval_minutes * 60)

        logger.info("Periodic redemption stopped")


# Standalone execution
async def main():
    """Test the position redeemer."""
    rpc_url = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
    wallet = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if not wallet:
        print("Set POLYMARKET_FUNDER_ADDRESS env var")
        return

    redeemer = PositionRedeemer(
        rpc_url=rpc_url,
        wallet_address=wallet,
        private_key=private_key,
        discord_webhook=discord_webhook if discord_webhook else None
    )

    # Get redeemable positions
    positions = await redeemer.get_redeemable_positions()

    print("\n=== REDEEMABLE POSITIONS ===\n")

    winning = []
    losing = []

    for pos in positions:
        status = "WIN" if pos.is_winning else "LOSS"
        value_str = f"${pos.current_value:.2f}" if pos.current_value > 0 else "$0.00"

        print(f"{status} | {value_str:>8} | {pos.title} ({pos.outcome})")

        if pos.is_winning and pos.current_value > 0:
            winning.append(pos)
        else:
            losing.append(pos)

    total_value = sum(p.current_value for p in winning)
    print(f"\nTotal redeemable: ${total_value:.2f} from {len(winning)} winning positions")
    print(f"Dead positions (worth $0): {len(losing)}")

    if not private_key:
        print("\nSet POLYMARKET_PRIVATE_KEY to enable redemption")
        return

    print("\n" + "="*50)
    print("Options:")
    print("  1. Redeem winning positions only")
    print("  2. Clean up dead (losing) positions only")
    print("  3. Both: redeem winning + clean up dead")
    print("  q. Quit")

    response = input("\nChoice (1/2/3/q): ").strip().lower()

    if response == '1' and winning:
        print(f"\nRedeeming ${total_value:.2f} from {len(winning)} positions...")
        results = await redeemer.redeem_all_winning()
        _print_results(results)

    elif response == '2' and losing:
        print(f"\nCleaning up {len(losing)} dead positions...")
        results = await redeemer.cleanup_dead_positions()
        _print_results(results)

    elif response == '3':
        if winning:
            print(f"\nRedeeming ${total_value:.2f} from {len(winning)} positions...")
            results = await redeemer.redeem_all_winning()
            _print_results(results)

        if losing:
            print(f"\nCleaning up {len(losing)} dead positions...")
            results = await redeemer.cleanup_dead_positions()
            _print_results(results)

    else:
        print("Cancelled")


def _print_results(results: List[RedemptionResult]):
    """Helper to print redemption results."""
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.condition_id[:10]}... - ${r.amount_redeemed:.2f}")
        if r.tx_hash:
            print(f"    TX: https://polygonscan.com/tx/{r.tx_hash}")
        if r.error:
            print(f"    Error: {r.error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
