#!/usr/bin/env python3
"""
Polygon Chain Client - On-chain transaction history via Polygonscan API.

Provides methods to fetch:
- Wallet transaction history
- USDC token transfers
- Internal transactions (contract calls)
- Transaction details
- Current USDC balance (via Web3 RPC)

Uses Polygonscan API for rich transaction metadata.
"""
import os
import asyncio
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

logger = logging.getLogger(__name__)

# Etherscan V2 API (unified endpoint for all chains)
# Legacy api.polygonscan.com is deprecated as of August 2025
ETHERSCAN_V2_API = "https://api.etherscan.io/v2/api"
POLYGON_CHAIN_ID = 137

# USDC.e token contract on Polygon (Bridged USDC from Ethereum)
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_DECIMALS = 6

# Minimal ERC-20 ABI
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]


@dataclass
class Transaction:
    """Represents a blockchain transaction."""
    tx_hash: str
    block_number: int
    timestamp: int
    from_address: str
    to_address: str
    value: float  # In native token (MATIC)
    gas_used: int
    gas_price: int
    is_error: bool
    method_id: str = ""
    function_name: str = ""
    contract_address: str = ""


@dataclass
class TokenTransfer:
    """Represents an ERC-20 token transfer."""
    tx_hash: str
    block_number: int
    timestamp: int
    from_address: str
    to_address: str
    value: float  # In token units
    token_symbol: str
    token_name: str
    token_decimal: int
    contract_address: str


class PolygonChainClient:
    """
    Client for querying Polygon blockchain data.

    Uses Polygonscan API for transaction history and
    Web3 RPC for balance queries.
    """

    def __init__(
        self,
        wallet_address: Optional[str] = None,
        polygonscan_api_key: Optional[str] = None,
        rpc_url: Optional[str] = None
    ):
        """
        Initialize the Polygon chain client.

        Args:
            wallet_address: Wallet to query (or from POLYMARKET_FUNDER_ADDRESS env)
            polygonscan_api_key: Polygonscan API key (or from POLYGONSCAN_API_KEY env)
            rpc_url: Polygon RPC URL (or from POLYGON_RPC_URL env)
        """
        self.wallet_address = wallet_address or os.getenv("POLYMARKET_FUNDER_ADDRESS")
        self.api_key = polygonscan_api_key or os.getenv("POLYGONSCAN_API_KEY")
        self.rpc_url = rpc_url or os.getenv("POLYGON_RPC_URL")

        # Initialize Web3 for balance queries
        self.w3: Optional[Web3] = None
        if self.rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if self.wallet_address:
            try:
                self.wallet_address = Web3.to_checksum_address(self.wallet_address)
            except ValueError:
                logger.error(f"Invalid wallet address: {self.wallet_address}")
                self.wallet_address = None

    async def _polygonscan_request(
        self,
        module: str,
        action: str,
        **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Make a request to Etherscan V2 API (unified endpoint for Polygon).

        Args:
            module: API module (account, transaction, etc.)
            action: API action
            **kwargs: Additional parameters

        Returns:
            List of results or None on error
        """
        if not self.api_key:
            logger.error("POLYGONSCAN_API_KEY not set")
            return None

        # Etherscan V2 uses unified endpoint with chainid parameter
        params = {
            "chainid": POLYGON_CHAIN_ID,
            "module": module,
            "action": action,
            "apikey": self.api_key,
            **kwargs
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(ETHERSCAN_V2_API, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Etherscan V2 API error: {resp.status}")
                        return None

                    data = await resp.json()

                    if data.get("status") != "1":
                        msg = data.get("message", "Unknown error")
                        # "No transactions found" is not an error
                        if "No transactions found" not in msg:
                            logger.warning(f"Etherscan V2: {msg}")
                        return []

                    return data.get("result", [])

        except Exception as e:
            logger.error(f"Polygonscan request failed: {e}")
            return None

    async def get_wallet_transactions(
        self,
        limit: int = 50,
        start_block: int = 0,
        end_block: int = 99999999
    ) -> List[Dict[str, Any]]:
        """
        Get all transactions for the wallet.

        Args:
            limit: Maximum transactions to return
            start_block: Start block (default: genesis)
            end_block: End block (default: latest)

        Returns:
            List of transaction records
        """
        if not self.wallet_address:
            return []

        results = await self._polygonscan_request(
            module="account",
            action="txlist",
            address=self.wallet_address,
            startblock=start_block,
            endblock=end_block,
            page=1,
            offset=limit,
            sort="desc"
        )

        if not results:
            return []

        # Enrich with readable data
        transactions = []
        for tx in results:
            transactions.append({
                "tx_hash": tx.get("hash"),
                "block_number": int(tx.get("blockNumber", 0)),
                "timestamp": int(tx.get("timeStamp", 0)),
                "datetime": datetime.fromtimestamp(
                    int(tx.get("timeStamp", 0)), tz=timezone.utc
                ).isoformat(),
                "from": tx.get("from"),
                "to": tx.get("to"),
                "value_matic": int(tx.get("value", 0)) / 1e18,
                "gas_used": int(tx.get("gasUsed", 0)),
                "gas_price_gwei": int(tx.get("gasPrice", 0)) / 1e9,
                "is_error": tx.get("isError") == "1",
                "method_id": tx.get("methodId", ""),
                "function_name": tx.get("functionName", "").split("(")[0],
                "contract_address": tx.get("contractAddress", ""),
                "confirmations": int(tx.get("confirmations", 0))
            })

        return transactions

    async def get_usdc_transfers(
        self,
        limit: int = 50,
        start_block: int = 0,
        end_block: int = 99999999
    ) -> List[Dict[str, Any]]:
        """
        Get USDC token transfers for the wallet.

        Args:
            limit: Maximum transfers to return
            start_block: Start block (default: genesis)
            end_block: End block (default: latest)

        Returns:
            List of USDC transfer records
        """
        if not self.wallet_address:
            return []

        results = await self._polygonscan_request(
            module="account",
            action="tokentx",
            address=self.wallet_address,
            contractaddress=USDC_ADDRESS,
            startblock=start_block,
            endblock=end_block,
            page=1,
            offset=limit,
            sort="desc"
        )

        if not results:
            return []

        transfers = []
        for tx in results:
            value_raw = int(tx.get("value", 0))
            decimals = int(tx.get("tokenDecimal", USDC_DECIMALS))
            value = value_raw / (10 ** decimals)

            # Determine direction
            from_addr = tx.get("from", "").lower()
            to_addr = tx.get("to", "").lower()
            wallet_lower = self.wallet_address.lower()

            if from_addr == wallet_lower:
                direction = "OUT"
            elif to_addr == wallet_lower:
                direction = "IN"
            else:
                direction = "UNKNOWN"

            transfers.append({
                "tx_hash": tx.get("hash"),
                "block_number": int(tx.get("blockNumber", 0)),
                "timestamp": int(tx.get("timeStamp", 0)),
                "datetime": datetime.fromtimestamp(
                    int(tx.get("timeStamp", 0)), tz=timezone.utc
                ).isoformat(),
                "from": from_addr,
                "to": to_addr,
                "direction": direction,
                "value_usdc": value,
                "token_symbol": tx.get("tokenSymbol", "USDC"),
                "token_name": tx.get("tokenName", "USD Coin"),
                "confirmations": int(tx.get("confirmations", 0))
            })

        return transfers

    async def get_internal_transactions(
        self,
        limit: int = 50,
        start_block: int = 0,
        end_block: int = 99999999
    ) -> List[Dict[str, Any]]:
        """
        Get internal transactions (contract calls) for the wallet.

        Args:
            limit: Maximum transactions to return
            start_block: Start block (default: genesis)
            end_block: End block (default: latest)

        Returns:
            List of internal transaction records
        """
        if not self.wallet_address:
            return []

        results = await self._polygonscan_request(
            module="account",
            action="txlistinternal",
            address=self.wallet_address,
            startblock=start_block,
            endblock=end_block,
            page=1,
            offset=limit,
            sort="desc"
        )

        if not results:
            return []

        transactions = []
        for tx in results:
            transactions.append({
                "tx_hash": tx.get("hash"),
                "block_number": int(tx.get("blockNumber", 0)),
                "timestamp": int(tx.get("timeStamp", 0)),
                "datetime": datetime.fromtimestamp(
                    int(tx.get("timeStamp", 0)), tz=timezone.utc
                ).isoformat(),
                "from": tx.get("from"),
                "to": tx.get("to"),
                "value_matic": int(tx.get("value", 0)) / 1e18,
                "type": tx.get("type", "call"),
                "is_error": tx.get("isError") == "1",
                "error_code": tx.get("errCode", ""),
                "contract_address": tx.get("contractAddress", "")
            })

        return transactions

    async def get_transaction_details(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get details of a specific transaction.

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction details or None
        """
        results = await self._polygonscan_request(
            module="proxy",
            action="eth_getTransactionByHash",
            txhash=tx_hash
        )

        if not results:
            return None

        # Also get receipt for gas used
        receipt = await self._polygonscan_request(
            module="proxy",
            action="eth_getTransactionReceipt",
            txhash=tx_hash
        )

        tx = results if isinstance(results, dict) else {}
        rx = receipt if isinstance(receipt, dict) else {}

        return {
            "tx_hash": tx_hash,
            "block_number": int(tx.get("blockNumber", "0x0"), 16),
            "from": tx.get("from"),
            "to": tx.get("to"),
            "value_matic": int(tx.get("value", "0x0"), 16) / 1e18,
            "gas_limit": int(tx.get("gas", "0x0"), 16),
            "gas_price_gwei": int(tx.get("gasPrice", "0x0"), 16) / 1e9,
            "gas_used": int(rx.get("gasUsed", "0x0"), 16),
            "status": "success" if rx.get("status") == "0x1" else "failed",
            "input_data": tx.get("input", "0x")[:66] + "..." if len(tx.get("input", "")) > 66 else tx.get("input", "0x")
        }

    async def get_usdc_balance(self) -> Optional[float]:
        """
        Get current USDC balance via Web3 RPC.

        Returns:
            Balance in USDC (float) or None
        """
        if not self.w3 or not self.wallet_address:
            return None

        try:
            usdc_address = Web3.to_checksum_address(USDC_ADDRESS)
            usdc_contract = self.w3.eth.contract(
                address=usdc_address,
                abi=ERC20_ABI
            )

            balance_raw = await asyncio.to_thread(
                usdc_contract.functions.balanceOf(self.wallet_address).call
            )

            return balance_raw / (10 ** USDC_DECIMALS)

        except Exception as e:
            logger.error(f"Failed to get USDC balance: {e}")
            return None

    async def get_matic_balance(self) -> Optional[float]:
        """
        Get current MATIC balance via Web3 RPC.

        Returns:
            Balance in MATIC (float) or None
        """
        if not self.w3 or not self.wallet_address:
            return None

        try:
            balance_wei = await asyncio.to_thread(
                self.w3.eth.get_balance,
                self.wallet_address
            )

            return balance_wei / 1e18

        except Exception as e:
            logger.error(f"Failed to get MATIC balance: {e}")
            return None

    async def get_wallet_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive wallet summary.

        Returns:
            Dict with balances and recent activity
        """
        if not self.wallet_address:
            return {"error": "No wallet address configured"}

        # Fetch all data in parallel
        usdc_balance, matic_balance, recent_txs, recent_usdc = await asyncio.gather(
            self.get_usdc_balance(),
            self.get_matic_balance(),
            self.get_wallet_transactions(limit=10),
            self.get_usdc_transfers(limit=10),
            return_exceptions=True
        )

        return {
            "wallet_address": self.wallet_address,
            "balances": {
                "usdc": usdc_balance if not isinstance(usdc_balance, Exception) else None,
                "matic": matic_balance if not isinstance(matic_balance, Exception) else None
            },
            "recent_transactions": recent_txs if not isinstance(recent_txs, Exception) else [],
            "recent_usdc_transfers": recent_usdc if not isinstance(recent_usdc, Exception) else [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def is_configured(self) -> bool:
        """Check if client is properly configured."""
        return bool(self.wallet_address and self.api_key)


# Convenience function
def create_polygon_client() -> PolygonChainClient:
    """
    Create a Polygon chain client using environment variables.

    Environment variables:
        POLYMARKET_FUNDER_ADDRESS: Wallet address to query
        POLYGONSCAN_API_KEY: Polygonscan API key
        POLYGON_RPC_URL: Polygon RPC endpoint
    """
    return PolygonChainClient()


if __name__ == "__main__":
    import asyncio

    async def test():
        client = create_polygon_client()

        if not client.is_configured():
            print("Client not configured - check environment variables")
            return

        print(f"Querying wallet: {client.wallet_address}")

        # Test USDC balance
        balance = await client.get_usdc_balance()
        print(f"\nUSDC Balance: ${balance:.2f}" if balance else "Balance unavailable")

        # Test MATIC balance
        matic = await client.get_matic_balance()
        print(f"MATIC Balance: {matic:.4f}" if matic else "MATIC unavailable")

        # Test USDC transfers
        transfers = await client.get_usdc_transfers(limit=5)
        print(f"\nRecent USDC transfers ({len(transfers)}):")
        for t in transfers[:3]:
            print(f"  {t['direction']} ${t['value_usdc']:.2f} | {t['datetime']}")

        # Test transactions
        txs = await client.get_wallet_transactions(limit=5)
        print(f"\nRecent transactions ({len(txs)}):")
        for tx in txs[:3]:
            print(f"  {tx['function_name'] or 'transfer'} | {tx['datetime']}")

    asyncio.run(test())
