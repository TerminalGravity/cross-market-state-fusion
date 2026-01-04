#!/usr/bin/env python3
"""
Set token allowances for Polymarket trading (EOA wallets only).

This is a ONE-TIME setup required before you can trade with an EOA wallet.
Approves Polymarket contracts to spend your USDC and Conditional Tokens.

Run: python scripts/set_allowances.py
"""
import os
import sys
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

# Polygon mainnet RPC
RPC_URL = os.getenv("POLYGON_RPC_URL")
PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY")

if not RPC_URL or not PRIVATE_KEY:
    print("❌ Missing POLYGON_RPC_URL or POLYMARKET_PRIVATE_KEY in .env")
    sys.exit(1)

# Connect to Polygon
w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    print("❌ Failed to connect to Polygon RPC")
    sys.exit(1)

# Get account from private key
account = w3.eth.account.from_key(PRIVATE_KEY)
print(f"📍 Wallet: {account.address}")

# Token contracts
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Polymarket contracts to approve (spend your tokens)
SPENDERS = [
    "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",  # Exchange
    "0xC5d563A36AE78145C45a50134d48A1215220f80a",  # Neg Risk Adapter
    "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"   # Neg Risk CTF Exchange
]

# ERC-20 ABI (just approve function)
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    }
]

# ERC-1155 ABI (for Conditional Tokens)
ERC1155_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"}
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "operator", "type": "address"}
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]

# Max uint256 for unlimited approval
MAX_UINT256 = 2**256 - 1


def check_allowance(token_address: str, spender: str) -> int:
    """Check current allowance."""
    token = w3.eth.contract(address=Web3.to_checksum_address(token_address), abi=ERC20_ABI)
    return token.functions.allowance(account.address, Web3.to_checksum_address(spender)).call()


def approve_token(token_address: str, token_name: str, spender: str, spender_name: str):
    """Approve a contract to spend tokens."""
    # Check current allowance (skip for CTF as it may not support allowance check)
    try:
        current = check_allowance(token_address, spender)
        if current >= MAX_UINT256 // 2:  # Already has high allowance
            print(f"  ✓ {token_name} → {spender_name[:20]}... (already approved)")
            return
    except Exception as e:
        # Some contracts don't support allowance check, just proceed with approval
        print(f"  ℹ️  Cannot check {token_name} allowance (proceeding anyway)")

    print(f"  ⏳ Approving {token_name} → {spender_name[:20]}...")

    # Build transaction
    token = w3.eth.contract(address=Web3.to_checksum_address(token_address), abi=ERC20_ABI)

    txn = token.functions.approve(
        Web3.to_checksum_address(spender),
        MAX_UINT256
    ).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 100000,  # Approval typically uses ~50k gas
        'gasPrice': w3.eth.gas_price
    })

    # Sign and send
    signed_txn = account.sign_transaction(txn)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

    print(f"    📤 Tx: {tx_hash.hex()}")
    print(f"    ⏳ Waiting for confirmation...")

    # Wait for receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt['status'] == 1:
        print(f"    ✅ Approved! Gas used: {receipt['gasUsed']:,}")
    else:
        print(f"    ❌ Transaction failed")
        sys.exit(1)


def approve_erc1155(operator: str, operator_name: str):
    """Approve an operator for all ERC-1155 tokens (Conditional Tokens)."""
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS), abi=ERC1155_ABI)

    # Check if already approved
    try:
        is_approved = ctf.functions.isApprovedForAll(account.address, Web3.to_checksum_address(operator)).call()
        if is_approved:
            print(f"  ✓ CTF → {operator_name[:20]}... (already approved)")
            return
    except Exception as e:
        print(f"  ℹ️  Cannot check CTF approval status (proceeding anyway)")

    print(f"  ⏳ Approving CTF → {operator_name[:20]}...")

    # Build transaction
    txn = ctf.functions.setApprovalForAll(
        Web3.to_checksum_address(operator),
        True
    ).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 100000,
        'gasPrice': w3.eth.gas_price
    })

    # Sign and send
    signed_txn = account.sign_transaction(txn)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

    print(f"    📤 Tx: {tx_hash.hex()}")
    print(f"    ⏳ Waiting for confirmation...")

    # Wait for receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt['status'] == 1:
        print(f"    ✅ Approved! Gas used: {receipt['gasUsed']:,}")
    else:
        print(f"    ❌ Transaction failed")
        sys.exit(1)


def main():
    print("\n🔧 Setting Polymarket token allowances...\n")

    # Get current balances
    print("📊 Current balances:")
    usdc_contract = w3.eth.contract(address=Web3.to_checksum_address(USDC_ADDRESS), abi=ERC20_ABI)
    # USDC has 6 decimals
    balance_raw = w3.eth.call({
        'to': Web3.to_checksum_address(USDC_ADDRESS),
        'data': w3.keccak(text='balanceOf(address)')[:4] + w3.codec.encode(['address'], [account.address])
    })
    usdc_balance = int.from_bytes(balance_raw, 'big') / 1e6
    print(f"  USDC: ${usdc_balance:.2f}")
    print()

    # Approve USDC for all three contracts
    print("1️⃣ USDC Approvals:")
    for i, spender in enumerate(SPENDERS, 1):
        approve_token(USDC_ADDRESS, "USDC", spender, f"Polymarket Contract {i}")
    print()

    # Approve Conditional Tokens for all three contracts (ERC-1155)
    print("2️⃣ Conditional Token Approvals:")
    for i, operator in enumerate(SPENDERS, 1):
        approve_erc1155(operator, f"Polymarket Contract {i}")
    print()

    print("✅ All allowances set! You can now trade on Polymarket.")
    print()
    print("🎯 Next steps:")
    print("  1. Restart the trading bot: python railway_worker.py")
    print("  2. Watch for successful LIVE orders in the logs")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
