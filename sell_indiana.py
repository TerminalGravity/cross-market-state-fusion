#!/usr/bin/env python3
"""
Quick sell script for Indiana CFP position.
Frees up capital for HFT automated trading.
"""
import os
import sys

# Load environment variables from .env file manually
def load_env():
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))

load_env()

from helpers.clob_executor import ClobExecutor, ExecutionMode, OrderSide

# Indiana CFP YES token ID (from get_open_positions)
INDIANA_TOKEN_ID = "96183293358900405327419356712562758572876725700872230341345248053405060086825"
INDIANA_SHARES = 5.81
CURRENT_PRICE = 0.421

def main():
    """Sell Indiana position to unlock capital for HFT."""
    print("\n" + "="*60)
    print("SELLING INDIANA CFP POSITION")
    print("="*60)
    print(f"Shares: {INDIANA_SHARES}")
    print(f"Current Price: ${CURRENT_PRICE}")
    print(f"Expected Return: ~${INDIANA_SHARES * CURRENT_PRICE:.2f}")
    print("="*60 + "\n")

    # Verify env vars
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder = os.getenv("POLYMARKET_FUNDER_ADDRESS")

    if not private_key or not funder:
        print("ERROR: POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS required")
        sys.exit(1)

    # Initialize LIVE executor
    executor = ClobExecutor(
        mode=ExecutionMode.LIVE,
        private_key=private_key,
        funder_address=funder,
        signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))
    )

    print("Placing SELL order...")

    # Use sell with retry for better fill rate
    order = executor.place_sell_with_retry(
        token_id=INDIANA_TOKEN_ID,
        shares=INDIANA_SHARES,
        current_price=CURRENT_PRICE,
        asset="Indiana CFP YES",
        max_retries=3
    )

    if order:
        print(f"\nOrder Result:")
        print(f"  Order ID: {order.order_id}")
        print(f"  Status: {order.status}")
        print(f"  Filled: {order.filled_size} shares")

        if order.status == "matched":
            print("\n✅ SELL SUCCESSFUL - Capital unlocked!")
            print(f"   Expected ~${INDIANA_SHARES * CURRENT_PRICE:.2f} USDC added")
        else:
            print(f"\n⚠️  Order status: {order.status}")
            print("   May need manual check")
    else:
        print("\n❌ SELL FAILED")
        print("   Check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
