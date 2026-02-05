#!/usr/bin/env python3
"""
Buy Liverpool FC YES shares @ 17¢
FA Cup: Arsenal vs Liverpool
Entry at 17% = positive EV zone (+$0.10 EV per trade based on historical data)
"""
import os
import sys

# Load environment variables from .env file
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

# Liverpool FC YES token ID (from Gamma API)
LIVERPOOL_YES_TOKEN_ID = "9381486464766162423631294452121816997637773872017384184888288570356971622495"
CONDITION_ID = "0xc10d3901187198b4bf4fed0876027f9d04619dc44f15e89246a6fd80d87bd9db"

# Trade parameters
ENTRY_PRICE = 0.17  # 17 cents
AVAILABLE_USDC = 1.23
SHARES_TO_BUY = int(AVAILABLE_USDC / ENTRY_PRICE)  # ~7 shares

def main():
    """Buy Liverpool YES shares."""
    print("\n" + "=" * 60)
    print("BUYING LIVERPOOL FC @ 17¢")
    print("FA Cup: Arsenal vs Liverpool")
    print("=" * 60)
    print(f"Token ID: {LIVERPOOL_YES_TOKEN_ID[:20]}...")
    print(f"Entry Price: {ENTRY_PRICE:.0%}")
    print(f"Available USDC: ${AVAILABLE_USDC:.2f}")
    print(f"Shares to Buy: ~{SHARES_TO_BUY}")
    print(f"Total Cost: ~${SHARES_TO_BUY * ENTRY_PRICE:.2f}")
    print("=" * 60 + "\n")

    # Verify env vars
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder = os.getenv("POLYMARKET_FUNDER_ADDRESS")

    if not private_key or not funder:
        print("ERROR: POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS required")
        print("Set these in .env file")
        sys.exit(1)

    # Initialize LIVE executor
    executor = ClobExecutor(
        mode=ExecutionMode.LIVE,
        private_key=private_key,
        funder_address=funder,
        signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))
    )

    print("Placing BUY order for Liverpool YES...")

    try:
        # Place market buy order (FOK - Fill or Kill)
        order = executor.place_market_order(
            token_id=LIVERPOOL_YES_TOKEN_ID,
            amount=AVAILABLE_USDC,  # Dollar amount to spend
            side=OrderSide.BUY,
            asset="Liverpool FC YES"
        )

        if order:
            print(f"\n✅ Order Result:")
            print(f"   Order ID: {order.order_id}")
            print(f"   Status: {order.status}")
            print(f"   Filled: {order.filled_size} shares")

            if order.status == "matched":
                shares = AVAILABLE_USDC / ENTRY_PRICE
                potential_payout = shares * 1.0  # If Liverpool wins, payout is $1 per share
                print(f"\n🎯 TRADE EXECUTED!")
                print(f"   Bought ~{shares:.1f} Liverpool YES shares @ 17¢")
                print(f"   If Liverpool wins: ${potential_payout:.2f} payout")
                print(f"   Profit if win: ${potential_payout - AVAILABLE_USDC:.2f}")
            else:
                print(f"\n⚠️  Order status: {order.status}")
                print("   May need manual check")
        else:
            print("\n❌ Order placement failed")
            print("   Check logs for details")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
