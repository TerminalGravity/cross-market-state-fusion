"""
Polymarket CLOB order execution via py-clob-client.

This module handles live order placement, cancellation, and position tracking.
Requires: pip install py-clob-client
"""
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List
from enum import Enum

# py-clob-client imports
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    HAS_CLOB_CLIENT = True
except ImportError:
    HAS_CLOB_CLIENT = False
    print("WARNING: py-clob-client not installed. Run: pip install py-clob-client")


class ExecutionMode(Enum):
    PAPER = "paper"  # Simulate orders (no real execution)
    LIVE = "live"    # Execute real orders on Polymarket


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class LiveOrder:
    """Represents a live or pending order."""
    order_id: str
    token_id: str
    side: OrderSide
    price: float
    size: float
    order_type: str  # GTC, FOK, GTD
    status: str  # PENDING, OPEN, FILLED, CANCELLED, REJECTED
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_type: str = "paper"  # paper or live
    clob_response: Optional[Dict] = None  # Full API response for audit


@dataclass
class LivePosition:
    """Represents a live position."""
    token_id: str
    side: str  # "UP" or "DOWN"
    asset: str  # BTC, ETH, SOL, XRP
    size: float  # Number of shares
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ClobExecutor:
    """
    Handles order execution on Polymarket CLOB.

    Supports both paper trading (simulation) and live trading modes.
    """

    # Polymarket CLOB endpoint
    HOST = "https://clob.polymarket.com"
    CHAIN_ID = 137  # Polygon mainnet

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        private_key: Optional[str] = None,
        funder_address: Optional[str] = None,
        signature_type: int = 0,  # 0=EOA, 1=Email/Magic, 2=Browser proxy
    ):
        """
        Initialize the CLOB executor.

        Args:
            mode: PAPER for simulation, LIVE for real orders
            private_key: Wallet private key (required for LIVE mode)
            funder_address: Address holding funds (required for LIVE mode)
            signature_type: 0=EOA/MetaMask, 1=Email/Magic wallet, 2=Browser proxy
        """
        self.mode = mode
        self.client: Optional[ClobClient] = None
        self.orders: Dict[str, LiveOrder] = {}
        self.positions: Dict[str, LivePosition] = {}

        # Stats
        self.total_orders = 0
        self.filled_orders = 0
        self.total_volume = 0.0

        if mode == ExecutionMode.LIVE:
            if not HAS_CLOB_CLIENT:
                raise RuntimeError("py-clob-client required for LIVE mode. Run: pip install py-clob-client")

            if not private_key:
                private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
            if not funder_address:
                funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS")

            if not private_key or not funder_address:
                raise ValueError(
                    "LIVE mode requires private_key and funder_address. "
                    "Set POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS env vars."
                )

            self._init_client(private_key, funder_address, signature_type)

    def _init_client(self, private_key: str, funder_address: str, signature_type: int):
        """Initialize the py-clob-client."""
        self.client = ClobClient(
            host=self.HOST,
            key=private_key,
            chain_id=self.CHAIN_ID,
            signature_type=signature_type,
            funder=funder_address
        )
        # Derive API credentials (only need to do once, cached by library)
        self.client.set_api_creds(self.client.create_or_derive_api_creds())
        print(f"[CLOB] Initialized client for {funder_address[:10]}...{funder_address[-6:]}")

    def place_market_order(
        self,
        token_id: str,
        amount: float,
        side: OrderSide,
        asset: str = "UNKNOWN"
    ) -> Optional[LiveOrder]:
        """
        Place a market order (Fill-or-Kill).

        Args:
            token_id: Polymarket token ID (UP or DOWN token)
            amount: Dollar amount to spend (BUY) or shares to sell (SELL)
            side: BUY or SELL
            asset: Asset name for logging (BTC, ETH, etc.)

        Returns:
            LiveOrder if successful, None if failed
        """
        self.total_orders += 1

        if self.mode == ExecutionMode.PAPER:
            return self._simulate_market_order(token_id, amount, side, asset)

        # Live execution
        try:
            order_side = BUY if side == OrderSide.BUY else SELL

            market_order = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=order_side,
                order_type=OrderType.FOK
            )

            signed_order = self.client.create_market_order(market_order)
            response = self.client.post_order(signed_order, OrderType.FOK)

            order = LiveOrder(
                order_id=response.get("orderID", f"live_failed_{self.total_orders}"),
                token_id=token_id,
                side=side,
                price=0.0,  # Market order, filled at market
                size=amount,
                order_type="FOK",
                status=response.get("status", "UNKNOWN"),
                filled_size=amount if response.get("status") == "matched" else 0.0,
                execution_type="live",
                clob_response=response  # Store full response for audit
            )

            if order.status == "matched":
                self.filled_orders += 1
                self.total_volume += amount
                self._update_position(token_id, side, amount, 0.0, asset)

            self.orders[order.order_id] = order
            print(f"[CLOB] LIVE Market {side.value} ${amount:.2f} on {asset}: {order.status} (order_id={order.order_id})")
            return order

        except Exception as e:
            import traceback
            print(f"[CLOB] LIVE Order failed: {e}")
            print(f"[CLOB] Full error traceback:\n{traceback.format_exc()}")

            # Try to extract error details from response
            error_details = str(e)
            if hasattr(e, 'response'):
                try:
                    error_details = f"{e} | Response: {e.response.json()}"
                except:
                    error_details = f"{e} | Response text: {getattr(e.response, 'text', 'N/A')}"

            print(f"[CLOB] Error details: {error_details}")
            return None

    def place_limit_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: OrderSide,
        order_type: str = "GTC",
        asset: str = "UNKNOWN"
    ) -> Optional[LiveOrder]:
        """
        Place a limit order.

        Args:
            token_id: Polymarket token ID
            price: Limit price (0.01 to 0.99)
            size: Number of shares
            side: BUY or SELL
            order_type: GTC (Good-Til-Cancelled) or GTD (Good-Til-Date)
            asset: Asset name for logging

        Returns:
            LiveOrder if successful, None if failed
        """
        self.total_orders += 1

        if self.mode == ExecutionMode.PAPER:
            return self._simulate_limit_order(token_id, price, size, side, order_type, asset)

        # Live execution
        try:
            order_side = BUY if side == OrderSide.BUY else SELL
            clob_order_type = OrderType.GTC if order_type == "GTC" else OrderType.GTD

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=order_side
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, clob_order_type)

            order = LiveOrder(
                order_id=response.get("orderID", f"paper_{self.total_orders}"),
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                order_type=order_type,
                status=response.get("status", "UNKNOWN")
            )

            self.orders[order.order_id] = order
            print(f"[CLOB] Limit {side.value} {size:.2f} @ {price:.2f} on {asset}: {order.status}")
            return order

        except Exception as e:
            import traceback
            print(f"[CLOB] Limit order failed: {e}")
            print(f"[CLOB] Full error traceback:\n{traceback.format_exc()}")

            # Try to extract error details from response
            error_details = str(e)
            if hasattr(e, 'response'):
                try:
                    error_details = f"{e} | Response: {e.response.json()}"
                except:
                    error_details = f"{e} | Response text: {getattr(e.response, 'text', 'N/A')}"

            print(f"[CLOB] Error details: {error_details}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.mode == ExecutionMode.PAPER:
            if order_id in self.orders:
                self.orders[order_id].status = "CANCELLED"
                return True
            return False

        try:
            self.client.cancel(order_id)
            if order_id in self.orders:
                self.orders[order_id].status = "CANCELLED"
            print(f"[CLOB] Cancelled order {order_id}")
            return True
        except Exception as e:
            print(f"[CLOB] Cancel failed: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        if self.mode == ExecutionMode.PAPER:
            count = 0
            for order in self.orders.values():
                if order.status == "OPEN":
                    order.status = "CANCELLED"
                    count += 1
            return count

        try:
            self.client.cancel_all()
            count = 0
            for order in self.orders.values():
                if order.status == "OPEN":
                    order.status = "CANCELLED"
                    count += 1
            print(f"[CLOB] Cancelled {count} orders")
            return count
        except Exception as e:
            print(f"[CLOB] Cancel all failed: {e}")
            return 0

    def get_open_orders(self) -> List[LiveOrder]:
        """Get all open orders."""
        return [o for o in self.orders.values() if o.status == "OPEN"]

    def get_position(self, token_id: str) -> Optional[LivePosition]:
        """Get position for a token."""
        return self.positions.get(token_id)

    def get_all_positions(self) -> List[LivePosition]:
        """Get all open positions."""
        return list(self.positions.values())

    def _simulate_market_order(
        self, token_id: str, amount: float, side: OrderSide, asset: str
    ) -> LiveOrder:
        """Simulate a market order for paper trading."""
        order = LiveOrder(
            order_id=f"paper_{self.total_orders}",
            token_id=token_id,
            side=side,
            price=0.0,
            size=amount,
            order_type="FOK",
            status="matched",  # Paper orders always fill
            filled_size=amount,
            execution_type="paper",
            clob_response=None  # No real API response for paper trades
        )

        self.filled_orders += 1
        self.total_volume += amount
        self.orders[order.order_id] = order

        print(f"[PAPER] Market {side.value} ${amount:.2f} on {asset}")
        return order

    def _simulate_limit_order(
        self, token_id: str, price: float, size: float,
        side: OrderSide, order_type: str, asset: str
    ) -> LiveOrder:
        """Simulate a limit order for paper trading."""
        order = LiveOrder(
            order_id=f"paper_{self.total_orders}",
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_type=order_type,
            status="OPEN",  # Limit orders start as open
            execution_type="paper",
            clob_response=None  # No real API response for paper trades
        )

        self.orders[order.order_id] = order
        print(f"[PAPER] Limit {side.value} {size:.2f} @ {price:.2f} on {asset}")
        return order

    def _update_position(
        self, token_id: str, side: OrderSide, size: float,
        fill_price: float, asset: str
    ):
        """Update position after a fill."""
        if token_id in self.positions:
            pos = self.positions[token_id]
            if side == OrderSide.BUY:
                # Add to position
                total_cost = pos.avg_entry_price * pos.size + fill_price * size
                pos.size += size
                pos.avg_entry_price = total_cost / pos.size if pos.size > 0 else 0
            else:
                # Reduce position
                pos.size -= size
                if pos.size <= 0:
                    del self.positions[token_id]
        else:
            # New position
            self.positions[token_id] = LivePosition(
                token_id=token_id,
                side="UP" if "UP" in asset else "DOWN",
                asset=asset,
                size=size,
                avg_entry_price=fill_price
            )


# Convenience function
def create_executor(live: bool = False) -> ClobExecutor:
    """
    Create a CLOB executor.

    Args:
        live: If True, use LIVE mode (requires env vars). Otherwise PAPER mode.

    Environment variables for LIVE mode:
        POLYMARKET_PRIVATE_KEY: Your wallet's private key
        POLYMARKET_FUNDER_ADDRESS: Address that holds your funds
        POLYMARKET_SIGNATURE_TYPE: 0=EOA, 1=Email/Magic, 2=Browser (default: 0)
    """
    if live:
        sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))
        return ClobExecutor(
            mode=ExecutionMode.LIVE,
            signature_type=sig_type
        )
    return ClobExecutor(mode=ExecutionMode.PAPER)


if __name__ == "__main__":
    # Test paper trading
    print("Testing CLOB Executor (PAPER mode)...")

    executor = create_executor(live=False)

    # Simulate some orders
    token_id = "test_token_123"

    order1 = executor.place_market_order(token_id, 50.0, OrderSide.BUY, "BTC")
    print(f"  Order 1: {order1}")

    order2 = executor.place_limit_order(token_id, 0.45, 100.0, OrderSide.BUY, "GTC", "BTC")
    print(f"  Order 2: {order2}")

    print(f"\nStats: {executor.total_orders} orders, {executor.filled_orders} filled, ${executor.total_volume:.2f} volume")
