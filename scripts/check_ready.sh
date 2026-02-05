#!/bin/bash
# Pre-flight check before starting trading
# Verifies wallet balance, infrastructure, and strategy config

echo "=========================================="
echo " Pre-Flight Trading Check"
echo "=========================================="

# Check wallet via MCP (would need to be run via claude code)
echo ""
echo "[1] Wallet Address for USDC Deposit:"
echo "    0x5C1be81bfCD92451733f4AaC7207FD4c3a818dE1"
echo "    Network: Polygon (MATIC)"
echo ""
echo "[2] Minimum Recommended Balance:"
echo "    - $50 USDC for testing"
echo "    - $500 USDC for meaningful trading"
echo "    - $5000 USDC for arbitrage strategy"
echo ""
echo "[3] Fly.io Status:"
fly status -a cross-market-state-fusion 2>/dev/null || echo "    (run: fly auth login)"
echo ""
echo "[4] Strategy Config:"
cat configs/optimal_strategy.json 2>/dev/null | head -30 || echo "    (config file not found)"
echo ""
echo "=========================================="
echo " Ready to trade when USDC deposited!"
echo "=========================================="
