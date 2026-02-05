#!/bin/bash
# Quick restart script for Fly.io trading worker
# Usage: ./scripts/restart_worker.sh [paper|live] [trade_size]

set -e

APP="cross-market-state-fusion"
MACHINE_ID="78172d2c0e0778"
MODE="${1:-paper}"
SIZE="${2:-5}"

echo "=========================================="
echo " Polymarket Trading Worker Restart"
echo "=========================================="
echo " App:    $APP"
echo " Mode:   $MODE"
echo " Size:   \$$SIZE per trade"
echo "=========================================="

# Update settings if live mode
if [ "$MODE" = "live" ]; then
    echo "[*] Setting TRADING_MODE=live..."
    fly secrets set TRADING_MODE=live -a $APP
    fly secrets set TRADE_SIZE=$SIZE -a $APP
else
    echo "[*] Setting TRADING_MODE=paper..."
    fly secrets set TRADING_MODE=paper -a $APP
    fly secrets set TRADE_SIZE=$SIZE -a $APP
fi

# Start the machine
echo "[*] Starting machine $MACHINE_ID..."
fly machines start $MACHINE_ID -a $APP

echo ""
echo "[+] Worker started! View logs:"
echo "    fly logs -a $APP"
echo ""
echo "[+] Check status:"
echo "    fly status -a $APP"
