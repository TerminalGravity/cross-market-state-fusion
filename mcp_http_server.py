#!/usr/bin/env python3
"""
MCP HTTP Server for Fly.io Deployment

Runs polymarket MCP servers with HTTP transport for remote access.
Designed for deployment on Fly.io with Canadian proxy for reliable Polymarket access.

Usage:
    python mcp_http_server.py --server trader --port 8001
    python mcp_http_server.py --server infra --port 8002
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description='Run MCP server with HTTP transport')
    parser.add_argument('--server', choices=['trader', 'infra'], required=True,
                        help='Which MCP server to run')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run on (default: 8000)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()

    if args.server == 'trader':
        print(f"Starting Polymarket Trader MCP on port {args.port}...")
        from polymarket_trader_mcp import mcp
    else:
        print(f"Starting Polymarket Infra MCP on port {args.port}...")
        from polymarket_infra_mcp import mcp

    # Run with HTTP transport
    mcp.run(
        transport='http',
        host=args.host,
        port=args.port
    )


if __name__ == '__main__':
    main()
