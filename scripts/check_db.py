#!/usr/bin/env python3
"""Check database state for P&L investigation."""
import asyncio
import os
import asyncpg

async def main():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])

    # Check sessions table schema
    schema = await conn.fetch("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'sessions'
        ORDER BY ordinal_position
    """)
    print("=== SESSIONS TABLE SCHEMA ===")
    for col in schema:
        print(f"  {col['column_name']}: {col['data_type']}")
    print()

    # Check all tables
    tables = await conn.fetch("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    print("=== ALL TABLES ===")
    for t in tables:
        print(f"  {t['table_name']}")
    print()

    # Get actual sessions data
    sessions = await conn.fetch("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 5")
    print("=== RECENT SESSIONS ===")
    if sessions:
        cols = list(sessions[0].keys())
        print(f"Columns: {cols}")
        for s in sessions:
            print(f"\n  Session ID: {s.get('id', 'N/A')}")
            for col in cols:
                if col != 'id':
                    val = s.get(col)
                    if isinstance(val, float):
                        print(f"    {col}: ${val:.2f}" if 'balance' in col or 'pnl' in col.lower() else f"    {col}: {val:.4f}")
                    else:
                        print(f"    {col}: {val}")
    else:
        print("  No sessions found")
    print()

    # Check trades table
    try:
        trades = await conn.fetch("SELECT * FROM trades ORDER BY created_at DESC LIMIT 10")
        print("=== RECENT TRADES ===")
        if trades:
            for t in trades:
                print(f"  Trade ID: {t.get('id')}, Asset: {t.get('asset', 'N/A')}")
                print(f"    Mode: {t.get('mode')}, Side: {t.get('side')}")
                print(f"    Entry: {t.get('entry_price')}, Exit: {t.get('exit_price')}")
                print(f"    P&L: ${t.get('pnl', 0):.2f if t.get('pnl') else 'N/A'}")
                print(f"    Created: {t.get('created_at')}")
                print()
        else:
            print("  No trades found")
    except Exception as e:
        print(f"  Error reading trades: {e}")

    # Check transfers table
    try:
        transfers = await conn.fetch("SELECT * FROM transfers ORDER BY created_at DESC LIMIT 5")
        print("=== PROFIT TRANSFERS ===")
        if transfers:
            for tr in transfers:
                print(f"  Transfer ID: {tr.get('id')}")
                for k, v in tr.items():
                    if k != 'id':
                        print(f"    {k}: {v}")
                print()
        else:
            print("  No transfers found")
    except Exception as e:
        print(f"  Error reading transfers: {e}")

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
