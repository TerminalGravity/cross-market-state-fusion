#!/usr/bin/env python3
import asyncio, os, asyncpg
async def main():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    print("=== TABLES ===")
    for t in await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"):
        print(f"  {t[0]}")
    print("\n=== SESSIONS ===")
    for s in await conn.fetch("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 5"):
        print(dict(s))
    print("\n=== TRADES (closed, with P&L) ===")
    for t in await conn.fetch("SELECT id,asset,side,entry_price,exit_price,pnl,execution_type,created_at FROM trades WHERE pnl IS NOT NULL ORDER BY created_at DESC LIMIT 10"):
        print(dict(t))
    print("\n=== TRANSFERS ===")
    try:
        for tr in await conn.fetch("SELECT * FROM transfers ORDER BY created_at DESC LIMIT 5"):
            print(dict(tr))
    except: print("  No transfers table")
    await conn.close()
asyncio.run(main())
