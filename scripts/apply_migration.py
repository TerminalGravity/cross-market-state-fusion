#!/usr/bin/env python3
"""Apply database migration 002_add_position_health.sql"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')

from db.connection import Database


async def main():
    # Read migration file
    migration_file = Path('db/migrations/002_add_position_health.sql')
    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        sys.exit(1)

    migration_sql = migration_file.read_text()

    # Connect to database
    db = Database()
    await db.connect()

    try:
        print("Applying migration 002_add_position_health.sql...")

        # Check if already applied
        existing = await db.fetchrow("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'trades' AND column_name = 'force_closed'
        """)

        if existing:
            print("✓ Migration already applied (force_closed column exists)")
            return

        # Apply migration
        await db.execute(migration_sql)
        print("✅ Migration 002 applied successfully!")

        # Verify
        columns = await db.fetch("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'trades'
            AND column_name IN ('force_closed', 'force_close_reason', 'position_age_seconds', 'market_expiry_time')
            ORDER BY column_name
        """)

        print(f"\nVerified {len(columns)} new columns in trades table:")
        for col in columns:
            print(f"  - {col['column_name']}")

        # Check health_events table
        health_count = await db.fetchval("SELECT COUNT(*) FROM health_events")
        print(f"\n✓ health_events table created ({health_count} events)")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
