#!/usr/bin/env python3
"""Apply database migration 002 without schema_migrations dependency"""
import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')

from db.connection import Database


async def main():
    db = Database()
    await db.connect()

    try:
        print("Checking if migration already applied...")

        # Check if already applied
        existing = await db.fetchrow("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'trades' AND column_name = 'force_closed'
        """)

        if existing:
            print("✓ Migration already applied (force_closed column exists)")
            await db.close()
            return

        print("\nApplying migration 002_add_position_health...\n")

        # Add columns to trades table
        print("1. Adding columns to trades table...")
        await db.execute("""
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS force_closed BOOLEAN DEFAULT FALSE
        """)
        await db.execute("""
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS force_close_reason VARCHAR(50)
        """)
        await db.execute("""
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS position_age_seconds INTEGER
        """)
        await db.execute("""
            ALTER TABLE trades ADD COLUMN IF NOT EXISTS market_expiry_time TIMESTAMPTZ
        """)
        print("   ✓ Columns added")

        # Create health_events table
        print("2. Creating health_events table...")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS health_events (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                event_type VARCHAR(20) NOT NULL CHECK (event_type IN ('degraded', 'critical', 'recovered')),
                asset VARCHAR(10) NOT NULL,
                staleness_seconds NUMERIC(6, 2),
                action_taken VARCHAR(50),
                open_positions INTEGER DEFAULT 0,
                affected_trades UUID[]
            )
        """)
        print("   ✓ Table created")

        # Create indexes
        print("3. Creating indexes...")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_health_events_asset ON health_events(asset)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_health_events_type ON health_events(event_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_health_events_time ON health_events(created_at DESC)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_trades_open_by_age ON trades(entry_time) WHERE exit_time IS NULL")
        print("   ✓ Indexes created")

        # Create view
        print("4. Creating position_health_summary view...")
        await db.execute("""
            CREATE OR REPLACE VIEW position_health_summary AS
            SELECT
                session_id,
                asset,
                COUNT(*) FILTER (WHERE force_closed = TRUE) as force_closed_count,
                COUNT(*) FILTER (WHERE force_close_reason = 'timeout') as timeout_exits,
                COUNT(*) FILTER (WHERE force_close_reason = 'health_check') as health_exits,
                COUNT(*) FILTER (WHERE force_close_reason = 'max_duration') as duration_exits,
                AVG(position_age_seconds) FILTER (WHERE force_closed = TRUE) as avg_force_close_age,
                COUNT(*) as total_trades
            FROM trades
            WHERE exit_time IS NOT NULL
            GROUP BY session_id, asset
        """)
        print("   ✓ View created")

        print("\n✅ Migration 002 applied successfully!\n")

        # Verify
        columns = await db.fetch("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'trades'
            AND column_name IN ('force_closed', 'force_close_reason', 'position_age_seconds', 'market_expiry_time')
            ORDER BY column_name
        """)

        print(f"Verified {len(columns)} new columns in trades table:")
        for col in columns:
            print(f"  ✓ {col['column_name']}")

        # Check health_events table
        health_count = await db.fetchval("SELECT COUNT(*) FROM health_events")
        print(f"\n✓ health_events table ready ({health_count} events)\n")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
