#!/usr/bin/env python3
"""Check and optionally reset kill switch."""
import asyncio
import asyncpg
import os
import sys

async def main():
    db_url = os.environ.get('DATABASE_URL')
    conn = await asyncpg.connect(db_url)

    # Check kill switch state
    print("=== Kill Switch State ===")
    row = await conn.fetchrow("SELECT * FROM kill_switch_state ORDER BY id LIMIT 1")
    if row:
        print(f"Active: {row.get('is_active')}")
        print(f"Reason: {row.get('activation_reason')}")
        print(f"Activated: {row.get('activated_at')}")
        details = row.get('details')
        if details:
            print(f"Details: {details}")
    else:
        print("No kill switch records found")

    # Check loss tracker state (table may be named differently)
    print("\n=== Loss Tracker State ===")
    try:
        loss_row = await conn.fetchrow("SELECT * FROM loss_tracker_state ORDER BY id LIMIT 1")
        if loss_row:
            for k, v in loss_row.items():
                print(f"{k}: {v}")
    except Exception as e:
        print(f"Could not query loss_tracker: {e}")

    # If --reset flag passed, reset the kill switch
    if len(sys.argv) > 1 and sys.argv[1] == '--reset':
        print("\n=== Resetting Kill Switch ===")
        await conn.execute("UPDATE kill_switch_state SET is_active = false, deactivated_at = NOW(), deactivation_reason = 'manual_reset_via_script'")
        print("Kill switch deactivated!")

        # Also reset loss tracker to start fresh
        try:
            await conn.execute("""
                UPDATE loss_tracker_state
                SET consecutive_losses = 0,
                    updated_at = NOW()
            """)
            print("Consecutive losses reset to 0")
        except Exception as e:
            print(f"Could not reset loss tracker: {e}")

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
