#!/usr/bin/env python3
"""
Debug why feature extraction missed Salman's training data
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from activity_cache import ActivityCache

def main():
    print("="*80)
    print("Debugging Feature Extraction for Salman's Jack & Jill 2025")
    print("="*80)

    # Race details
    race_date = datetime(2025, 7, 27)
    lookback_weeks = 12

    # Training window (same logic as feature extraction)
    taper_start = race_date - timedelta(days=7)
    training_start = taper_start - timedelta(weeks=lookback_weeks)

    print(f"\nRace date: {race_date.date()}")
    print(f"Training window: {training_start.date()} to {taper_start.date()}")
    print(f"(Race date - 7 days) - 12 weeks")

    # Load Salman's cache
    cache_dir = Path.home() / ".strava_guru_cache" / "salman"
    cache = ActivityCache(cache_dir=cache_dir)

    print(f"\nCache location: {cache_dir}")
    print(f"Database: {cache.db_path}")

    # Check total activities in cache
    conn = sqlite3.connect(cache.db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM activities")
    total_activities = cursor.fetchone()[0]
    print(f"Total activities in cache: {total_activities}")

    # Check activities in date range
    print(f"\n{'='*80}")
    print("Step 1: Query activities in training window")
    print(f"{'='*80}")

    cursor.execute("""
        SELECT COUNT(*)
        FROM activities
        WHERE activity_date >= ? AND activity_date < ?
    """, (training_start.isoformat(), taper_start.isoformat()))

    count_in_window = cursor.fetchone()[0]
    print(f"Activities in window: {count_in_window}")

    # Check running activities specifically
    cursor.execute("""
        SELECT COUNT(*)
        FROM activities
        WHERE activity_date >= ?
          AND activity_date < ?
          AND activity_type = 'running'
    """, (training_start.isoformat(), taper_start.isoformat()))

    running_count = cursor.fetchone()[0]
    print(f"Running activities in window: {running_count}")

    # Check date range of all activities
    print(f"\n{'='*80}")
    print("Step 2: Check date range of ALL activities in cache")
    print(f"{'='*80}")

    cursor.execute("""
        SELECT MIN(activity_date), MAX(activity_date)
        FROM activities
    """)
    min_date, max_date = cursor.fetchone()
    print(f"Earliest activity: {min_date}")
    print(f"Latest activity: {max_date}")

    # Check if target date range is covered
    if min_date and max_date:
        min_dt = datetime.fromisoformat(min_date)
        max_dt = datetime.fromisoformat(max_date)

        if min_dt > training_start:
            print(f"\n⚠️ PROBLEM: Cache starts AFTER training window!")
            print(f"   Cache starts: {min_date}")
            print(f"   Need data from: {training_start.isoformat()}")
        elif max_dt < taper_start:
            print(f"\n⚠️ PROBLEM: Cache ends BEFORE training window!")
            print(f"   Cache ends: {max_date}")
            print(f"   Need data until: {taper_start.isoformat()}")
        else:
            print(f"\n✓ Cache date range covers training window")

    # Show sample activities in the window
    print(f"\n{'='*80}")
    print("Step 3: Sample activities in training window")
    print(f"{'='*80}")

    cursor.execute("""
        SELECT activity_date, activity_type, distance_meters, duration_seconds
        FROM activities
        WHERE activity_date >= ?
          AND activity_date < ?
        ORDER BY activity_date
        LIMIT 10
    """, (training_start.isoformat(), taper_start.isoformat()))

    rows = cursor.fetchall()
    if rows:
        print(f"First 10 activities in window:")
        for row in rows:
            date, atype, dist_m, dur_s = row
            dist_mi = dist_m / 1609.34 if dist_m else 0
            print(f"  {date[:10]}: {atype:10s} {dist_mi:6.2f} mi")
    else:
        print("NO ACTIVITIES FOUND IN WINDOW!")

    # Check activities around the expected dates
    print(f"\n{'='*80}")
    print("Step 4: Check activities around expected dates")
    print(f"{'='*80}")

    # Check May 2025 (should have lots of activity)
    cursor.execute("""
        SELECT COUNT(*)
        FROM activities
        WHERE activity_date >= '2025-05-01'
          AND activity_date < '2025-06-01'
    """)
    may_count = cursor.fetchone()[0]
    print(f"May 2025 activities: {may_count}")

    # Check June 2025
    cursor.execute("""
        SELECT COUNT(*)
        FROM activities
        WHERE activity_date >= '2025-06-01'
          AND activity_date < '2025-07-01'
    """)
    june_count = cursor.fetchone()[0]
    print(f"June 2025 activities: {june_count}")

    # Check July 2025
    cursor.execute("""
        SELECT COUNT(*)
        FROM activities
        WHERE activity_date >= '2025-07-01'
          AND activity_date < '2025-08-01'
    """)
    july_count = cursor.fetchone()[0]
    print(f"July 2025 activities: {july_count}")

    # Test the get_activities_by_date_range method directly
    print(f"\n{'='*80}")
    print("Step 5: Test ActivityCache.get_activities_by_date_range()")
    print(f"{'='*80}")

    activities = cache.get_activities_by_date_range(training_start, taper_start)
    print(f"Activities returned by get_activities_by_date_range(): {len(activities)}")

    if activities:
        print(f"\nFirst 5 activities:")
        for i, act in enumerate(activities[:5], 1):
            print(f"  {i}. {act.get('activity_date')}: {act.get('distance_meters', 0)/1609.34:.2f} mi")
    else:
        print("\n⚠️ NO ACTIVITIES RETURNED!")
        print("\nThis is the ROOT CAUSE - the cache query returns nothing.")

    # Show the actual SQL query being used
    print(f"\n{'='*80}")
    print("Step 6: Check SQL query in ActivityCache")
    print(f"{'='*80}")

    # Manually run the query that get_activities_by_date_range should use
    cursor.execute("""
        SELECT *
        FROM activities
        WHERE activity_date >= ?
          AND activity_date <= ?
        ORDER BY activity_date
    """, (training_start.isoformat(), taper_start.isoformat()))

    manual_results = cursor.fetchall()
    print(f"Manual query with >= and <=: {len(manual_results)} rows")

    # Try with < instead of <=
    cursor.execute("""
        SELECT *
        FROM activities
        WHERE activity_date >= ?
          AND activity_date < ?
        ORDER BY activity_date
    """, (training_start.isoformat(), taper_start.isoformat()))

    manual_results2 = cursor.fetchall()
    print(f"Manual query with >= and <: {len(manual_results2)} rows")

    conn.close()

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Expected from CSV: 88 runs")
    print(f"Found in cache: {running_count} running activities")
    print(f"Returned by get_activities_by_date_range(): {len(activities)}")

    if running_count == 0:
        print(f"\n❌ ROOT CAUSE: No activities cached in training window")
        print("   Solution: Re-build cache or check date parsing during cache build")
    elif len(activities) == 0 and running_count > 0:
        print(f"\n❌ ROOT CAUSE: Cache has data but query doesn't return it")
        print("   Solution: Fix get_activities_by_date_range() query")
    elif len(activities) != 88:
        print(f"\n⚠️ PARTIAL DATA: Cache returns {len(activities)} but CSV has 88")
        print("   Solution: Investigate which activities are missing")
    else:
        print(f"\n✓ Cache data looks correct - issue must be in feature extraction")

if __name__ == '__main__':
    main()
