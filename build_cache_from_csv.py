#!/usr/bin/env python3
"""
Build cache from activities.csv (fallback for corrupted FIT files)
"""

import csv
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from activity_cache import ActivityCache

def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string like '1:23:45' to timedelta"""
    try:
        parts = duration_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return timedelta(minutes=minutes, seconds=seconds)
        else:
            return timedelta(seconds=int(duration_str))
    except:
        return timedelta(0)

def build_cache_from_csv(csv_path: Path, cache: ActivityCache, activities_dir: Path = None):
    """
    Build cache from activities.csv file

    This is a fallback for when FIT files are corrupted.
    CSV contains all the essential data: date, type, distance, duration, HR, etc.
    """
    print(f"Building cache from CSV: {csv_path}")
    print(f"Cache location: {cache.cache_dir}")

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    conn = sqlite3.connect(cache.db_path)
    cursor = conn.cursor()

    cached_count = 0
    skipped_count = 0
    error_count = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, 1):
            try:
                # Parse activity data
                activity_type = row.get('Activity Type', '')

                # Convert to our activity_type format
                if activity_type == 'Run':
                    activity_type = 'running'
                elif activity_type == 'Ride':
                    activity_type = 'cycling'
                elif activity_type == 'Walk':
                    activity_type = 'walking'
                else:
                    activity_type = activity_type.lower() if activity_type else 'unknown'

                # Parse date
                date_str = row.get('Activity Date', '')
                if not date_str:
                    skipped_count += 1
                    continue

                activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")

                # Get filename - strip 'activities/' prefix if present
                filename = row.get('Filename', '')
                if not filename:
                    filename = f"csv_import_{i}.fit"
                elif filename.startswith('activities/'):
                    filename = filename[11:]

                # Parse metrics
                distance_m = float(row.get('Distance', 0) or 0)

                # Duration - parse time string
                moving_time_str = row.get('Moving Time', '0')
                elapsed_time_str = row.get('Elapsed Time', moving_time_str)

                moving_time = parse_duration(moving_time_str)
                elapsed_time = parse_duration(elapsed_time_str)

                # Pace (min/mile)
                avg_pace = None
                if distance_m > 0 and moving_time.total_seconds() > 0:
                    distance_miles = distance_m / 1609.34
                    time_minutes = moving_time.total_seconds() / 60.0
                    avg_pace = time_minutes / distance_miles if distance_miles > 0 else None

                # Heart rate
                avg_hr = None
                max_hr = None
                avg_hr_str = row.get('Average Heart Rate', '')
                max_hr_str = row.get('Max Heart Rate', '')

                if avg_hr_str:
                    try:
                        avg_hr = int(float(avg_hr_str))
                    except:
                        pass

                if max_hr_str:
                    try:
                        max_hr = int(float(max_hr_str))
                    except:
                        pass

                # Elevation
                elev_gain = float(row.get('Elevation Gain', 0) or 0)
                elev_loss = float(row.get('Elevation Loss', 0) or 0)

                # Calories
                calories = None
                calories_str = row.get('Calories', '')
                if calories_str:
                    try:
                        calories = int(float(calories_str))
                    except:
                        pass

                # Insert into database
                cursor.execute("""
                    INSERT OR REPLACE INTO activities (
                        file_name, file_hash, activity_date, activity_type,
                        distance_meters, duration_seconds, moving_time_seconds,
                        avg_pace, avg_gap, elevation_gain_meters, elevation_loss_meters,
                        avg_heart_rate, max_heart_rate, calories,
                        processed_at, track_points_file, laps_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    filename,
                    'csv_import',  # file_hash
                    activity_date.isoformat(),
                    activity_type,
                    distance_m,
                    elapsed_time.total_seconds(),
                    moving_time.total_seconds(),
                    avg_pace,
                    None,  # avg_gap
                    elev_gain,
                    elev_loss,
                    avg_hr,
                    max_hr,
                    calories,
                    datetime.now().isoformat(),
                    None,  # track_points_file
                    None   # laps_json
                ))

                cached_count += 1

                if i % 100 == 0:
                    print(f"Progress: {i} rows processed, {cached_count} cached")
                    conn.commit()

            except Exception as e:
                error_count += 1
                continue

    conn.commit()
    conn.close()

    print(f"\nCSV import complete!")
    print(f"  Cached: {cached_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")

    # Show cache stats
    stats = cache.get_stats()
    print(f"\nCache stats:")
    print(f"  Total activities: {stats['total_activities']}")
    print(f"  With HR data: {stats['with_hr']}")
    print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")

def main():
    print("="*80)
    print("Build Cache from CSV (Fallback for Corrupted FIT Files)")
    print("="*80)
    print()

    # Paths
    csv_path = Path('/Users/osman/Downloads/export_1884062_salman/activities.csv')
    cache_dir = Path.home() / ".strava_guru_cache" / "salman"
    activities_dir = Path('/Users/osman/Downloads/export_1884062_salman/activities')

    # Create cache
    cache = ActivityCache(cache_dir=cache_dir)

    # Clear existing cache
    print("Clearing existing cache to rebuild from CSV...")
    cache.db_path.unlink(missing_ok=True)
    cache._init_database()
    print()

    # Build from CSV
    build_cache_from_csv(csv_path, cache, activities_dir)

    print("\n" + "="*80)
    print("Cache successfully rebuilt from CSV!")
    print("="*80)

if __name__ == '__main__':
    main()
