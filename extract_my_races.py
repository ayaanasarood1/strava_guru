#!/usr/bin/env python3
"""
Extract Real Race Results from Activity Cache
Finds likely race efforts and creates training dataset
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime


def extract_races_from_cache():
    """Extract race-like efforts from activity cache"""

    cache_db = Path.home() / ".strava_guru_cache" / "activities.db"

    print("=" * 60)
    print("Extracting Race Results from Activity Cache")
    print("=" * 60)

    conn = sqlite3.connect(cache_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find activities at common race distances with hard efforts
    query = """
    SELECT
        file_name,
        activity_date,
        distance_meters,
        duration_seconds,
        avg_pace,
        avg_heart_rate,
        max_heart_rate,
        ROUND(distance_meters * 0.000621371, 2) as miles,
        ROUND(duration_seconds / 60.0, 2) as minutes,
        CASE
            WHEN ABS(distance_meters * 0.000621371 - 3.1) < 0.3 THEN '5K'
            WHEN ABS(distance_meters * 0.000621371 - 6.2) < 0.3 THEN '10K'
            WHEN ABS(distance_meters * 0.000621371 - 13.1) < 0.5 THEN 'Half'
            WHEN ABS(distance_meters * 0.000621371 - 26.2) < 1.0 THEN 'Marathon'
            ELSE 'Other'
        END as race_type
    FROM activities
    WHERE distance_meters > 4800
        AND avg_pace > 0
        AND avg_pace < 9.0
        AND avg_heart_rate > 140
    ORDER BY activity_date DESC
    """

    cursor.execute(query)
    results = cursor.fetchall()

    # Filter to only common race distances
    races = []
    race_types_found = {'5K': 0, '10K': 0, 'Half': 0, 'Marathon': 0}

    for row in results:
        race_type = row['race_type']

        if race_type == 'Other':
            continue

        race_types_found[race_type] += 1

        # Get exact distance for race type
        if race_type == '5K':
            distance = 3.1
        elif race_type == '10K':
            distance = 6.2
        elif race_type == 'Half':
            distance = 13.1
        else:  # Marathon
            distance = 26.2

        race = {
            "race_id": f"{race_type.lower()}_{row['activity_date'][:10].replace('-', '')}",
            "runner_id": "my_runner",
            "race_date": row['activity_date'][:10],
            "race_distance_miles": distance,
            "actual_time_minutes": round(row['minutes'], 1),

            # Runner profile (UPDATE THESE WITH YOUR INFO)
            "age": 35,  # TODO: Update with your age at race time
            "sex": "M",  # TODO: Update with your sex (M/F)
            "max_hr": 185,  # TODO: Update with your max HR
            "experience_years": 8,  # TODO: Update with years of running
            "resting_hr": 52,  # TODO: Update with your resting HR (optional)

            # Training lookback
            "lookback_weeks": 12 if distance >= 13.1 else 8,

            # Metadata for reference
            "_race_type": race_type,
            "_file_name": row['file_name'],
            "_avg_pace": round(row['avg_pace'], 2),
            "_avg_hr": row['avg_heart_rate'],
            "_max_hr": row['max_heart_rate']
        }

        races.append(race)

    conn.close()

    # Print summary
    print(f"\nFound {len(races)} race efforts:")
    for race_type, count in race_types_found.items():
        if count > 0:
            print(f"  {race_type}: {count} races")

    # Show sample
    print(f"\n{'='*60}")
    print("Sample Races (most recent):")
    print(f"{'='*60}")

    for i, race in enumerate(races[:10], 1):
        time_str = format_time(race['actual_time_minutes'])
        print(f"{i:2d}. {race['race_date']}  {race['_race_type']:<10}  {time_str}  "
              f"(Pace: {race['_avg_pace']:.2f}, HR: {race['_avg_hr']})")

    if len(races) > 10:
        print(f"     ... and {len(races) - 10} more races")

    # Save to file
    output_file = "my_race_results.json"

    with open(output_file, 'w') as f:
        json.dump(races, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(races)} races to {output_file}")
    print(f"{'='*60}")

    print("\n⚠️  IMPORTANT: Review and update the file:")
    print("  1. Check that these are actual races (not hard workouts)")
    print("  2. Update runner profile (age, sex, max_hr, etc.)")
    print("  3. Remove any non-race efforts")
    print("  4. Adjust ages if they changed over time")

    print(f"\nNext step: python train_race_model.py {output_file}")

    return races


def format_time(minutes):
    """Format minutes as H:MM:SS or MM:SS"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes % 1) * 60)

    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    else:
        return f"{mins}:{secs:02d}"


if __name__ == '__main__':
    races = extract_races_from_cache()
