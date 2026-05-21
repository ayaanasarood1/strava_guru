#!/usr/bin/env python3
"""
Smart Race Extraction
More conservative approach - only likely races
"""

import json
import sqlite3
from pathlib import Path


def extract_likely_races():
    """Extract only highly likely race efforts"""

    cache_db = Path.home() / ".strava_guru_cache" / "activities.db"

    print("=" * 60)
    print("Smart Race Extraction")
    print("=" * 60)

    conn = sqlite3.connect(cache_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Very conservative criteria for races:
    # 1. Exact race distances (within 0.1 miles)
    # 2. Hard effort (HR > 150, pace < 8.0)
    # 3. Not too long (exclude ultras)
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
            WHEN ABS(distance_meters * 0.000621371 - 3.1) < 0.15 THEN '5K'
            WHEN ABS(distance_meters * 0.000621371 - 6.2) < 0.2 THEN '10K'
            WHEN ABS(distance_meters * 0.000621371 - 13.1) < 0.3 THEN 'Half'
            WHEN ABS(distance_meters * 0.000621371 - 26.2) < 0.5 THEN 'Marathon'
            ELSE NULL
        END as race_type
    FROM activities
    WHERE distance_meters > 4800
        AND avg_pace > 0
        AND avg_pace < 8.0
        AND avg_heart_rate > 150
        AND distance_meters < 45000
    """

    cursor.execute(query)
    results = cursor.fetchall()

    # Group by race type
    races_by_type = {'5K': [], '10K': [], 'Half': [], 'Marathon': []}

    for row in results:
        race_type = row['race_type']
        if race_type:
            races_by_type[race_type].append(row)

    conn.close()

    # Print summary with review
    print("\nFound likely races (conservative criteria):\n")

    all_races = []
    for race_type in ['Marathon', 'Half', '10K', '5K']:
        races = races_by_type[race_type]
        if races:
            print(f"\n{race_type} ({len(races)} found):")
            print("-" * 60)

            for i, row in enumerate(races, 1):
                time_str = format_time(row['minutes'])
                date = row['activity_date'][:10]

                print(f"  {i:2d}. {date}  {time_str:>10}  "
                      f"(Pace: {row['avg_pace']:.2f}, HR: {row['avg_heart_rate']})")

                # Get exact distance
                if race_type == '5K':
                    distance = 3.1
                elif race_type == '10K':
                    distance = 6.2
                elif race_type == 'Half':
                    distance = 13.1
                else:
                    distance = 26.2

                race = {
                    "race_id": f"{race_type.lower().replace(' ', '_')}_{date.replace('-', '')}",
                    "runner_id": "my_runner",
                    "race_date": date,
                    "race_distance_miles": distance,
                    "actual_time_minutes": round(row['minutes'], 1),
                    "age": 35,  # TODO: Update
                    "sex": "M",  # TODO: Update
                    "max_hr": 185,  # TODO: Update
                    "experience_years": 8,  # TODO: Update
                    "resting_hr": 52,  # Optional
                    "lookback_weeks": 12 if distance >= 13.1 else 8,
                    "_race_type": race_type,
                    "_file_name": row['file_name'],
                    "_avg_pace": round(row['avg_pace'], 2),
                    "_avg_hr": row['avg_heart_rate'],
                    "_max_hr": row['max_heart_rate']
                }
                all_races.append(race)

    # Save
    output_file = "likely_races.json"
    with open(output_file, 'w') as f:
        json.dump(all_races, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(all_races)} likely races to {output_file}")
    print(f"{'='*60}")

    print("\n📋 Review Steps:")
    print("  1. Open likely_races.json")
    print("  2. Review each race - remove any that are NOT actual races")
    print("  3. Update your runner profile (age, sex, max_hr)")
    print("  4. If you have race flags from Strava, use those instead")

    print("\n💡 To check if Strava marked these as races:")
    print("  - Look at the activity in Strava")
    print("  - Check if it has a 'Race' flag or is marked as a race")
    print("  - Check the activity title/description for race names")

    print(f"\nNext: python train_race_model.py {output_file}")

    return all_races


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
    races = extract_likely_races()
