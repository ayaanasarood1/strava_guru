#!/usr/bin/env python3
"""
Extract Actual Races from Strava CSV
Uses the Competition flag to identify true races
"""

import csv
import json
from datetime import datetime
from pathlib import Path


def extract_races_from_csv(csv_path):
    """Extract races marked as Competition in Strava CSV"""

    print("=" * 60)
    print("Extracting Races from Strava CSV")
    print("=" * 60)

    races = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Check if marked as competition/race
            if row.get('Competition', '').lower() in ['true', '1', 'yes']:

                # Get activity details
                try:
                    activity_date = row['Activity Date']
                    # Parse date (format: "Dec 8, 2024, 10:03:38 AM")
                    date_obj = datetime.strptime(activity_date, "%b %d, %Y, %I:%M:%S %p")
                    race_date = date_obj.strftime("%Y-%m-%d")

                    # Get distance in miles
                    distance_str = row.get('Distance.1', row.get('Distance', '0'))
                    if distance_str:
                        # Remove any units and convert to float
                        distance_km = float(distance_str.replace(',', ''))
                        distance_miles = distance_km * 0.621371
                    else:
                        continue

                    # Get time in minutes
                    elapsed = row.get('Elapsed Time.1', row.get('Elapsed Time', '0'))
                    if elapsed:
                        elapsed_secs = int(elapsed)
                        time_minutes = elapsed_secs / 60.0
                    else:
                        continue

                    # Skip if too short
                    if distance_miles < 2.5:
                        continue

                    # Determine race type
                    if 2.8 <= distance_miles <= 3.4:
                        race_type = "5K"
                        distance_miles = 3.1
                    elif 5.9 <= distance_miles <= 6.5:
                        race_type = "10K"
                        distance_miles = 6.2
                    elif 12.8 <= distance_miles <= 13.5:
                        race_type = "Half"
                        distance_miles = 13.1
                    elif 25.5 <= distance_miles <= 27.0:
                        race_type = "Marathon"
                        distance_miles = 26.2
                    else:
                        race_type = "Other"

                    # Get HR data
                    avg_hr = row.get('Average Heart Rate', '')
                    max_hr = row.get('Max Heart Rate.1', row.get('Max Heart Rate', ''))

                    race = {
                        "race_id": f"{race_type.lower().replace(' ', '_')}_{race_date.replace('-', '')}",
                        "runner_id": "my_runner",
                        "race_date": race_date,
                        "race_distance_miles": distance_miles,
                        "actual_time_minutes": round(time_minutes, 1),

                        # Runner profile (UPDATE WITH YOUR INFO)
                        "age": 35,  # TODO: Update
                        "sex": "M",  # TODO: Update
                        "max_hr": 185,  # TODO: Update
                        "experience_years": 8,  # TODO: Update
                        "resting_hr": 52,  # Optional

                        "lookback_weeks": 12 if distance_miles >= 13.1 else 8,

                        # Metadata
                        "_race_type": race_type,
                        "_activity_name": row.get('Activity Name', ''),
                        "_avg_hr": int(avg_hr) if avg_hr else None,
                        "_max_hr": int(max_hr) if max_hr else None,
                        "_filename": row.get('Filename', ''),
                    }

                    races.append(race)

                except Exception as e:
                    print(f"Skipping row: {e}")
                    continue

    # Sort by date
    races.sort(key=lambda x: x['race_date'], reverse=True)

    # Group by type
    by_type = {}
    for race in races:
        race_type = race['_race_type']
        if race_type not in by_type:
            by_type[race_type] = []
        by_type[race_type].append(race)

    # Print summary
    print(f"\nFound {len(races)} races marked as Competition:\n")

    for race_type in ['Marathon', 'Half', '10K', '5K', 'Other']:
        if race_type in by_type:
            type_races = by_type[race_type]
            print(f"\n{race_type} ({len(type_races)} races):")
            print("-" * 60)

            for i, race in enumerate(type_races[:10], 1):
                time_str = format_time(race['actual_time_minutes'])
                print(f"  {i:2d}. {race['race_date']}  {time_str:>10}  {race['_activity_name'][:30]}")

            if len(type_races) > 10:
                print(f"      ... and {len(type_races) - 10} more")

    # Save
    output_file = "strava_races.json"
    with open(output_file, 'w') as f:
        json.dump(races, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(races)} confirmed races to {output_file}")
    print(f"{'='*60}")

    print("\n📋 Next Steps:")
    print("  1. Review strava_races.json")
    print("  2. Update runner profile (age, sex, max_hr)")
    print(f"  3. python train_race_model.py {output_file}")

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
    csv_path = Path.home() / "Downloads" / "export_40402578" / "activities.csv"

    if not csv_path.exists():
        print(f"ERROR: Could not find {csv_path}")
        print("Please update the path to your Strava export CSV")
    else:
        races = extract_races_from_csv(csv_path)
