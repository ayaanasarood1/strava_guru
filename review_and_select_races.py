#!/usr/bin/env python3
"""
Interactive Race Selection
Review candidates and confirm which are actual races
"""

import json
import sqlite3
import csv
from pathlib import Path
from datetime import datetime


def get_activity_names_from_csv(csv_path):
    """Get activity names from Strava CSV"""
    names = {}

    if not csv_path.exists():
        return names

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('Filename', '')
            if filename:
                names[filename] = {
                    'name': row.get('Activity Name', ''),
                    'description': row.get('Activity Description', '')
                }

    return names


def review_races():
    """Load likely races and match with Strava names for review"""

    # Load likely races
    with open('likely_races.json', 'r') as f:
        candidates = json.load(f)

    # Get activity names from Strava CSV
    csv_path = Path.home() / "Downloads" / "export_40402578" / "activities.csv"
    activity_names = get_activity_names_from_csv(csv_path)

    print("=" * 70)
    print("Review Race Candidates")
    print("=" * 70)
    print("\nFor each activity, decide if it was an actual race")
    print("(not a training run, tempo workout, or time trial)\n")

    confirmed_races = []

    # Group by type
    by_type = {}
    for race in candidates:
        race_type = race['_race_type']
        if race_type not in by_type:
            by_type[race_type] = []
        by_type[race_type].append(race)

    for race_type in ['Marathon', 'Half', '10K', '5K']:
        if race_type not in by_type:
            continue

        races = by_type[race_type]
        print(f"\n{'='*70}")
        print(f"{race_type} Candidates ({len(races)} found)")
        print(f"{'='*70}\n")

        for i, race in enumerate(races, 1):
            time_str = format_time(race['actual_time_minutes'])
            filename = race['_file_name']

            # Get Strava name if available
            strava_info = activity_names.get(filename, {})
            name = strava_info.get('name', 'No name available')
            desc = strava_info.get('description', '')[:60]

            print(f"{i:2d}. {race['race_date']}  {time_str:>10}  "
                  f"(HR: {race.get('_avg_hr', '?')})")
            print(f"    Name: {name}")
            if desc:
                print(f"    Desc: {desc}...")
            print()

            # Ask user
            response = input(f"    Was this an actual RACE? (y/n/q to quit): ").strip().lower()

            if response == 'q':
                print("\nStopping review...")
                break
            elif response == 'y':
                confirmed_races.append(race)
                print("    ✓ Added as race\n")
            else:
                print("    ✗ Skipped\n")

        if response == 'q':
            break

    # Save confirmed races
    if confirmed_races:
        output_file = "confirmed_races.json"
        with open(output_file, 'w') as f:
            json.dump(confirmed_races, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Saved {len(confirmed_races)} confirmed races to {output_file}")
        print(f"{'='*70}")

        # Show summary
        summary = {}
        for race in confirmed_races:
            rt = race['_race_type']
            summary[rt] = summary.get(rt, 0) + 1

        print("\nConfirmed Races:")
        for race_type, count in sorted(summary.items()):
            print(f"  {race_type}: {count}")

        print(f"\nNext: python train_race_model.py {output_file}")

    else:
        print("\nNo races confirmed.")

    return confirmed_races


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
    if not Path('likely_races.json').exists():
        print("ERROR: likely_races.json not found")
        print("Run: python extract_races_smart.py first")
    else:
        races = review_races()
