#!/usr/bin/env python3
"""
Show Race Candidates with Strava Names
Display all candidates for manual review
"""

import json
import csv
from pathlib import Path


def show_candidates():
    """Display all race candidates with Strava names"""

    # Load likely races
    with open('likely_races.json', 'r') as f:
        candidates = json.load(f)

    # Get activity names from Strava CSV
    csv_path = Path.home() / "Downloads" / "export_40402578" / "activities.csv"
    activity_names = {}

    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('Filename', '')
                if filename:
                    activity_names[filename] = {
                        'name': row.get('Activity Name', ''),
                        'description': row.get('Activity Description', '')[:80]
                    }

    print("=" * 80)
    print("Race Candidates - Review and Delete Non-Races from likely_races.json")
    print("=" * 80)

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
        print(f"\n{'='*80}")
        print(f"{race_type} ({len(races)} candidates)")
        print(f"{'='*80}\n")

        for i, race in enumerate(races, 1):
            time_str = format_time(race['actual_time_minutes'])
            filename = race['_file_name']

            # Get Strava name
            strava_info = activity_names.get(filename, {})
            name = strava_info.get('name', 'No name available')

            print(f"{i:2d}. {race['race_date']}  {time_str:>10}  HR: {race.get('_avg_hr', '?'):>3}")
            print(f"    race_id: {race['race_id']}")
            print(f"    Name: {name}")

            desc = strava_info.get('description', '')
            if desc:
                print(f"    Desc: {desc}")
            print()

    print("=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Review the candidates above")
    print("2. Open likely_races.json in a text editor")
    print("3. Delete any entries that are NOT actual races")
    print("   (Search for the race_id to find the entry)")
    print("4. Save the file")
    print("5. Run: python train_race_model.py likely_races.json")
    print()
    print("OR if most look good, just train with them all:")
    print("   python train_race_model.py likely_races.json")
    print("=" * 80)


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
    show_candidates()
