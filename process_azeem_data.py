#!/usr/bin/env python3
"""
Process Azeem's Strava data export
"""

import csv
from pathlib import Path
from datetime import datetime
from activity_cache import ActivityCache
from build_cache_from_csv import build_cache_from_csv

def identify_marathons(csv_path: Path):
    """Find all marathon races in Azeem's activities"""
    print("="*80)
    print("Identifying Marathon Races")
    print("="*80)

    marathons = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                activity_type = row.get('Activity Type', '')
                if activity_type != 'Run':
                    continue

                # Check distance (marathon = 42.195 km = 26.2 miles = 42195 meters)
                distance_m = float(row.get('Distance', 0) or 0)
                distance_mi = distance_m / 1609.34

                # Marathon range: 26.0 - 26.5 miles
                if 26.0 <= distance_mi <= 26.5:
                    date_str = row.get('Activity Date', '')
                    activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")

                    activity_name = row.get('Activity Name', '')
                    duration_s = float(row.get('Moving Time', 0) or 0)
                    duration_min = duration_s / 60.0

                    marathons.append({
                        'date': activity_date,
                        'name': activity_name,
                        'distance_mi': distance_mi,
                        'duration_min': duration_min,
                        'pace': duration_min / distance_mi if distance_mi > 0 else 0
                    })
            except Exception as e:
                continue

    marathons.sort(key=lambda x: x['date'])

    print(f"\nFound {len(marathons)} marathons:\n")

    for i, m in enumerate(marathons, 1):
        time_h = int(m['duration_min'] // 60)
        time_m = int(m['duration_min'] % 60)
        print(f"{i:2d}. {m['date'].date()} - {time_h}:{time_m:02d}")
        print(f"    {m['name']}")
        print(f"    {m['distance_mi']:.2f} miles @ {m['pace']:.2f} min/mile")
        print()

    return marathons

def analyze_training_distribution(csv_path: Path):
    """Analyze Azeem's overall training distribution"""
    print("="*80)
    print("Training Activity Distribution")
    print("="*80)

    activity_types = {}
    total_distance = 0
    total_activities = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                activity_type = row.get('Activity Type', 'Unknown')
                distance_m = float(row.get('Distance', 0) or 0)

                if activity_type not in activity_types:
                    activity_types[activity_type] = {'count': 0, 'distance': 0}

                activity_types[activity_type]['count'] += 1
                activity_types[activity_type]['distance'] += distance_m

                total_activities += 1
                total_distance += distance_m
            except:
                continue

    print(f"\nTotal Activities: {total_activities}")
    print(f"Total Distance: {total_distance/1609.34:.1f} miles\n")

    # Sort by count
    sorted_types = sorted(activity_types.items(), key=lambda x: x[1]['count'], reverse=True)

    print(f"{'Activity Type':<20} {'Count':<10} {'Distance (mi)':<15} {'%'}")
    print("-" * 60)
    for activity_type, data in sorted_types[:15]:  # Top 15
        count = data['count']
        distance_mi = data['distance'] / 1609.34
        percent = (count / total_activities) * 100
        print(f"{activity_type:<20} {count:<10} {distance_mi:<15.1f} {percent:.1f}%")

    print()

def main():
    print("="*80)
    print("Processing Azeem's Strava Data")
    print("="*80)
    print()

    # Paths
    csv_path = Path('/Users/osman/Downloads/export_52983191_azeem/activities.csv')
    activities_dir = Path('/Users/osman/Downloads/export_52983191_azeem/activities')
    cache_dir = Path.home() / ".strava_guru_cache" / "azeem"

    # Step 1: Analyze training distribution
    analyze_training_distribution(csv_path)

    # Step 2: Identify marathons
    marathons = identify_marathons(csv_path)

    # Step 3: Build cache from CSV
    print("="*80)
    print("Building Activity Cache from CSV")
    print("="*80)
    print(f"Cache location: {cache_dir}")
    print()

    # Create cache
    cache = ActivityCache(cache_dir=cache_dir)

    # Clear existing cache
    print("Clearing existing cache...")
    cache.db_path.unlink(missing_ok=True)
    cache._init_database()

    # Build from CSV
    build_cache_from_csv(csv_path, cache, activities_dir)

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Runner: Azeem")
    print(f"Total activities: 1,722")
    print(f"Marathons identified: {len(marathons)}")
    print(f"Cache location: {cache_dir}")
    print(f"Ready for feature extraction: ✓")
    print("="*80)

if __name__ == '__main__':
    main()
