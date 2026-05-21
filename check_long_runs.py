#!/usr/bin/env python3
"""
Check what long runs exist in the data
"""

import csv

def check_long_runs(csv_path):
    """Find all runs over 15 miles"""
    long_runs = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # Get distance
                distance_str = row.get('Distance', '').strip()
                if not distance_str:
                    continue

                distance = float(distance_str)

                # Look for runs over 15 miles
                if distance >= 15.0:
                    activity_type = row.get('Activity Type', '')
                    activity_name = row.get('Activity Name', '')
                    activity_date = row.get('Activity Date', '')
                    elapsed_time = row.get('Elapsed Time', '0')

                    if activity_type == 'Run':
                        time_seconds = float(elapsed_time) if elapsed_time else 0
                        time_minutes = time_seconds / 60.0

                        long_runs.append({
                            'date': activity_date,
                            'name': activity_name,
                            'distance': distance,
                            'time_minutes': time_minutes,
                            'pace': time_minutes / distance if distance > 0 else 0
                        })

            except (ValueError, KeyError) as e:
                continue

    return sorted(long_runs, key=lambda x: x['distance'], reverse=True)


if __name__ == '__main__':
    csv_path = '/Users/osman/Downloads/export_1884062/activities.csv'

    long_runs = check_long_runs(csv_path)

    print(f"\nFound {len(long_runs)} runs over 15 miles:\n")

    for i, r in enumerate(long_runs[:20], 1):  # Show top 20
        print(f"{i}. {r['name']}")
        print(f"   Date: {r['date']}")
        print(f"   Distance: {r['distance']:.2f} miles")
        print(f"   Time: {r['time_minutes']:.1f} minutes ({r['pace']:.2f} min/mile)")
        print()

    # Also show distance distribution
    print("\nDistance ranges:")
    print(f"  Over 25 miles: {len([r for r in long_runs if r['distance'] >= 25.0])}")
    print(f"  20-25 miles: {len([r for r in long_runs if 20.0 <= r['distance'] < 25.0])}")
    print(f"  15-20 miles: {len([r for r in long_runs if 15.0 <= r['distance'] < 20.0])}")
