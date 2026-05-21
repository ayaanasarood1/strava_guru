#!/usr/bin/env python3
"""
Find marathon races from another runner's Strava export
"""

import csv
from datetime import datetime

def find_marathons(csv_path):
    """Find potential marathon races from activities.csv"""
    marathons = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # Get distance (in METERS based on the CSV format)
                distance_str = row.get('Distance', '').strip()
                if not distance_str:
                    continue

                distance_meters = float(distance_str)
                distance_miles = distance_meters / 1609.34

                # Marathon range: 41,000 to 44,000 meters (25.5 to 27.3 miles)
                # This allows for GPS variance
                if 41000 <= distance_meters <= 44000:
                    # Get other fields
                    activity_type = row.get('Activity Type', '')
                    activity_name = row.get('Activity Name', '')
                    activity_date = row.get('Activity Date', '')
                    elapsed_time = row.get('Elapsed Time', '0')
                    avg_hr = row.get('Average Heart Rate', '0')
                    max_hr = row.get('Max Heart Rate', '0')

                    # Convert time to minutes
                    time_seconds = float(elapsed_time) if elapsed_time else 0
                    time_minutes = time_seconds / 60.0

                    # Sanity check: marathon should take 2-6 hours (120-360 minutes)
                    # and should be a Run activity
                    if (activity_type == 'Run' and
                        120 <= time_minutes <= 360):

                        pace = time_minutes / distance_miles if distance_miles > 0 else 0

                        marathons.append({
                            'date': activity_date,
                            'name': activity_name,
                            'distance_miles': distance_miles,
                            'distance_meters': distance_meters,
                            'time_minutes': time_minutes,
                            'time_formatted': f"{int(time_minutes // 60)}:{int(time_minutes % 60):02d}",
                            'pace': pace,
                            'avg_hr': avg_hr,
                            'max_hr': max_hr,
                            'type': activity_type
                        })

            except (ValueError, KeyError) as e:
                continue

    return sorted(marathons, key=lambda x: x['date'])


if __name__ == '__main__':
    csv_path = '/Users/osman/Downloads/export_1884062/activities.csv'

    marathons = find_marathons(csv_path)

    print(f"\nFound {len(marathons)} marathons:\n")
    print("=" * 100)

    for i, m in enumerate(marathons, 1):
        print(f"\n{i}. {m['name']}")
        print(f"   Date: {m['date']}")
        print(f"   Distance: {m['distance_miles']:.2f} miles ({m['distance_meters']:.0f} meters)")
        print(f"   Time: {m['time_formatted']} ({m['time_minutes']:.1f} minutes)")
        print(f"   Pace: {m['pace']:.2f} min/mile")
        print(f"   Avg HR: {m['avg_hr']}, Max HR: {m['max_hr']}")

    print("\n" + "=" * 100)
    print(f"\nTotal: {len(marathons)} marathons")
