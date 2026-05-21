#!/usr/bin/env python3
"""
Extract training features from activities.csv for runner 2
"""

import json
import csv
from datetime import datetime, timedelta
from collections import defaultdict

def load_activities_from_csv(csv_path):
    """Load all activities from CSV"""
    activities = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                activity_type = row.get('Activity Type', '')
                if activity_type != 'Run':
                    continue

                # Parse date
                date_str = row.get('Activity Date', '')
                if not date_str:
                    continue

                # Parse "May 17, 2026, 4:34:19 PM" format
                activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")

                # Get key metrics
                distance_m = float(row.get('Distance', 0) or 0)
                distance_mi = distance_m / 1609.34

                duration_s = float(row.get('Moving Time', 0) or 0)
                duration_min = duration_s / 60.0

                avg_hr = int(float(row.get('Average Heart Rate', 0) or 0))
                max_hr = int(float(row.get('Max Heart Rate', 0) or 0))

                if distance_mi > 0.5 and duration_min > 5:  # Filter out very short runs
                    activities.append({
                        'date': activity_date,
                        'distance_mi': distance_mi,
                        'duration_min': duration_min,
                        'avg_hr': avg_hr,
                        'max_hr': max_hr,
                        'pace': duration_min / distance_mi if distance_mi > 0 else 0
                    })

            except Exception as e:
                continue

    return sorted(activities, key=lambda x: x['date'])


def extract_simple_features(activities, race_date, lookback_weeks=12):
    """Extract simplified features from activities"""

    end_date = race_date - timedelta(days=7)  # Exclude taper week
    start_date = end_date - timedelta(weeks=lookback_weeks)

    # Filter to training window
    training_runs = [a for a in activities if start_date <= a['date'] < end_date]

    if not training_runs:
        return None

    # Calculate features
    total_miles = sum(r['distance_mi'] for r in training_runs)
    total_weeks = lookback_weeks

    # Weekly stats
    weekly_miles = defaultdict(float)
    for run in training_runs:
        week = (run['date'] - start_date).days // 7
        weekly_miles[week] += run['distance_mi']

    avg_weekly_miles = total_miles / total_weeks if total_weeks > 0 else 0
    peak_weekly_miles = max(weekly_miles.values()) if weekly_miles else 0

    # Long runs
    long_runs = [r for r in training_runs if r['distance_mi'] >= 10]
    avg_long_run = sum(r['distance_mi'] for r in long_runs) / len(long_runs) if long_runs else 0
    max_long_run = max([r['distance_mi'] for r in long_runs]) if long_runs else 0

    # HR stats
    runs_with_hr = [r for r in training_runs if r['avg_hr'] > 0]
    avg_hr = sum(r['avg_hr'] for r in runs_with_hr) / len(runs_with_hr) if runs_with_hr else 0

    # Pace stats
    avg_pace = sum(r['pace'] for r in training_runs) / len(training_runs) if training_runs else 0

    return {
        'total_weekly_mileage': avg_weekly_miles,
        'peak_weekly_mileage': peak_weekly_miles,
        'long_run_distance': max_long_run,
        'total_runs': len(training_runs),
        'runs_per_week': len(training_runs) / total_weeks if total_weeks > 0 else 0,
        'avg_hr': avg_hr,
        'avg_pace': avg_pace,
        'num_long_runs': len(long_runs)
    }


def main():
    csv_path = '/Users/osman/Downloads/export_1884062/activities.csv'

    print("Loading activities from CSV...")
    activities = load_activities_from_csv(csv_path)
    print(f"Loaded {len(activities)} running activities")

    # Load marathons
    with open('runner2_marathons.json', 'r') as f:
        marathons = json.load(f)

    print(f"\nExtracting features for {len(marathons)} marathons...")

    race_data = []
    for marathon in marathons:
        race_date = datetime.fromisoformat(marathon['race_date'])

        features = extract_simple_features(activities, race_date, marathon['lookback_weeks'])

        if features:
            race_data.append({
                'race_id': marathon['race_id'],
                'runner_id': marathon['runner_id'],
                'race_date': marathon['race_date'],
                'race_distance_miles': marathon['race_distance_miles'],
                'actual_time_minutes': marathon['actual_time_minutes'],
                'features': features
            })
            print(f"✓ {marathon['_race_name']}: {features['total_runs']} runs, {features['total_weekly_mileage']:.1f} mi/week")
        else:
            print(f"✗ {marathon['_race_name']}: No training data found")

    # Save
    output_path = 'race_data/runner2_simple_dataset.json'
    with open(output_path, 'w') as f:
        json.dump(race_data, f, indent=2)

    print(f"\n✓ Saved {len(race_data)} races to {output_path}")

    # Display summary
    print("\n" + "="*80)
    print("Runner 2 Summary:")
    print(f"  Total races with training data: {len(race_data)}")

    mileages = [r['features']['total_weekly_mileage'] for r in race_data]
    print(f"  Weekly mileage range: {min(mileages):.1f} - {max(mileages):.1f} mi/week")

    times = [r['actual_time_minutes'] for r in race_data]
    print(f"  Marathon times: {min(times)//60}:{int(min(times)%60):02d} - {max(times)//60}:{int(max(times)%60):02d}")
    print("="*80)


if __name__ == '__main__':
    main()
