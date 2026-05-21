#!/usr/bin/env python3
"""
Extract simple features for your marathons from activity cache
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

def get_training_runs(db_path, race_date, lookback_weeks=12):
    """Get training runs from activity cache"""
    end_date = race_date - timedelta(days=7)
    start_date = end_date - timedelta(weeks=lookback_weeks)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT
            activity_date,
            distance_meters,
            duration_seconds,
            avg_heart_rate,
            avg_pace
        FROM activities
        WHERE activity_type = 'running'
          AND activity_date >= ?
          AND activity_date < ?
          AND distance_meters > 800
        ORDER BY activity_date
    """

    cursor.execute(query, (start_date.isoformat(), end_date.isoformat()))
    rows = cursor.fetchall()
    conn.close()

    runs = []
    for row in rows:
        runs.append({
            'date': datetime.fromisoformat(row[0]),
            'distance_mi': row[1] / 1609.34,
            'duration_min': row[2] / 60.0,
            'avg_hr': row[3] or 0,
            'avg_pace': row[4] or 0
        })

    return runs


def extract_simple_features(runs, lookback_weeks=12):
    """Extract simplified features from runs"""
    if not runs:
        return None

    total_miles = sum(r['distance_mi'] for r in runs)
    total_weeks = lookback_weeks

    # Weekly stats
    start_date = min(r['date'] for r in runs)
    weekly_miles = defaultdict(float)
    for run in runs:
        week = (run['date'] - start_date).days // 7
        weekly_miles[week] += run['distance_mi']

    avg_weekly_miles = total_miles / total_weeks if total_weeks > 0 else 0
    peak_weekly_miles = max(weekly_miles.values()) if weekly_miles else 0

    # Long runs
    long_runs = [r for r in runs if r['distance_mi'] >= 10]
    max_long_run = max([r['distance_mi'] for r in long_runs]) if long_runs else 0

    # HR stats
    runs_with_hr = [r for r in runs if r['avg_hr'] > 0]
    avg_hr = sum(r['avg_hr'] for r in runs_with_hr) / len(runs_with_hr) if runs_with_hr else 0

    # Pace stats
    runs_with_pace = [r for r in runs if r['avg_pace'] > 0]
    avg_pace = sum(r['avg_pace'] for r in runs_with_pace) / len(runs_with_pace) if runs_with_pace else 0

    return {
        'total_weekly_mileage': avg_weekly_miles,
        'peak_weekly_mileage': peak_weekly_miles,
        'long_run_distance': max_long_run,
        'total_runs': len(runs),
        'runs_per_week': len(runs) / total_weeks if total_weeks > 0 else 0,
        'avg_hr': avg_hr,
        'avg_pace': avg_pace,
        'num_long_runs': len(long_runs)
    }


def main():
    # Load your marathons
    with open('my_actual_marathons.json', 'r') as f:
        marathons = json.load(f)

    # Your activity cache
    cache_path = Path.home() / ".strava_guru_cache" / "activities.db"

    print(f"Using cache: {cache_path}")
    print(f"Extracting features for {len(marathons)} marathons...\n")

    your_data = []
    for marathon in marathons:
        race_date = datetime.fromisoformat(marathon['race_date'])

        # Get training runs
        runs = get_training_runs(cache_path, race_date, marathon['lookback_weeks'])

        # Extract features
        features = extract_simple_features(runs, marathon['lookback_weeks'])

        if features:
            your_data.append({
                'race_id': marathon['race_id'],
                'runner_id': marathon['runner_id'],
                'race_date': marathon['race_date'],
                'race_distance_miles': marathon['race_distance_miles'],
                'actual_time_minutes': marathon['actual_time_minutes'],
                'features': features
            })
            print(f"✓ {marathon.get('_race_name', marathon['race_date'])}: {features['total_runs']} runs, {features['total_weekly_mileage']:.1f} mi/week")
        else:
            print(f"✗ {marathon.get('_race_name', marathon['race_date'])}: No training data")

    # Save
    output_path = 'race_data/your_simple_dataset.json'
    with open(output_path, 'w') as f:
        json.dump(your_data, f, indent=2)

    print(f"\n✓ Saved {len(your_data)} races to {output_path}")

    # Summary
    print("\n" + "="*80)
    print("Your Marathon Training Summary:")
    mileages = [r['features']['total_weekly_mileage'] for r in your_data]
    print(f"  Weekly mileage range: {min(mileages):.1f} - {max(mileages):.1f} mi/week")

    times = [r['actual_time_minutes'] for r in your_data]
    print(f"  Marathon times: {min(times)//60}:{int(min(times)%60):02d} - {max(times)//60}:{int(max(times)%60):02d}")
    print("="*80)


if __name__ == '__main__':
    main()
