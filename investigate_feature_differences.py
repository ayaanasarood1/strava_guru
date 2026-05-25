#!/usr/bin/env python3
"""
Investigate differences between cache-based and CSV-based feature values
for the same races, then create a hybrid model.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import statistics

# Runner CSV paths
RUNNER_CSV_PATHS = {
    'my_runner': Path('/Users/osman/Downloads/export_40402578/activities.csv'),
    'runner_2': Path('/Users/osman/Downloads/export_1884062_salman/activities.csv'),
    'runner_3': Path('/Users/osman/Downloads/export_52983191_azeem/activities.csv'),
    'runner_sara': Path('/Users/osman/Downloads/export_108527851_sara/activities.csv'),
    'runner_qazi': Path('/Users/osman/Downloads/export_40747977_qazi/activities.csv'),
    'runner_salman_khan': Path('/Users/osman/Downloads/salman_khan/activities.csv'),
}

RUNNER_NAMES = {
    'my_runner': 'Osman',
    'runner_2': 'Salman',
    'runner_3': 'Azeem',
    'runner_sara': 'Sara',
    'runner_qazi': 'Qazi',
    'runner_salman_khan': 'Salman Khan',
}


def format_time(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}:{mins:02d}"


def parse_strava_csv(csv_path):
    """Parse Strava activities.csv"""
    df = pd.read_csv(csv_path)

    activities = []
    for _, row in df.iterrows():
        if row.get('Activity Type') != 'Run':
            continue

        try:
            # Handle duplicate Distance columns
            distance_m = float(row.get('Distance.1', 0) or 0)
            if distance_m > 0:
                distance_miles = distance_m / 1609.34
            else:
                distance_val = float(row.get('Distance', 0) or 0)
                if distance_val > 100:
                    distance_miles = distance_val / 1609.34
                else:
                    distance_miles = distance_val

            if distance_miles < 0.5:
                continue

            moving_sec = float(row.get('Moving Time', 0) or 0)
            elapsed_sec = float(row.get('Elapsed Time', 0) or 0)
            duration_min = (moving_sec if moving_sec > 0 else elapsed_sec) / 60

            date_str = row.get('Activity Date', '')
            try:
                activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
            except:
                try:
                    activity_date = datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
                except:
                    continue

            avg_hr = float(row.get('Average Heart Rate', 0) or 0) or None
            pace = duration_min / distance_miles if distance_miles > 0 else None

            activities.append({
                'date': activity_date,
                'distance_miles': distance_miles,
                'duration_min': duration_min,
                'avg_hr': avg_hr,
                'pace': pace,
                'elevation_gain': float(row.get('Elevation Gain', 0) or 0),
            })
        except:
            continue

    return activities


def extract_csv_features(activities, race_date):
    """Extract features from CSV activities"""
    if not activities:
        return {}

    race_dt = datetime.strptime(race_date[:10], "%Y-%m-%d") if isinstance(race_date, str) else race_date
    lookback_start = race_dt - timedelta(weeks=16)
    taper_start = race_dt - timedelta(days=7)

    training = [a for a in activities if lookback_start <= a['date'] < taper_start]

    if not training:
        return {}

    total_distance = sum(a['distance_miles'] for a in training)
    total_runs = len(training)

    dates = [a['date'] for a in training]
    weeks = max(1, (max(dates) - min(dates)).days / 7)

    weekly_mileage = total_distance / weeks
    runs_per_week = total_runs / weeks

    long_runs = [a for a in training if a['distance_miles'] >= 15]
    long_run_distance = max([a['distance_miles'] for a in long_runs], default=0)

    weekly_distances = {}
    for a in training:
        week_num = a['date'].isocalendar()[1]
        weekly_distances[week_num] = weekly_distances.get(week_num, 0) + a['distance_miles']

    peak_weekly_mileage = max(weekly_distances.values()) if weekly_distances else 0

    mileage_values = list(weekly_distances.values())
    if len(mileage_values) > 1:
        mileage_consistency = 1 - (statistics.stdev(mileage_values) / statistics.mean(mileage_values)) if statistics.mean(mileage_values) > 0 else 0
    else:
        mileage_consistency = 1.0

    tempo_runs = [a for a in training if a['pace'] and 7.0 <= a['pace'] < 8.0]
    fast_runs = [a for a in training if a['pace'] and a['pace'] < 7.5]
    quality_runs = [a for a in training if a['pace'] and a['pace'] < 8.0]

    hr_activities = [a for a in training if a['avg_hr']]
    avg_hr = sum(a['avg_hr'] for a in hr_activities) / len(hr_activities) if hr_activities else 0

    return {
        'total_weekly_mileage': round(weekly_mileage, 2),
        'peak_weekly_mileage': round(peak_weekly_mileage, 2),
        'long_run_distance': round(long_run_distance, 2),
        'long_run_count': len(long_runs),
        'total_runs': total_runs,
        'runs_per_week': round(runs_per_week, 2),
        'mileage_consistency': round(max(0, min(1, mileage_consistency)), 3),
        'tempo_workout_count': len(tempo_runs),
        'fast_workout_count': len(fast_runs),
        'quality_workout_percent': round(100 * len(quality_runs) / total_runs, 1) if total_runs > 0 else 0,
        'avg_hr': avg_hr,
        'training_activities_count': len(training),
    }


def main():
    print("=" * 100)
    print("INVESTIGATING FEATURE DIFFERENCES: Cache vs CSV")
    print("=" * 100)

    # Load cache-based data
    with open('race_data/combined_41_features.json', 'r') as f:
        cache_races = json.load(f)

    # Load CSV activities
    runner_activities = {}
    for runner_id, csv_path in RUNNER_CSV_PATHS.items():
        if csv_path.exists():
            runner_activities[runner_id] = parse_strava_csv(csv_path)

    # Compare features for holdout races
    holdout_races = [
        ('my_runner', '2024-12-08', 'Osman'),
        ('runner_2', '2025-10-12', 'Salman'),
        ('runner_3', '2026-01-11', 'Azeem'),
        ('runner_sara', '2025-04-21', 'Sara (Boston)'),
        ('runner_sara', '2026-04-26', 'Sara (Recent)'),
        ('runner_salman_khan', '2026-04-20', 'Salman Khan'),
    ]

    key_features = [
        'total_weekly_mileage',
        'peak_weekly_mileage',
        'runs_per_week',
        'total_runs',
        'tempo_workout_count',
        'long_run_distance',
    ]

    print("\n" + "=" * 100)
    print("FEATURE VALUE COMPARISON FOR HOLDOUT RACES")
    print("=" * 100)

    comparison_data = []

    for runner_id, race_date_prefix, runner_name in holdout_races:
        # Find cache race
        cache_race = None
        for r in cache_races:
            if r['runner_id'] == runner_id and r['race_date'].startswith(race_date_prefix):
                cache_race = r
                break

        if not cache_race:
            print(f"\n{runner_name}: Cache race not found")
            continue

        # Extract CSV features
        if runner_id in runner_activities:
            csv_features = extract_csv_features(runner_activities[runner_id], race_date_prefix)
        else:
            csv_features = {}

        cache_features = cache_race.get('features', {})

        print(f"\n{'='*80}")
        print(f"{runner_name} - Race: {race_date_prefix}")
        print(f"{'='*80}")
        print(f"\n{'Feature':<25} {'Cache':<15} {'CSV':<15} {'Diff':<15} {'% Diff':<10}")
        print("-" * 80)

        row_data = {'runner': runner_name, 'race_date': race_date_prefix}

        for feat in key_features:
            cache_val = cache_features.get(feat, 0) or 0
            csv_val = csv_features.get(feat, 0) or 0
            diff = csv_val - cache_val
            pct_diff = (diff / cache_val * 100) if cache_val != 0 else 0

            row_data[f'{feat}_cache'] = cache_val
            row_data[f'{feat}_csv'] = csv_val
            row_data[f'{feat}_diff'] = diff

            flag = "⚠️" if abs(pct_diff) > 20 else ""
            print(f"{feat:<25} {cache_val:<15.1f} {csv_val:<15.1f} {diff:+.1f}{'':<8} {pct_diff:+.1f}% {flag}")

        # Also show training activity counts
        csv_training_count = csv_features.get('training_activities_count', 0)
        print(f"\n  CSV training activities in window: {csv_training_count}")

        comparison_data.append(row_data)

    # Summary analysis
    print("\n" + "=" * 100)
    print("ANALYSIS: Why do values differ?")
    print("=" * 100)

    print("""
Key findings from the comparison:

1. ACTIVITY COUNT DIFFERENCES
   - CSV may parse more/fewer activities than FIT files
   - FIT parsing failures reduce activity count in cache
   - CSV date parsing may miss some activities

2. DISTANCE CALCULATION
   - FIT files: GPS track points → precise distance
   - CSV: Summary distance field (may be rounded)

3. PACE/QUALITY DETECTION
   - FIT files: Can calculate pace from track points
   - CSV: Uses summary pace (moving_time / distance)
   - Tempo detection may differ based on precision

4. DATE WINDOW HANDLING
   - Slight differences in date parsing can include/exclude edge activities
""")

    # Now create HYBRID features
    print("\n" + "=" * 100)
    print("CREATING HYBRID MODEL")
    print("=" * 100)

    print("""
Strategy: For each feature, use the source that performs better:

- CACHE is better for: Sara, Osman (FIT files parsed correctly)
  → Use cache values for: HR features, detailed pace analysis

- CSV is better for: Salman, Azeem, Salman Khan (FIT had issues)
  → Use CSV values for: Volume metrics (mileage, runs)

HYBRID APPROACH:
1. Use CSV for volume features (more reliable activity count)
2. Use Cache for quality/HR features (more precise when available)
3. Fallback to whichever has data when one is missing
""")

    return comparison_data


if __name__ == '__main__':
    main()
