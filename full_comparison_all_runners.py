#!/usr/bin/env python3
"""
Full comparison of all three models across ALL races for ALL runners.
Shows training window timeframes for each race.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import statistics

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
    df = pd.read_csv(csv_path)
    activities = []
    for _, row in df.iterrows():
        if row.get('Activity Type') != 'Run':
            continue
        try:
            distance_m = float(row.get('Distance.1', 0) or 0)
            if distance_m > 0:
                distance_miles = distance_m / 1609.34
            else:
                distance_val = float(row.get('Distance', 0) or 0)
                distance_miles = distance_val / 1609.34 if distance_val > 100 else distance_val
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
            })
        except:
            continue
    return activities


def extract_csv_features(activities, race_date):
    if not activities:
        return {}, 0, None, None

    race_dt = datetime.strptime(race_date[:10], "%Y-%m-%d") if isinstance(race_date, str) else race_date
    lookback_start = race_dt - timedelta(weeks=16)
    taper_start = race_dt - timedelta(days=7)

    training = [a for a in activities if lookback_start <= a['date'] < taper_start]
    if not training:
        return {}, 0, lookback_start, taper_start

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
    mileage_consistency = 1 - (statistics.stdev(mileage_values) / statistics.mean(mileage_values)) if len(mileage_values) > 1 and statistics.mean(mileage_values) > 0 else 1.0

    tempo_runs = [a for a in training if a['pace'] and 7.0 <= a['pace'] < 8.0]
    fast_runs = [a for a in training if a['pace'] and a['pace'] < 7.5]
    quality_runs = [a for a in training if a['pace'] and a['pace'] < 8.0]

    hr_activities = [a for a in training if a['avg_hr']]
    avg_hr = sum(a['avg_hr'] for a in hr_activities) / len(hr_activities) if hr_activities else 0

    features = {
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
    }

    return features, len(training), lookback_start, taper_start


def create_hybrid_features(cache_features, csv_features):
    hybrid = {}

    # Volume - prefer CSV
    hybrid['total_weekly_mileage'] = csv_features.get('total_weekly_mileage') or cache_features.get('total_weekly_mileage', 0)
    hybrid['runs_per_week'] = csv_features.get('runs_per_week') or cache_features.get('runs_per_week', 0)
    hybrid['total_runs'] = csv_features.get('total_runs') or cache_features.get('total_runs', 0)

    # MAX for peak/long runs
    hybrid['peak_weekly_mileage'] = max(csv_features.get('peak_weekly_mileage', 0) or 0, cache_features.get('peak_weekly_mileage', 0) or 0)
    hybrid['long_run_distance'] = max(csv_features.get('long_run_distance', 0) or 0, cache_features.get('long_run_distance', 0) or 0)
    hybrid['long_run_count'] = max(csv_features.get('long_run_count', 0) or 0, cache_features.get('long_run_count', 0) or 0)

    # Consistency - average
    csv_cons = csv_features.get('mileage_consistency', 0) or 0
    cache_cons = cache_features.get('mileage_consistency', 0) or 0
    hybrid['mileage_consistency'] = (csv_cons + cache_cons) / 2 if csv_cons and cache_cons else csv_cons or cache_cons

    # Quality - MAX
    hybrid['tempo_workout_count'] = max(csv_features.get('tempo_workout_count', 0) or 0, cache_features.get('tempo_workout_count', 0) or 0)
    hybrid['fast_workout_count'] = max(csv_features.get('fast_workout_count', 0) or 0, cache_features.get('fast_workout_count', 0) or 0)
    hybrid['quality_workout_percent'] = max(csv_features.get('quality_workout_percent', 0) or 0, cache_features.get('quality_workout_percent', 0) or 0)

    # HR - prefer Cache
    hybrid['avg_hr'] = cache_features.get('avg_hr') or csv_features.get('avg_hr', 0)
    hybrid['hr_at_easy_pace'] = cache_features.get('hr_at_easy_pace', 0)
    hybrid['hr_at_marathon_pace'] = cache_features.get('hr_at_marathon_pace', 0)

    # Context from Cache
    for key in ['age_normalized', 'sex_encoded', 'max_hr_normalized', 'experience_years',
                'historical_pr_minutes', 'race_temperature', 'race_humidity',
                'race_apparent_temperature', 'race_wind_speed', 'race_distance_miles',
                'elevation_tolerance', 'taper_quality_score', 'days_since_last_hard_effort']:
        hybrid[key] = cache_features.get(key, 0)

    hybrid['training_consistency_score'] = hybrid['mileage_consistency']
    hybrid['long_run_percent_weekly'] = (hybrid['long_run_distance'] / hybrid['total_weekly_mileage'] * 100) if hybrid['total_weekly_mileage'] > 0 else 0

    return hybrid


def main():
    print("=" * 120)
    print("FULL COMPARISON: All Races, All Runners, All Models")
    print("=" * 120)

    # Load data
    with open('race_data/combined_41_features.json', 'r') as f:
        cache_races = json.load(f)

    bonked_races = [
        ('my_runner', 'marathon_20251012'),
        ('my_runner', 'marathon_20231008'),
        ('runner_2', 'marathon_20231008'),
        ('runner_3', 'marathon_20231008'),
        ('runner_sara', 'sara_marathon_20240623'),
    ]
    clean_races = [r for r in cache_races if (r['runner_id'], r['race_id']) not in bonked_races]

    # Load CSV activities
    runner_activities = {}
    for runner_id, csv_path in RUNNER_CSV_PATHS.items():
        if csv_path.exists():
            runner_activities[runner_id] = parse_strava_csv(csv_path)

    # Build all three feature sets for each race
    all_data = []

    for race in clean_races:
        runner_id = race['runner_id']
        cache_features = race.get('features', {})

        # CSV features
        csv_features = {}
        csv_activity_count = 0
        window_start = None
        window_end = None

        if runner_id in runner_activities:
            csv_features, csv_activity_count, window_start, window_end = extract_csv_features(
                runner_activities[runner_id], race['race_date']
            )

        # Hybrid features
        hybrid_features = create_hybrid_features(cache_features, csv_features)

        all_data.append({
            'runner_id': runner_id,
            'runner_name': RUNNER_NAMES.get(runner_id, runner_id),
            'race_id': race['race_id'],
            'race_date': race['race_date'][:10],
            'race_name': race.get('race_name', 'Marathon')[:30],
            'actual_time': race['actual_time_minutes'],
            'cache_features': cache_features,
            'csv_features': csv_features,
            'hybrid_features': hybrid_features,
            'csv_activity_count': csv_activity_count,
            'window_start': window_start.strftime('%Y-%m-%d') if window_start else 'N/A',
            'window_end': window_end.strftime('%Y-%m-%d') if window_end else 'N/A',
        })

    # Get feature names for each model
    cache_feature_names = sorted(all_data[0]['cache_features'].keys())
    csv_feature_names = sorted([k for k in all_data[0]['csv_features'].keys()]) if all_data[0]['csv_features'] else []
    hybrid_feature_names = sorted(all_data[0]['hybrid_features'].keys())

    # Use common features for CSV model (add missing ones from cache)
    common_csv_features = []
    for feat in cache_feature_names:
        if feat in csv_feature_names or any(d['csv_features'].get(feat) is not None for d in all_data):
            common_csv_features.append(feat)

    # Train all three models using leave-one-out for each race
    print("\nTraining models and predicting each race...")

    results = []

    for i, test_race in enumerate(all_data):
        # Training data = all except current race
        train_data = [d for j, d in enumerate(all_data) if j != i]

        # Cache model
        X_cache = np.array([[d['cache_features'].get(k, 0) or 0 for k in cache_feature_names] for d in train_data])
        y_train = np.array([d['actual_time'] for d in train_data])

        cache_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        cache_model.fit(X_cache, y_train)
        cache_pred = cache_model.predict([[test_race['cache_features'].get(k, 0) or 0 for k in cache_feature_names]])[0]

        # CSV model (use cache features as base, override with CSV where available)
        csv_features_for_model = []
        for d in train_data:
            feat_vals = []
            for k in cache_feature_names:
                csv_val = d['csv_features'].get(k)
                cache_val = d['cache_features'].get(k, 0)
                feat_vals.append(csv_val if csv_val is not None else cache_val or 0)
            csv_features_for_model.append(feat_vals)

        X_csv = np.array(csv_features_for_model)
        csv_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        csv_model.fit(X_csv, y_train)

        test_csv_features = []
        for k in cache_feature_names:
            csv_val = test_race['csv_features'].get(k)
            cache_val = test_race['cache_features'].get(k, 0)
            test_csv_features.append(csv_val if csv_val is not None else cache_val or 0)
        csv_pred = csv_model.predict([test_csv_features])[0]

        # Hybrid model
        X_hybrid = np.array([[d['hybrid_features'].get(k, 0) or 0 for k in hybrid_feature_names] for d in train_data])
        hybrid_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        hybrid_model.fit(X_hybrid, y_train)
        hybrid_pred = hybrid_model.predict([[test_race['hybrid_features'].get(k, 0) or 0 for k in hybrid_feature_names]])[0]

        results.append({
            'runner': test_race['runner_name'],
            'race_date': test_race['race_date'],
            'race_name': test_race['race_name'],
            'actual': test_race['actual_time'],
            'cache_pred': cache_pred,
            'csv_pred': csv_pred,
            'hybrid_pred': hybrid_pred,
            'cache_err': abs(cache_pred - test_race['actual_time']),
            'csv_err': abs(csv_pred - test_race['actual_time']),
            'hybrid_err': abs(hybrid_pred - test_race['actual_time']),
            'window_start': test_race['window_start'],
            'window_end': test_race['window_end'],
            'csv_activities': test_race['csv_activity_count'],
        })

    # Sort by runner then date
    results.sort(key=lambda x: (x['runner'], x['race_date']))

    # Print results grouped by runner
    print("\n" + "=" * 120)
    print("PREDICTIONS FOR ALL RACES (Leave-One-Out Cross-Validation)")
    print("Training window: 16 weeks before race, excluding last 7 days (taper)")
    print("=" * 120)

    current_runner = None
    runner_totals = {}

    for r in results:
        if r['runner'] != current_runner:
            if current_runner is not None:
                # Print runner summary
                rt = runner_totals[current_runner]
                print(f"\n  {current_runner} AVERAGE: Cache={rt['cache']/rt['count']:.1f} | CSV={rt['csv']/rt['count']:.1f} | Hybrid={rt['hybrid']/rt['count']:.1f} min error")
                print("-" * 120)

            current_runner = r['runner']
            runner_totals[current_runner] = {'cache': 0, 'csv': 0, 'hybrid': 0, 'count': 0}

            print(f"\n{'='*120}")
            print(f"RUNNER: {current_runner}")
            print(f"{'='*120}")
            print(f"\n{'Race Date':<12} {'Race Name':<25} {'Window':<25} {'#Acts':<6} {'Actual':<8} {'Cache':<8} {'CSV':<8} {'Hybrid':<8} {'Best':<8}")
            print("-" * 120)

        # Determine best model for this race
        errors = {'Cache': r['cache_err'], 'CSV': r['csv_err'], 'Hybrid': r['hybrid_err']}
        best = min(errors, key=errors.get)

        window = f"{r['window_start']} to {r['window_end']}"

        print(f"{r['race_date']:<12} {r['race_name']:<25} {window:<25} {r['csv_activities']:<6} "
              f"{format_time(r['actual']):<8} "
              f"{format_time(r['cache_pred']):<8} "
              f"{format_time(r['csv_pred']):<8} "
              f"{format_time(r['hybrid_pred']):<8} "
              f"{best:<8}")

        runner_totals[current_runner]['cache'] += r['cache_err']
        runner_totals[current_runner]['csv'] += r['csv_err']
        runner_totals[current_runner]['hybrid'] += r['hybrid_err']
        runner_totals[current_runner]['count'] += 1

    # Final runner summary
    rt = runner_totals[current_runner]
    print(f"\n  {current_runner} AVERAGE: Cache={rt['cache']/rt['count']:.1f} | CSV={rt['csv']/rt['count']:.1f} | Hybrid={rt['hybrid']/rt['count']:.1f} min error")

    # Overall summary
    print("\n" + "=" * 120)
    print("OVERALL SUMMARY")
    print("=" * 120)

    total_cache = sum(r['cache_err'] for r in results)
    total_csv = sum(r['csv_err'] for r in results)
    total_hybrid = sum(r['hybrid_err'] for r in results)
    n = len(results)

    print(f"\n{'Model':<15} {'Total Error':<15} {'Avg Error':<15} {'Races':<10}")
    print("-" * 60)
    print(f"{'Cache':<15} {total_cache:.1f} min{'':<6} {total_cache/n:.1f} min{'':<6} {n}")
    print(f"{'CSV':<15} {total_csv:.1f} min{'':<6} {total_csv/n:.1f} min{'':<6} {n}")
    print(f"{'Hybrid':<15} {total_hybrid:.1f} min{'':<6} {total_hybrid/n:.1f} min{'':<6} {n}")

    # Best model wins
    cache_wins = sum(1 for r in results if r['cache_err'] <= r['csv_err'] and r['cache_err'] <= r['hybrid_err'])
    csv_wins = sum(1 for r in results if r['csv_err'] < r['cache_err'] and r['csv_err'] <= r['hybrid_err'])
    hybrid_wins = sum(1 for r in results if r['hybrid_err'] < r['cache_err'] and r['hybrid_err'] < r['csv_err'])

    print(f"\n{'Model Wins:':<15} Cache={cache_wins}, CSV={csv_wins}, Hybrid={hybrid_wins}")

    # Per-runner summary table
    print("\n" + "=" * 120)
    print("PER-RUNNER AVERAGE ERROR")
    print("=" * 120)

    print(f"\n{'Runner':<15} {'Races':<8} {'Cache Err':<12} {'CSV Err':<12} {'Hybrid Err':<12} {'Best Model':<12}")
    print("-" * 80)

    for runner in sorted(runner_totals.keys()):
        rt = runner_totals[runner]
        cache_avg = rt['cache'] / rt['count']
        csv_avg = rt['csv'] / rt['count']
        hybrid_avg = rt['hybrid'] / rt['count']

        best = 'Cache' if cache_avg <= csv_avg and cache_avg <= hybrid_avg else ('CSV' if csv_avg <= hybrid_avg else 'Hybrid')

        print(f"{runner:<15} {rt['count']:<8} {cache_avg:.1f} min{'':<5} {csv_avg:.1f} min{'':<5} {hybrid_avg:.1f} min{'':<5} {best:<12}")


if __name__ == '__main__':
    main()
