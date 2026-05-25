#!/usr/bin/env python3
"""
Train a HYBRID model that combines:
- CSV features for volume metrics (more complete activity count)
- Cache features for quality/HR metrics (more precise when FIT parsed)
- Best of both for certain features (max long_run_distance, etc.)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import statistics
import pickle

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
            })
        except:
            continue
    return activities


def extract_csv_features(activities, race_date):
    """Extract features from CSV"""
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
        'csv_total_weekly_mileage': round(weekly_mileage, 2),
        'csv_peak_weekly_mileage': round(peak_weekly_mileage, 2),
        'csv_long_run_distance': round(long_run_distance, 2),
        'csv_long_run_count': len(long_runs),
        'csv_total_runs': total_runs,
        'csv_runs_per_week': round(runs_per_week, 2),
        'csv_mileage_consistency': round(max(0, min(1, mileage_consistency)), 3),
        'csv_tempo_workout_count': len(tempo_runs),
        'csv_fast_workout_count': len(fast_runs),
        'csv_quality_workout_percent': round(100 * len(quality_runs) / total_runs, 1) if total_runs > 0 else 0,
        'csv_avg_hr': avg_hr,
    }


def create_hybrid_features(cache_features, csv_features):
    """
    Create hybrid feature set combining best of both sources.

    Strategy:
    - Volume features: Use CSV (more complete activity count)
    - Long run distance: Use MAX of both (catches long runs either might miss)
    - Quality features: Use CSV tempo count (often higher/more accurate)
    - HR features: Use Cache when available (more precise from FIT)
    - Runner context: Use Cache (age, sex, historical PR)
    - Weather: Use Cache (same for both)
    """
    hybrid = {}

    # Volume features - prefer CSV (more activities captured)
    hybrid['total_weekly_mileage'] = csv_features.get('csv_total_weekly_mileage') or cache_features.get('total_weekly_mileage', 0)
    hybrid['runs_per_week'] = csv_features.get('csv_runs_per_week') or cache_features.get('runs_per_week', 0)
    hybrid['total_runs'] = csv_features.get('csv_total_runs') or cache_features.get('total_runs', 0)

    # Peak mileage - use MAX (catches peak either source might miss)
    hybrid['peak_weekly_mileage'] = max(
        csv_features.get('csv_peak_weekly_mileage', 0) or 0,
        cache_features.get('peak_weekly_mileage', 0) or 0
    )

    # Long run - use MAX (catches long runs either might miss)
    hybrid['long_run_distance'] = max(
        csv_features.get('csv_long_run_distance', 0) or 0,
        cache_features.get('long_run_distance', 0) or 0
    )
    hybrid['long_run_count'] = max(
        csv_features.get('csv_long_run_count', 0) or 0,
        cache_features.get('long_run_count', 0) or 0
    )

    # Consistency - average of both
    csv_cons = csv_features.get('csv_mileage_consistency', 0) or 0
    cache_cons = cache_features.get('mileage_consistency', 0) or 0
    if csv_cons and cache_cons:
        hybrid['mileage_consistency'] = (csv_cons + cache_cons) / 2
    else:
        hybrid['mileage_consistency'] = csv_cons or cache_cons

    # Quality workouts - use MAX (CSV often finds more)
    hybrid['tempo_workout_count'] = max(
        csv_features.get('csv_tempo_workout_count', 0) or 0,
        cache_features.get('tempo_workout_count', 0) or 0
    )
    hybrid['fast_workout_count'] = max(
        csv_features.get('csv_fast_workout_count', 0) or 0,
        cache_features.get('fast_workout_count', 0) or 0
    )
    hybrid['quality_workout_percent'] = max(
        csv_features.get('csv_quality_workout_percent', 0) or 0,
        cache_features.get('quality_workout_percent', 0) or 0
    )

    # HR features - prefer Cache (more precise from FIT files)
    hybrid['avg_hr'] = cache_features.get('avg_hr') or csv_features.get('csv_avg_hr', 0)
    hybrid['hr_at_easy_pace'] = cache_features.get('hr_at_easy_pace', 0)
    hybrid['hr_at_marathon_pace'] = cache_features.get('hr_at_marathon_pace', 0)

    # Runner context - from Cache only
    hybrid['age_normalized'] = cache_features.get('age_normalized', 0.9)
    hybrid['sex_encoded'] = cache_features.get('sex_encoded', 1)
    hybrid['max_hr_normalized'] = cache_features.get('max_hr_normalized', 0.9)
    hybrid['experience_years'] = cache_features.get('experience_years', 3)
    hybrid['historical_pr_minutes'] = cache_features.get('historical_pr_minutes', 0)

    # Weather - from Cache
    hybrid['race_temperature'] = cache_features.get('race_temperature', 50)
    hybrid['race_humidity'] = cache_features.get('race_humidity', 0.6)
    hybrid['race_apparent_temperature'] = cache_features.get('race_apparent_temperature', 50)
    hybrid['race_wind_speed'] = cache_features.get('race_wind_speed', 5)

    # Other features - from Cache
    hybrid['race_distance_miles'] = cache_features.get('race_distance_miles', 26.2)
    hybrid['training_consistency_score'] = hybrid['mileage_consistency']
    hybrid['long_run_percent_weekly'] = (
        hybrid['long_run_distance'] / hybrid['total_weekly_mileage'] * 100
        if hybrid['total_weekly_mileage'] > 0 else 0
    )

    # Defaults for features we can't hybrid
    hybrid['elevation_tolerance'] = cache_features.get('elevation_tolerance', 1.0)
    hybrid['taper_quality_score'] = cache_features.get('taper_quality_score', 0.5)
    hybrid['days_since_last_hard_effort'] = cache_features.get('days_since_last_hard_effort', 7)

    return hybrid


def main():
    print("=" * 100)
    print("TRAINING HYBRID MODEL: Best of Cache + CSV Features")
    print("=" * 100)

    # Load cache data
    with open('race_data/combined_41_features.json', 'r') as f:
        cache_races = json.load(f)

    # Filter bonked races
    bonked_races = [
        ('my_runner', 'marathon_20251012'),
        ('my_runner', 'marathon_20231008'),
        ('runner_2', 'marathon_20231008'),
        ('runner_3', 'marathon_20231008'),
        ('runner_sara', 'sara_marathon_20240623'),
    ]
    clean_races = [r for r in cache_races if (r['runner_id'], r['race_id']) not in bonked_races]
    print(f"\nClean races: {len(clean_races)}")

    # Load CSV activities
    print("\nLoading CSV activities...")
    runner_activities = {}
    for runner_id, csv_path in RUNNER_CSV_PATHS.items():
        if csv_path.exists():
            runner_activities[runner_id] = parse_strava_csv(csv_path)
            print(f"  {RUNNER_NAMES.get(runner_id, runner_id)}: {len(runner_activities[runner_id])} activities")

    # Create hybrid features for each race
    print("\nCreating hybrid features...")
    hybrid_races = []

    for race in clean_races:
        runner_id = race['runner_id']
        cache_features = race.get('features', {})

        # Get CSV features
        if runner_id in runner_activities:
            csv_features = extract_csv_features(runner_activities[runner_id], race['race_date'])
        else:
            csv_features = {}

        # Create hybrid
        hybrid_features = create_hybrid_features(cache_features, csv_features)

        hybrid_races.append({
            'runner_id': runner_id,
            'race_id': race['race_id'],
            'race_date': race['race_date'],
            'race_name': race.get('race_name', 'Marathon'),
            'actual_time_minutes': race['actual_time_minutes'],
            'features': hybrid_features,
        })

    print(f"  Created hybrid features for {len(hybrid_races)} races")

    # Prepare holdout split
    runner_ids = sorted(set(r['runner_id'] for r in hybrid_races))

    runners_data = {}
    for runner_id in runner_ids:
        races = [r for r in hybrid_races if r['runner_id'] == runner_id]
        races.sort(key=lambda x: x['race_date'])
        runners_data[runner_id] = races

    holdouts = {}
    training_races = []

    for runner_id, races in runners_data.items():
        if len(races) < 2:
            training_races.extend(races)
            continue

        if runner_id == 'runner_sara':
            boston_idx = None
            for i, r in enumerate(races):
                if 'Boston' in r.get('race_name', '') and '2025' in r['race_date']:
                    boston_idx = i
                    break
            if boston_idx is not None:
                holdouts[f'{runner_id}_boston'] = races[boston_idx]
                holdouts[f'{runner_id}_recent'] = races[-1]
                for i, r in enumerate(races):
                    if i != boston_idx and i != len(races) - 1:
                        training_races.append(r)
                continue

        holdouts[runner_id] = races[-1]
        training_races.extend(races[:-1])

    print(f"\nTraining: {len(training_races)} races, Holdout: {len(holdouts)} races")

    # Extract feature names
    feature_names = sorted(training_races[0]['features'].keys())
    print(f"Features: {len(feature_names)}")

    # Prepare training data
    X_train = []
    y_train = []
    for race in training_races:
        features = race['features']
        X_train.append([features.get(k, 0) or 0 for k in feature_names])
        y_train.append(race['actual_time_minutes'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train model
    print("\n" + "=" * 100)
    print("TRAINING HYBRID MODEL")
    print("=" * 100)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    print(f"\nCV MAE: {cv_mae:.1f} minutes")

    # Train on full data
    model.fit(X_train, y_train)

    # Predict holdouts
    print("\n" + "=" * 100)
    print("HOLDOUT PREDICTIONS")
    print("=" * 100)

    print(f"\n{'Runner':<15} {'Predicted':<10} {'Actual':<10} {'Error':<10}")
    print("-" * 50)

    total_error = 0
    predictions = []

    for holdout_key, holdout in holdouts.items():
        if holdout_key.startswith('runner_sara'):
            name = "Sara (Boston)" if 'boston' in holdout_key else "Sara (Recent)"
        else:
            name = RUNNER_NAMES.get(holdout_key, holdout_key)

        features = [holdout['features'].get(k, 0) or 0 for k in feature_names]
        pred = model.predict([features])[0]
        actual = holdout['actual_time_minutes']
        error = abs(pred - actual)
        total_error += error

        predictions.append({'name': name, 'pred': pred, 'actual': actual, 'error': error})
        print(f"{name:<15} {format_time(pred):<10} {format_time(actual):<10} {error:.1f} min")

    avg_error = total_error / len(holdouts)
    print(f"\n{'AVERAGE':<15} {'':<10} {'':<10} {avg_error:.1f} min")

    # Compare with previous results
    print("\n" + "=" * 100)
    print("COMPARISON WITH PREVIOUS MODELS")
    print("=" * 100)

    print(f"""
    Model                    CV MAE      Holdout MAE
    ------------------------------------------------
    Cache-only (46 feat)     14.7 min    5.5 min
    CSV-only (37 feat)       15.4 min    5.0 min
    HYBRID (best of both)    {cv_mae:.1f} min    {avg_error:.1f} min
    """)

    # Feature importance
    print("\n" + "=" * 100)
    print("TOP 10 FEATURES (Hybrid Model)")
    print("=" * 100)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:<35}: {importances[idx]*100:.1f}%")

    # Save hybrid model
    print("\n" + "=" * 100)
    print("SAVING HYBRID MODEL")
    print("=" * 100)

    model_data = {
        'model': model,
        'feature_names': feature_names,
        'cv_mae': cv_mae,
        'holdout_mae': avg_error,
        'training_races': len(training_races),
        'model_type': 'hybrid',
    }

    with open('race_time_model_hybrid.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("\nSaved to: race_time_model_hybrid.pkl")

    # Also save hybrid dataset for reference
    with open('race_data/combined_hybrid_features.json', 'w') as f:
        json.dump(hybrid_races, f, indent=2, default=str)
    print("Saved hybrid dataset to: race_data/combined_hybrid_features.json")


if __name__ == '__main__':
    main()
