#!/usr/bin/env python3
"""
Compare prediction accuracy: Cache-based features vs CSV-only features
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
    """Parse Strava activities.csv - handle duplicate columns"""
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

            # Parse time
            moving_sec = float(row.get('Moving Time', 0) or 0)
            elapsed_sec = float(row.get('Elapsed Time', 0) or 0)
            duration_min = (moving_sec if moving_sec > 0 else elapsed_sec) / 60

            # Parse date
            date_str = row.get('Activity Date', '')
            try:
                activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
            except:
                try:
                    activity_date = datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
                except:
                    continue

            # Parse HR
            avg_hr = float(row.get('Average Heart Rate', 0) or 0) or None

            # Calculate pace
            pace = duration_min / distance_miles if distance_miles > 0 else None

            activities.append({
                'date': activity_date,
                'distance_miles': distance_miles,
                'duration_min': duration_min,
                'avg_hr': avg_hr,
                'pace': pace,
                'elevation_gain': float(row.get('Elevation Gain', 0) or 0),
            })
        except Exception as e:
            continue

    return activities


def extract_csv_features(activities, race_date, runner_context, weather):
    """Extract features from CSV activities only"""
    features = {}

    if not activities:
        return features

    # Filter to training window (4 months before race, excluding last week taper)
    race_dt = datetime.strptime(race_date[:10], "%Y-%m-%d") if isinstance(race_date, str) else race_date
    lookback_start = race_dt - timedelta(weeks=16)
    taper_start = race_dt - timedelta(days=7)

    training = [a for a in activities if lookback_start <= a['date'] < taper_start]

    if not training:
        return features

    # Basic volume features
    total_distance = sum(a['distance_miles'] for a in training)
    total_runs = len(training)

    dates = [a['date'] for a in training]
    weeks = max(1, (max(dates) - min(dates)).days / 7)

    weekly_mileage = total_distance / weeks
    runs_per_week = total_runs / weeks

    # Long runs (15+ miles)
    long_runs = [a for a in training if a['distance_miles'] >= 15]
    long_run_distance = max([a['distance_miles'] for a in long_runs], default=0)

    # Weekly breakdown
    weekly_distances = {}
    for a in training:
        week_num = a['date'].isocalendar()[1]
        weekly_distances[week_num] = weekly_distances.get(week_num, 0) + a['distance_miles']

    peak_weekly_mileage = max(weekly_distances.values()) if weekly_distances else 0

    # Consistency
    mileage_values = list(weekly_distances.values())
    if len(mileage_values) > 1:
        mileage_consistency = 1 - (statistics.stdev(mileage_values) / statistics.mean(mileage_values)) if statistics.mean(mileage_values) > 0 else 0
    else:
        mileage_consistency = 1.0

    # Quality workouts
    tempo_runs = [a for a in training if a['pace'] and 7.0 <= a['pace'] < 8.0]
    fast_runs = [a for a in training if a['pace'] and a['pace'] < 7.5]
    quality_runs = [a for a in training if a['pace'] and a['pace'] < 8.0]

    # HR features
    hr_activities = [a for a in training if a['avg_hr']]
    avg_hr = sum(a['avg_hr'] for a in hr_activities) / len(hr_activities) if hr_activities else 0

    # Populate features (matching the cache-based feature names)
    features['race_distance_miles'] = 26.2
    features['total_weekly_mileage'] = round(weekly_mileage, 2)
    features['peak_weekly_mileage'] = round(peak_weekly_mileage, 2)
    features['long_run_distance'] = round(long_run_distance, 2)
    features['long_run_count'] = len(long_runs)
    features['total_runs'] = total_runs
    features['runs_per_week'] = round(runs_per_week, 2)
    features['mileage_consistency'] = round(max(0, min(1, mileage_consistency)), 3)
    features['long_run_percent_weekly'] = round(long_run_distance / weekly_mileage * 100, 1) if weekly_mileage > 0 else 0
    features['training_consistency_score'] = features['mileage_consistency']

    # Quality workouts
    features['tempo_workout_count'] = len(tempo_runs)
    features['fast_workout_count'] = len(fast_runs)
    features['quality_workout_percent'] = round(100 * len(quality_runs) / total_runs, 1) if total_runs > 0 else 0

    # HR features
    features['avg_hr'] = avg_hr
    features['hr_at_easy_pace'] = avg_hr
    features['hr_at_marathon_pace'] = avg_hr

    # Runner context
    features['age_normalized'] = runner_context.get('age_normalized', 0.9)
    features['sex_encoded'] = runner_context.get('sex_encoded', 1)
    features['max_hr_normalized'] = runner_context.get('max_hr_normalized', 0.9)
    features['experience_years'] = runner_context.get('experience_years', 3)
    features['historical_pr_minutes'] = runner_context.get('historical_pr_minutes', 0)

    # Weather
    features['race_temperature'] = weather.get('temperature', 50)
    features['race_humidity'] = weather.get('humidity', 0.6)
    features['race_apparent_temperature'] = weather.get('apparent_temperature', 50)
    features['race_wind_speed'] = weather.get('wind_speed', 5)

    # Defaults for features that require track points (not available in CSV)
    features['elevation_tolerance'] = 1.0
    features['taper_quality_score'] = 0.5
    features['days_since_last_hard_effort'] = 7
    features['cardiac_drift'] = 0
    features['hr_zone_1_percent'] = 0
    features['hr_zone_2_percent'] = 0
    features['hr_zone_3_percent'] = 0
    features['hr_zone_4_percent'] = 0
    features['hr_zone_5_percent'] = 0
    features['avg_pace_variability'] = 0
    features['elevation_gain_per_mile'] = 0
    features['downhill_running_percent'] = 0

    return features


def main():
    print("=" * 80)
    print("COMPARISON: Cache-Based Features vs CSV-Only Features")
    print("=" * 80)

    # Load existing dataset (cache-based)
    script_dir = Path(__file__).parent
    with open(script_dir / 'race_data' / 'combined_41_features.json', 'r') as f:
        all_races = json.load(f)

    # Filter bonked races
    bonked_races = [
        ('my_runner', 'marathon_20251012'),
        ('my_runner', 'marathon_20231008'),
        ('runner_2', 'marathon_20231008'),
        ('runner_3', 'marathon_20231008'),
        ('runner_sara', 'sara_marathon_20240623'),
    ]
    clean_races = [r for r in all_races if (r['runner_id'], r['race_id']) not in bonked_races]

    print(f"\nTotal clean races: {len(clean_races)}")

    # Load CSV activities for each runner
    print("\nLoading CSV activities for each runner...")
    runner_activities = {}
    for runner_id, csv_path in RUNNER_CSV_PATHS.items():
        if csv_path.exists():
            activities = parse_strava_csv(csv_path)
            runner_activities[runner_id] = activities
            print(f"  {RUNNER_NAMES.get(runner_id, runner_id)}: {len(activities)} activities from CSV")
        else:
            print(f"  {RUNNER_NAMES.get(runner_id, runner_id)}: CSV not found at {csv_path}")

    # Extract CSV-only features for each race
    print("\nExtracting CSV-only features for each race...")
    csv_races = []

    for race in clean_races:
        runner_id = race['runner_id']
        if runner_id not in runner_activities:
            continue

        activities = runner_activities[runner_id]

        # Get runner context from existing race data
        existing_features = race.get('features', {})
        runner_context = {
            'age_normalized': existing_features.get('age_normalized', 0.9),
            'sex_encoded': existing_features.get('sex_encoded', 1),
            'max_hr_normalized': existing_features.get('max_hr_normalized', 0.9),
            'experience_years': existing_features.get('experience_years', 3),
            'historical_pr_minutes': existing_features.get('historical_pr_minutes', 0),
        }

        weather = {
            'temperature': existing_features.get('race_temperature', 50),
            'humidity': existing_features.get('race_humidity', 0.6),
            'apparent_temperature': existing_features.get('race_apparent_temperature', 50),
            'wind_speed': existing_features.get('race_wind_speed', 5),
        }

        csv_features = extract_csv_features(activities, race['race_date'], runner_context, weather)

        if csv_features:
            csv_races.append({
                'runner_id': runner_id,
                'race_id': race['race_id'],
                'race_date': race['race_date'],
                'race_name': race.get('race_name', 'Marathon'),
                'actual_time_minutes': race['actual_time_minutes'],
                'features': csv_features,
            })

    print(f"  Extracted features for {len(csv_races)} races")

    # Now run holdout validation for BOTH datasets
    print("\n" + "=" * 80)
    print("Running Holdout Validation on BOTH Feature Sets")
    print("=" * 80)

    # Get unique runners
    runner_ids = sorted(set(r['runner_id'] for r in clean_races))

    # Prepare holdout splits (same for both)
    def prepare_holdout_split(races):
        runners_data = {}
        for runner_id in runner_ids:
            runner_races = [r for r in races if r['runner_id'] == runner_id]
            runner_races.sort(key=lambda x: x['race_date'])
            runners_data[runner_id] = runner_races

        holdouts = {}
        training = []

        for runner_id, runner_races in runners_data.items():
            if len(runner_races) < 2:
                training.extend(runner_races)
                continue

            # Sara: hold out Boston and most recent
            if runner_id == 'runner_sara':
                boston_idx = None
                for i, r in enumerate(runner_races):
                    if 'Boston' in r.get('race_name', '') and '2025' in r['race_date']:
                        boston_idx = i
                        break

                if boston_idx is not None:
                    holdouts[f'{runner_id}_boston'] = runner_races[boston_idx]
                    holdouts[f'{runner_id}_recent'] = runner_races[-1]
                    for i, r in enumerate(runner_races):
                        if i != boston_idx and i != len(runner_races) - 1:
                            training.append(r)
                    continue

            # Default: hold out most recent
            holdouts[runner_id] = runner_races[-1]
            training.extend(runner_races[:-1])

        return training, holdouts

    # Cache-based holdout
    cache_training, cache_holdouts = prepare_holdout_split(clean_races)

    # CSV-based holdout
    csv_training, csv_holdouts = prepare_holdout_split(csv_races)

    # Get common feature names
    cache_feature_names = sorted(cache_training[0]['features'].keys())
    csv_feature_names = sorted(csv_training[0]['features'].keys()) if csv_training else []

    # Use intersection of features for fair comparison
    common_features = sorted(set(cache_feature_names) & set(csv_feature_names))
    print(f"\nCommon features for comparison: {len(common_features)}")

    def train_and_predict(training, holdouts, feature_names, label):
        X_train = []
        y_train = []

        for race in training:
            features = race.get('features', {})
            if not features:
                continue
            feature_values = [features.get(k, 0) or 0 for k in feature_names]
            X_train.append(feature_values)
            y_train.append(race['actual_time_minutes'])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()

        # Train on full training set
        model.fit(X_train, y_train)

        # Predict holdouts
        predictions = []
        for holdout_key, holdout in holdouts.items():
            if holdout_key.startswith('runner_sara'):
                name = "Sara (Boston)" if 'boston' in holdout_key else "Sara (Recent)"
            else:
                name = RUNNER_NAMES.get(holdout_key, holdout_key)

            features = [holdout['features'].get(k, 0) or 0 for k in feature_names]
            pred = model.predict([features])[0]
            actual = holdout['actual_time_minutes']

            predictions.append({
                'name': name,
                'predicted': pred,
                'actual': actual,
                'error': abs(pred - actual),
            })

        return cv_mae, predictions, model, feature_names

    # =====================================================================
    # FAIR COMPARISON: Use SAME features for both
    # =====================================================================
    print(f"\nCache features: {len(cache_feature_names)}")
    print(f"CSV features: {len(csv_feature_names)}")
    print(f"Common features: {len(common_features)}")

    print("\n" + "=" * 80)
    print("FAIR COMPARISON: Both using SAME {} common features".format(len(common_features)))
    print("=" * 80)

    # Train both models with COMMON features only
    print("\n" + "-" * 80)
    print("CACHE-BASED (using {} common features)".format(len(common_features)))
    print("-" * 80)
    cache_cv_mae, cache_predictions, cache_model, _ = train_and_predict(
        cache_training, cache_holdouts, common_features, "Cache"
    )

    print(f"CV MAE: {cache_cv_mae:.1f} min")
    print(f"\n{'Runner':<15} {'Predicted':<10} {'Actual':<10} {'Error':<10}")
    print("-" * 50)
    for p in cache_predictions:
        print(f"{p['name']:<15} {format_time(p['predicted']):<10} {format_time(p['actual']):<10} {p['error']:.1f} min")
    cache_avg_error = sum(p['error'] for p in cache_predictions) / len(cache_predictions)
    print(f"\nAverage Holdout Error: {cache_avg_error:.1f} min")

    print("\n" + "-" * 80)
    print("CSV-ONLY (using {} common features)".format(len(common_features)))
    print("-" * 80)
    csv_cv_mae, csv_predictions, csv_model, _ = train_and_predict(
        csv_training, csv_holdouts, common_features, "CSV"
    )

    print(f"CV MAE: {csv_cv_mae:.1f} min")
    print(f"\n{'Runner':<15} {'Predicted':<10} {'Actual':<10} {'Error':<10}")
    print("-" * 50)
    for p in csv_predictions:
        print(f"{p['name']:<15} {format_time(p['predicted']):<10} {format_time(p['actual']):<10} {p['error']:.1f} min")
    csv_avg_error = sum(p['error'] for p in csv_predictions) / len(csv_predictions)
    print(f"\nAverage Holdout Error: {csv_avg_error:.1f} min")

    # Side-by-side comparison
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Cache-Based':<15} {'CSV-Only':<15} {'Difference':<15}")
    print("-" * 75)
    print(f"{'CV MAE':<30} {cache_cv_mae:.1f} min{'':<7} {csv_cv_mae:.1f} min{'':<7} {csv_cv_mae - cache_cv_mae:+.1f} min")
    print(f"{'Holdout MAE':<30} {cache_avg_error:.1f} min{'':<7} {csv_avg_error:.1f} min{'':<7} {csv_avg_error - cache_avg_error:+.1f} min")

    print(f"\n{'Runner':<15} {'Cache Err':<12} {'CSV Err':<12} {'Diff':<10}")
    print("-" * 50)
    for cp, csvp in zip(cache_predictions, csv_predictions):
        diff = csvp['error'] - cp['error']
        print(f"{cp['name']:<15} {cp['error']:.1f} min{'':<5} {csvp['error']:.1f} min{'':<5} {diff:+.1f} min")

    # Feature importance comparison
    print("\n" + "=" * 80)
    print("TOP FEATURES COMPARISON (same features, different values)")
    print("=" * 80)

    print(f"\n{'Cache-Based Top 5':<40} {'CSV-Only Top 5':<40}")
    print("-" * 80)

    cache_imp = cache_model.feature_importances_
    cache_idx = np.argsort(cache_imp)[::-1][:5]

    csv_imp = csv_model.feature_importances_
    csv_idx = np.argsort(csv_imp)[::-1][:5]

    for i in range(5):
        cache_feat = f"{common_features[cache_idx[i]]}: {cache_imp[cache_idx[i]]*100:.1f}%"
        csv_feat = f"{common_features[csv_idx[i]]}: {csv_imp[csv_idx[i]]*100:.1f}%"
        print(f"{cache_feat:<40} {csv_feat:<40}")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if csv_avg_error <= cache_avg_error * 1.1:  # Within 10%
        print("\n CSV-only features perform comparably to cache-based features!")
        print("   The simpler CSV approach is sufficient for the Hugging Face app.")
    else:
        print(f"\n Cache-based features outperform CSV-only by {csv_avg_error - cache_avg_error:.1f} min.")
        print("   Consider adding FIT file parsing to improve accuracy.")


if __name__ == '__main__':
    main()
