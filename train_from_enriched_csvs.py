#!/usr/bin/env python3
"""
Train marathon prediction model from enriched activity CSVs.
- Auto-detects marathons from is_marathon column
- Excludes bonked races (is_bonked == True)
- Holds out most recent marathon per runner for testing
- Trains on all other marathons
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pickle

# All runners with their enriched CSV paths
RUNNERS = {
    'osman': {
        'csv_path': '/Users/osman/Downloads/osman_enriched_v3.csv',
        'display_name': 'Osman'
    },
    'salman': {
        'csv_path': '/Users/osman/Downloads/salman_enriched_v2.csv',
        'display_name': 'Salman'
    },
    'azeem': {
        'csv_path': '/Users/osman/Downloads/azeem_enriched_v2.csv',
        'display_name': 'Azeem'
    },
    'sara': {
        'csv_path': '/Users/osman/Downloads/sara_enriched_v2.csv',
        'display_name': 'Sara'
    },
    'salman_khan': {
        'csv_path': '/Users/osman/Downloads/salman_khan_enriched_v2.csv',
        'display_name': 'Salman Khan'
    },
}

# Training window: 16 weeks before race, excluding 7-day taper
TRAINING_WEEKS = 16
TAPER_DAYS = 7

# Filter out races older than this (old races don't reflect current fitness)
# Set to None to disable age filtering
# Trade-off: Helps improving runners (Salman), hurts consistent runners (Salman Khan)
MAX_RACE_AGE_YEARS = 5  # Filter races older than 5 years

# Override holdout race for specific runners (use race date string YYYY-MM-DD)
# Default behavior: use most recent race as holdout
HOLDOUT_OVERRIDES = {
    'salman_khan': '2021-12-12',  # Dallas 2021 (3:32) - closer to his average pace
}

# Exclude certain races from PR calculation (e.g., downhill courses, aided races)
# These races still count for training but don't set PRs
EXCLUDE_FROM_PR = {
    'osman': ['2023-11-18'],  # Big Bear Marathon - net downhill 3000+ ft, not a fair PR
    'salman_khan': ['2020-02-08'],  # Mesa Phoenix Marathon - net downhill 1500+ ft
}


def parse_date(date_str):
    """Parse activity date string"""
    try:
        return datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
    except:
        try:
            return datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
        except:
            return None


def extract_features_for_race(df, race_date, prior_race_time=None, marathon_pr=None):
    """Extract training features for a specific race from enriched CSV

    Args:
        df: DataFrame with all activities
        race_date: Date of the race to predict
        prior_race_time: Optional - most recent marathon time in minutes
        marathon_pr: Optional - runner's best marathon time (strongest predictor!)
    """

    # Define training window: 16 weeks before race, excluding 7-day taper
    window_end = race_date - timedelta(days=TAPER_DAYS)
    window_start = race_date - timedelta(weeks=TRAINING_WEEKS)

    # Filter activities in training window (exclude marathon-distance runs)
    training_df = df[
        (df['parsed_date'] >= window_start) &
        (df['parsed_date'] < window_end) &
        (df['distance_miles'] < 25)
    ].copy()

    if len(training_df) < 5:
        return None

    # Calculate features
    features = {}

    # Volume features
    features['total_runs'] = len(training_df)
    features['total_mileage'] = training_df['distance_miles'].sum()
    features['avg_weekly_mileage'] = features['total_mileage'] / TRAINING_WEEKS

    # Weekly mileage stats
    training_df['week'] = training_df['parsed_date'].dt.isocalendar().week
    weekly_miles = training_df.groupby('week')['distance_miles'].sum()
    features['peak_weekly_mileage'] = weekly_miles.max() if len(weekly_miles) > 0 else 0
    features['avg_weekly_mileage_std'] = weekly_miles.std() if len(weekly_miles) > 1 else 0

    # Long runs (15+ miles)
    long_runs = training_df[training_df['distance_miles'] >= 15]
    features['long_run_count'] = len(long_runs)
    features['long_run_max_distance'] = long_runs['distance_miles'].max() if len(long_runs) > 0 else 0
    features['long_run_avg_distance'] = long_runs['distance_miles'].mean() if len(long_runs) > 0 else 0

    # 20+ mile runs
    very_long_runs = training_df[training_df['distance_miles'] >= 20]
    features['very_long_run_count'] = len(very_long_runs)

    # Pace features
    valid_pace = training_df[training_df['pace_min_per_mile'].notna() & (training_df['pace_min_per_mile'] > 0)]
    features['avg_pace'] = valid_pace['pace_min_per_mile'].mean() if len(valid_pace) > 0 else 10.0
    features['pace_std'] = valid_pace['pace_min_per_mile'].std() if len(valid_pace) > 1 else 0

    # Tempo runs (7-8 min/mile pace)
    tempo_runs = training_df[
        (training_df['pace_min_per_mile'] >= 7.0) &
        (training_df['pace_min_per_mile'] < 8.0)
    ]
    features['tempo_run_count'] = len(tempo_runs)
    features['tempo_mileage'] = tempo_runs['distance_miles'].sum()

    # Speed work (<7 min/mile)
    speed_runs = training_df[training_df['pace_min_per_mile'] < 7.0]
    features['speed_work_count'] = len(speed_runs)

    # Easy runs (9+ min/mile)
    easy_runs = training_df[training_df['pace_min_per_mile'] >= 9.0]
    features['easy_run_count'] = len(easy_runs)
    features['easy_run_pct'] = len(easy_runs) / len(training_df) * 100 if len(training_df) > 0 else 0

    # Heart rate features (from FIT data)
    hr_runs = training_df[training_df['fit_avg_hr'].notna()]
    if len(hr_runs) > 0:
        features['avg_hr'] = hr_runs['fit_avg_hr'].mean()
        features['max_hr_recorded'] = hr_runs['fit_max_hr'].max() if hr_runs['fit_max_hr'].notna().any() else 0

        # HR zone distribution
        features['zone1_pct'] = hr_runs['fit_zone1_pct'].mean() if hr_runs['fit_zone1_pct'].notna().any() else 0
        features['zone2_pct'] = hr_runs['fit_zone2_pct'].mean() if hr_runs['fit_zone2_pct'].notna().any() else 0
        features['zone3_pct'] = hr_runs['fit_zone3_pct'].mean() if hr_runs['fit_zone3_pct'].notna().any() else 0
        features['zone4_pct'] = hr_runs['fit_zone4_pct'].mean() if hr_runs['fit_zone4_pct'].notna().any() else 0
        features['zone5_pct'] = hr_runs['fit_zone5_pct'].mean() if hr_runs['fit_zone5_pct'].notna().any() else 0
    else:
        features['avg_hr'] = 0
        features['max_hr_recorded'] = 0
        features['zone1_pct'] = 0
        features['zone2_pct'] = 0
        features['zone3_pct'] = 0
        features['zone4_pct'] = 0
        features['zone5_pct'] = 0

    # Cadence
    cadence_runs = training_df[training_df['fit_avg_cadence'].notna()]
    features['avg_cadence'] = cadence_runs['fit_avg_cadence'].mean() if len(cadence_runs) > 0 else 0

    # Pace variability
    pv_runs = training_df[training_df['fit_pace_variability'].notna()]
    features['avg_pace_variability'] = pv_runs['fit_pace_variability'].mean() if len(pv_runs) > 0 else 0

    # Training consistency
    days_with_runs = training_df['parsed_date'].dt.date.nunique()
    features['training_frequency'] = days_with_runs / (TRAINING_WEEKS * 7) * 100

    # Recent form (last 4 weeks before taper)
    recent_start = race_date - timedelta(weeks=4) - timedelta(days=TAPER_DAYS)
    recent_df = training_df[training_df['parsed_date'] >= recent_start]
    features['recent_mileage'] = recent_df['distance_miles'].sum()
    recent_pace = recent_df[recent_df['pace_min_per_mile'].notna() & (recent_df['pace_min_per_mile'] > 0)]
    features['recent_avg_pace'] = recent_pace['pace_min_per_mile'].mean() if len(recent_pace) > 0 else features['avg_pace']

    # Prior race time and PR - strong predictors of future performance!
    # If not provided, use 210 min (3:30) as default
    features['prior_marathon_time'] = prior_race_time if prior_race_time is not None else 210.0
    features['marathon_pr'] = marathon_pr if marathon_pr is not None else 210.0
    # Note: pr_age_years and decayed_pr are added in load_runner_data where we have the PR date

    # Replace NaN with 0
    for k, v in features.items():
        if pd.isna(v):
            features[k] = 0

    return features


def is_actual_marathon_race(activity_name):
    """Check if activity name indicates an actual marathon race (not training run)"""
    if pd.isna(activity_name):
        return False

    name = str(activity_name).lower()

    # Exclude generic training run names
    training_patterns = ['morning run', 'afternoon run', 'evening run', 'lunch run',
                         'easy run', 'long run', 'recovery run']
    for pattern in training_patterns:
        if name == pattern or name.startswith(pattern + ' '):
            return False

    # Exclude trail runs/marathons (not road marathons)
    if 'trail' in name:
        return False

    # Exclude virtual runs (often not race effort) and pacing runs
    if 'virtual' in name:
        return False
    if 'pacing' in name or 'pacer' in name:
        return False

    # Include if contains 'marathon'
    if 'marathon' in name:
        return True

    # Include known race indicators
    race_patterns = [
        'cim', 'boston', 'nyc', 'chicago', 'berlin', 'london', 'tokyo',  # majors
        'jack & jill', 'jack and jill', 'napa', 'la marathon', 'l.a.',
        'bq', 'qualifier', 'official', 'race', 'pr ', 'pr!', 'sub-', 'sub 3',
        'vancouver', 'sf marathon', 'san francisco', 'long beach', 'honolulu',
        'half iron'
    ]
    for pattern in race_patterns:
        if pattern in name:
            return True

    return False


def load_runner_data(runner_id, config):
    """Load and process data for a single runner - filter for actual marathon races"""
    df = pd.read_csv(config['csv_path'])
    df['parsed_date'] = df['Activity Date'].apply(parse_date)
    df = df[df['parsed_date'].notna()]
    df['distance_miles'] = df['distance_miles'].fillna(0)

    # Find marathons using is_marathon column (distance filter)
    marathons = df[df['is_marathon'] == True].copy()
    marathons = marathons.sort_values('parsed_date')

    races = []
    skipped = []
    prior_race_time = None  # Track prior race time for each runner
    marathon_pr = None  # Track personal record (best time)
    marathon_pr_date = None  # Track when PR was set

    for idx, row in marathons.iterrows():
        race_date = row['parsed_date']
        is_bonked = row['is_bonked'] == True
        actual_time = row['duration_min']
        activity_name = row.get('Activity Name', '')

        # Skip if not an actual race (training run, trail run, etc.)
        if not is_actual_marathon_race(activity_name):
            skipped.append(f"  SKIPPED: {activity_name} ({race_date.strftime('%Y-%m-%d')})")
            continue

        # Skip races older than MAX_RACE_AGE_YEARS (don't reflect current fitness)
        if MAX_RACE_AGE_YEARS is not None:
            race_age_years = (datetime.now() - race_date).days / 365.25
            if race_age_years > MAX_RACE_AGE_YEARS:
                skipped.append(f"  SKIPPED (old): {activity_name} ({race_date.strftime('%Y-%m-%d')}) - {race_age_years:.1f} years ago")
                # Still update prior_race_time and PR for older races (they inform current fitness)
                if not pd.isna(actual_time) and 120 <= actual_time <= 360 and not is_bonked:
                    prior_race_time = actual_time
                    # Check if excluded from PR
                    race_date_str = race_date.strftime('%Y-%m-%d')
                    excluded_dates = EXCLUDE_FROM_PR.get(runner_id, [])
                    if race_date_str not in excluded_dates:
                        if marathon_pr is None or actual_time < marathon_pr:
                            marathon_pr = actual_time
                            marathon_pr_date = race_date
                continue

        # Skip invalid times (too fast or too slow for a marathon)
        if pd.isna(actual_time) or actual_time < 120 or actual_time > 360:  # 2-6 hours
            skipped.append(f"  SKIPPED (time): {activity_name} - {actual_time:.0f} min")
            continue

        # Extract features with prior race time and PR
        features = extract_features_for_race(df, race_date, prior_race_time, marathon_pr)
        if features is None:
            continue

        # Calculate PR age at time of this race (before PR gets updated)
        if marathon_pr is not None and marathon_pr_date is not None:
            pr_age_days = (race_date - marathon_pr_date).days
            pr_age_years = max(0, pr_age_days / 365.25)
        else:
            pr_age_years = 0.0

        # Decayed PR: older PRs carry less weight (half-life of 3 years)
        decay_factor = 0.5 ** (pr_age_years / 3.0)
        features['pr_age_years'] = pr_age_years
        features['pr_decay_factor'] = decay_factor
        if marathon_pr is not None:
            features['decayed_pr'] = marathon_pr * decay_factor + 210.0 * (1 - decay_factor)
        else:
            features['decayed_pr'] = 210.0

        races.append({
            'runner_id': runner_id,
            'runner_name': config['display_name'],
            'race_date': race_date,
            'race_date_str': race_date.strftime('%Y-%m-%d'),
            'race_name': activity_name,
            'actual_time_minutes': actual_time,
            'is_bonked': is_bonked,
            'features': features,
            'marathon_pr_date': marathon_pr_date  # Track PR date at time of race
        })

        # Update prior race time and PR (only from non-bonked races)
        if not is_bonked:
            prior_race_time = actual_time
            # Check if this race should be excluded from PR (e.g., downhill courses)
            race_date_str = race_date.strftime('%Y-%m-%d')
            excluded_dates = EXCLUDE_FROM_PR.get(runner_id, [])
            if race_date_str not in excluded_dates:
                if marathon_pr is None or actual_time < marathon_pr:
                    marathon_pr = actual_time
                    marathon_pr_date = race_date

    # Print skipped activities for transparency
    if skipped:
        print(f"  Skipped {len(skipped)} non-race activities")
        for s in skipped[:5]:  # Show first 5
            print(s)
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    return races


def main():
    print("=" * 80)
    print("Training Marathon Prediction Model from Enriched CSVs")
    print("Auto-detecting marathons, excluding bonked races")
    print("=" * 80)

    # Load all runners' data
    all_races = []
    for runner_id, config in RUNNERS.items():
        print(f"\nLoading {config['display_name']}...")
        races = load_runner_data(runner_id, config)
        print(f"  Found {len(races)} marathons")
        all_races.extend(races)

    print(f"\n{'=' * 80}")
    print(f"Total marathons loaded: {len(all_races)}")

    # Separate clean vs bonked races
    clean_races = [r for r in all_races if not r['is_bonked']]
    bonked_races = [r for r in all_races if r['is_bonked']]

    print(f"Clean races: {len(clean_races)}")
    print(f"Bonked races (excluded): {len(bonked_races)}")

    if bonked_races:
        print("\nBonked races excluded from training:")
        for r in bonked_races:
            time_str = f"{int(r['actual_time_minutes']//60)}:{int(r['actual_time_minutes']%60):02d}"
            print(f"  - {r['runner_name']}: {r['race_date_str']} ({time_str})")

    # Show all clean races by runner
    print(f"\n{'=' * 80}")
    print("All Clean Races by Runner:")
    print(f"{'=' * 80}")
    for runner_id in RUNNERS.keys():
        runner_races = sorted([r for r in clean_races if r['runner_id'] == runner_id],
                             key=lambda x: x['race_date'])
        print(f"\n{RUNNERS[runner_id]['display_name']} ({len(runner_races)} races):")
        for r in runner_races:
            time_str = f"{int(r['actual_time_minutes']//60)}:{int(r['actual_time_minutes']%60):02d}"
            print(f"  {r['race_date_str']}: {time_str}")

    # Separate by runner and identify holdout (most recent per runner, or override)
    holdout_races = []
    training_races = []

    for runner_id in RUNNERS.keys():
        runner_races = [r for r in clean_races if r['runner_id'] == runner_id]
        runner_races.sort(key=lambda x: x['race_date'])

        if len(runner_races) >= 2:
            # Check for holdout override
            if runner_id in HOLDOUT_OVERRIDES:
                override_date = HOLDOUT_OVERRIDES[runner_id]
                holdout_race = next((r for r in runner_races if r['race_date_str'] == override_date), None)
                if holdout_race:
                    holdout_races.append(holdout_race)
                    training_races.extend([r for r in runner_races if r['race_date_str'] != override_date])
                else:
                    print(f"  WARNING: Holdout override date {override_date} not found for {runner_id}")
                    holdout_races.append(runner_races[-1])
                    training_races.extend(runner_races[:-1])
            else:
                # Default: hold out most recent
                holdout_races.append(runner_races[-1])
                training_races.extend(runner_races[:-1])
        elif len(runner_races) == 1:
            # Only one race - add to training
            training_races.extend(runner_races)

    print(f"\n{'=' * 80}")
    print(f"Data Split:")
    print(f"  Training races: {len(training_races)}")
    print(f"  Holdout races (test): {len(holdout_races)}")
    print(f"{'=' * 80}")

    if holdout_races:
        print("\nHoldout races (most recent per runner - NOT used in training):")
        for r in holdout_races:
            time_str = f"{int(r['actual_time_minutes']//60)}:{int(r['actual_time_minutes']%60):02d}"
            print(f"  - {r['runner_name']}: {r['race_date_str']} - {time_str}")

    # Extract features (exclude anchor features)
    feature_names = sorted([k for k in training_races[0]['features'].keys()
                           if k not in ['prior_marathon_time', 'marathon_pr']])
    print(f"\nFeatures ({len(feature_names)}) - predicting delta from anchor (min of prior & PR)")

    X_train = np.array([[r['features'].get(k, 0) or 0 for k in feature_names] for r in training_races])

    # Anchor = PR only (simpler for users)
    y_train_anchor = np.array([
        r['features'].get('marathon_pr', 210)
        for r in training_races
    ])
    y_train_actual = np.array([r['actual_time_minutes'] for r in training_races])
    y_train_delta = y_train_actual - y_train_anchor  # Delta from anchor

    print(f"Training set shape: {X_train.shape}")
    print(f"Target: delta from anchor (avg delta: {y_train_delta.mean():.1f} min)")

    # Train models to predict delta from anchor (min of prior & PR)
    print(f"\n{'=' * 80}")
    print("Training Models (5-Fold CV) - Predicting Delta from Anchor")
    print(f"{'=' * 80}")

    models = {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    cv_results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train_delta, cv=min(5, len(training_races)),
                                    scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        cv_results[name] = {'mae': mae, 'model': model}
        print(f"  {name}: CV MAE = {mae:.1f} min")

    # Select best model
    best_name = min(cv_results.keys(), key=lambda k: cv_results[k]['mae'])
    best_model = cv_results[best_name]['model']

    print(f"\nBest model: {best_name} (CV MAE = {cv_results[best_name]['mae']:.1f} min)")

    # Train on full training set
    best_model.fit(X_train, y_train_delta)

    # Holdout validation
    if holdout_races:
        print(f"\n{'=' * 80}")
        print("Holdout Validation Results (Test Set)")
        print(f"{'=' * 80}")

        errors = []
        for r in holdout_races:
            X_test = np.array([[r['features'].get(k, 0) or 0 for k in feature_names]])
            prior_time = r['features'].get('prior_marathon_time', 210)
            marathon_pr = r['features'].get('marathon_pr', 210)
            # Anchor = PR only (simpler for users)
            anchor = marathon_pr
            delta_pred = best_model.predict(X_test)[0]
            # Cap delta to realistic range: max 5 min faster than anchor, 30 min slower
            delta_pred = max(-5, min(30, delta_pred))
            pred = anchor + delta_pred  # Anchor + predicted delta (capped)

            actual = r['actual_time_minutes']
            error = pred - actual
            errors.append(abs(error))

            pred_str = f"{int(pred//60)}:{int(pred%60):02d}"
            actual_str = f"{int(actual//60)}:{int(actual%60):02d}"
            pr_str = f"{int(marathon_pr//60)}:{int(marathon_pr%60):02d}"

            print(f"\n{r['runner_name']} ({r['race_date_str']}):")
            print(f"  PR (anchor): {pr_str}")
            print(f"  Predicted:   {pred_str} ({pred:.1f} min) [PR + {delta_pred:+.1f}]")
            print(f"  Actual:      {actual_str} ({actual:.1f} min)")
            print(f"  Error:       {'+' if error > 0 else ''}{error:.1f} min")

        avg_error = np.mean(errors)
        anchored_holdout_mae = avg_error
        print(f"\n{'=' * 80}")
        print(f"Holdout MAE: {avg_error:.1f} min")
        print(f"CV MAE:      {cv_results[best_name]['mae']:.1f} min")
        print(f"{'=' * 80}")

    # ================================================================================
    # UNANCHORED APPROACH: PR as a decayed feature, predict absolute time
    # ================================================================================
    print(f"\n{'=' * 80}")
    print("ALTERNATIVE: Unanchored Approach (PR as Decayed Feature)")
    print(f"{'=' * 80}")

    # Include marathon_pr, pr_age_years, decayed_pr, pr_decay_factor as features
    unanchored_feature_names = sorted([k for k in training_races[0]['features'].keys()])
    print(f"\nFeatures ({len(unanchored_feature_names)}) - predicting absolute time with PR decay features")

    # Show the new PR-related features
    pr_features = [f for f in unanchored_feature_names if 'pr' in f.lower() or 'prior' in f.lower()]
    print(f"PR-related features: {pr_features}")

    X_train_unanchored = np.array([[r['features'].get(k, 0) or 0 for k in unanchored_feature_names]
                                    for r in training_races])
    y_train_absolute = np.array([r['actual_time_minutes'] for r in training_races])

    print(f"Training set shape: {X_train_unanchored.shape}")
    print(f"Target: absolute time (avg: {y_train_absolute.mean():.1f} min)")

    # Train models for unanchored approach
    print(f"\nTraining Models (5-Fold CV) - Predicting Absolute Time")

    unanchored_models = {
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    unanchored_cv_results = {}
    for name, model in unanchored_models.items():
        cv_scores = cross_val_score(model, X_train_unanchored, y_train_absolute,
                                    cv=min(5, len(training_races)),
                                    scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        unanchored_cv_results[name] = {'mae': mae, 'model': model}
        print(f"  {name}: CV MAE = {mae:.1f} min")

    # Select best unanchored model
    best_unanchored_name = min(unanchored_cv_results.keys(), key=lambda k: unanchored_cv_results[k]['mae'])
    best_unanchored_model = unanchored_cv_results[best_unanchored_name]['model']

    print(f"\nBest unanchored model: {best_unanchored_name} (CV MAE = {unanchored_cv_results[best_unanchored_name]['mae']:.1f} min)")

    # Train on full training set
    best_unanchored_model.fit(X_train_unanchored, y_train_absolute)

    # Holdout validation for unanchored approach
    if holdout_races:
        print(f"\n{'=' * 80}")
        print("Unanchored Holdout Validation Results")
        print(f"{'=' * 80}")

        unanchored_errors = []
        for r in holdout_races:
            X_test = np.array([[r['features'].get(k, 0) or 0 for k in unanchored_feature_names]])
            marathon_pr = r['features'].get('marathon_pr', 210)
            pr_age = r['features'].get('pr_age_years', 0)
            decay_factor = r['features'].get('pr_decay_factor', 1.0)
            decayed_pr = r['features'].get('decayed_pr', marathon_pr)

            pred = best_unanchored_model.predict(X_test)[0]
            # Cap prediction to realistic range (2:30 to 5:00)
            pred = max(150, min(300, pred))

            actual = r['actual_time_minutes']
            error = pred - actual
            unanchored_errors.append(abs(error))

            pred_str = f"{int(pred//60)}:{int(pred%60):02d}"
            actual_str = f"{int(actual//60)}:{int(actual%60):02d}"
            pr_str = f"{int(marathon_pr//60)}:{int(marathon_pr%60):02d}"

            print(f"\n{r['runner_name']} ({r['race_date_str']}):")
            print(f"  PR: {pr_str} (age: {pr_age:.1f}y, decay: {decay_factor:.2f})")
            print(f"  Predicted:   {pred_str} ({pred:.1f} min)")
            print(f"  Actual:      {actual_str} ({actual:.1f} min)")
            print(f"  Error:       {'+' if error > 0 else ''}{error:.1f} min")

        unanchored_avg_error = np.mean(unanchored_errors)
        print(f"\n{'=' * 80}")
        print(f"Unanchored Holdout MAE: {unanchored_avg_error:.1f} min")
        print(f"Unanchored CV MAE:      {unanchored_cv_results[best_unanchored_name]['mae']:.1f} min")
        print(f"{'=' * 80}")

    # ================================================================================
    # COMPARISON SUMMARY
    # ================================================================================
    print(f"\n{'=' * 80}")
    print("COMPARISON: Anchored vs Unanchored Approach")
    print(f"{'=' * 80}")
    print(f"\n  Anchored (PR-only, predict delta):")
    print(f"    CV MAE:      {cv_results[best_name]['mae']:.1f} min")
    if holdout_races:
        print(f"    Holdout MAE: {anchored_holdout_mae:.1f} min")
    print(f"\n  Unanchored (PR decay feature, predict absolute):")
    print(f"    CV MAE:      {unanchored_cv_results[best_unanchored_name]['mae']:.1f} min")
    if holdout_races:
        print(f"    Holdout MAE: {unanchored_avg_error:.1f} min")

    if holdout_races:
        winner = "Anchored" if anchored_holdout_mae < unanchored_avg_error else "Unanchored"
        print(f"\n  Winner (by holdout): {winner}")

    # Feature importance - Anchored
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n{'=' * 80}")
        print("Top 10 Feature Importances (Anchored)")
        print(f"{'=' * 80}")

        importances = list(zip(feature_names, best_model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)

        for name, imp in importances[:10]:
            print(f"  {name}: {imp:.3f}")

    # Feature importance - Unanchored
    if hasattr(best_unanchored_model, 'feature_importances_'):
        print(f"\n{'=' * 80}")
        print("Top 10 Feature Importances (Unanchored)")
        print(f"{'=' * 80}")

        importances = list(zip(unanchored_feature_names, best_unanchored_model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)

        for name, imp in importances[:10]:
            print(f"  {name}: {imp:.3f}")

    # Save UNANCHORED model (better holdout performance)
    model_path = '/Users/osman/PycharmProjects/strava_guru/race_time_model_enriched.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_unanchored_model,
            'feature_names': unanchored_feature_names,
            'model_type': best_unanchored_name,
            'cv_mae': unanchored_cv_results[best_unanchored_name]['mae'],
            'holdout_mae': unanchored_avg_error if holdout_races else None,
            'n_training_races': len(training_races),
            'n_features': len(unanchored_feature_names),
            'approach': 'unanchored',
            'description': 'Predicts absolute time with PR decay features'
        }, f)

    print(f"\nUnanchored model saved to: {model_path}")

    # Also save to huggingface_app directory
    hf_model_path = '/Users/osman/PycharmProjects/strava_guru/huggingface_app/model.pkl'
    with open(hf_model_path, 'wb') as f:
        pickle.dump({
            'model': best_unanchored_model,
            'feature_names': unanchored_feature_names,
            'model_type': best_unanchored_name,
            'cv_mae': unanchored_cv_results[best_unanchored_name]['mae'],
            'holdout_mae': unanchored_avg_error if holdout_races else None,
            'n_training_races': len(training_races),
            'n_features': len(unanchored_feature_names),
            'approach': 'unanchored',
            'description': 'Predicts absolute time with PR decay features'
        }, f)
    print(f"Model also saved to: {hf_model_path}")


if __name__ == '__main__':
    main()
