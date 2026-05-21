#!/usr/bin/env python3
"""
Holdout Validation: Predict most recent marathons
Train on all races except the last one for each runner
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from datetime import datetime

def main():
    print("="*80)
    print("Holdout Validation: Predicting Most Recent Marathons")
    print("="*80)

    # Load dataset (relative to script location)
    script_dir = Path(__file__).parent
    dataset_path = script_dir / 'race_data' / 'combined_41_features.json'
    with open(dataset_path, 'r') as f:
        all_races = json.load(f)

    print(f"\nLoaded {len(all_races)} total races")

    # Filter out bonked races
    bonked_ids = ['marathon_20251012', 'marathon_20231008']
    clean_races = [r for r in all_races if r['race_id'] not in bonked_ids]

    print(f"Clean races: {len(clean_races)}")

    # Separate by runner
    your_races = [r for r in clean_races if r['runner_id'] == 'my_runner']
    salman_races = [r for r in clean_races if r['runner_id'] == 'runner_2']

    # Sort by date
    your_races.sort(key=lambda x: x['race_date'])
    salman_races.sort(key=lambda x: x['race_date'])

    print(f"\nYour races: {len(your_races)}")
    print(f"Salman's races: {len(salman_races)}")

    # Identify most recent for each
    your_holdout = your_races[-1]
    salman_holdout = salman_races[-1]

    print(f"\n{'='*80}")
    print("Holdout Races (will predict these):")
    print(f"{'='*80}")
    print(f"\nYour most recent:")
    print(f"  Date: {your_holdout['race_date']}")
    print(f"  Actual time: {int(your_holdout['actual_time_minutes']//60)}:{int(your_holdout['actual_time_minutes']%60):02d}")

    print(f"\nSalman's most recent:")
    print(f"  Date: {salman_holdout['race_date']}")
    print(f"  Actual time: {int(salman_holdout['actual_time_minutes']//60)}:{int(salman_holdout['actual_time_minutes']%60):02d}")

    # Training set: everything except the holdouts
    training_races = your_races[:-1] + salman_races[:-1]

    print(f"\n{'='*80}")
    print(f"Training on {len(training_races)} races:")
    print(f"  Your training races: {len(your_races) - 1}")
    print(f"  Salman's training races: {len(salman_races) - 1}")
    print(f"{'='*80}")

    # Extract features for training
    X_train = []
    y_train = []
    feature_names = None

    for race in training_races:
        features = race.get('features', {})
        if not features:
            continue

        if feature_names is None:
            feature_names = sorted(features.keys())

        feature_values = [features.get(k, 0) or 0 for k in feature_names]
        X_train.append(feature_values)
        y_train.append(race['actual_time_minutes'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Features: {len(feature_names)}")

    # Train models with cross-validation on training set
    print(f"\n{'='*80}")
    print("Training Models (5-Fold CV on training set)")
    print(f"{'='*80}")

    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.5, max_iter=10000),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        results[name] = {'mae': mae, 'model': model}
        print(f"\n{name}: MAE = {mae:.2f} min ({mae//60}:{int(mae%60):02d})")

    # Best model
    best_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_model = results[best_name]['model']

    print(f"\n{'='*80}")
    print(f"Best Model: {best_name}")
    print(f"  CV MAE: {results[best_name]['mae']:.2f} minutes")
    print(f"{'='*80}")

    # Train best model on all training data
    best_model.fit(X_train, y_train)

    # Extract features for holdout races
    your_features = [your_holdout['features'].get(k, 0) or 0 for k in feature_names]
    salman_features = [salman_holdout['features'].get(k, 0) or 0 for k in feature_names]

    # Make predictions
    your_prediction = best_model.predict([your_features])[0]
    salman_prediction = best_model.predict([salman_features])[0]

    # Show results
    print(f"\n{'='*80}")
    print("PREDICTIONS vs ACTUAL")
    print(f"{'='*80}")

    print(f"\nYour Marathon ({your_holdout['race_date']}):")
    print(f"  Predicted: {int(your_prediction//60)}:{int(your_prediction%60):02d} ({your_prediction:.1f} min)")
    print(f"  Actual:    {int(your_holdout['actual_time_minutes']//60)}:{int(your_holdout['actual_time_minutes']%60):02d} ({your_holdout['actual_time_minutes']:.1f} min)")
    your_error = abs(your_prediction - your_holdout['actual_time_minutes'])
    print(f"  Error:     {your_error:.1f} minutes ({'+' if your_prediction > your_holdout['actual_time_minutes'] else ''}{your_prediction - your_holdout['actual_time_minutes']:.1f})")

    print(f"\nSalman's Marathon ({salman_holdout['race_date']}):")
    print(f"  Predicted: {int(salman_prediction//60)}:{int(salman_prediction%60):02d} ({salman_prediction:.1f} min)")
    print(f"  Actual:    {int(salman_holdout['actual_time_minutes']//60)}:{int(salman_holdout['actual_time_minutes']%60):02d} ({salman_holdout['actual_time_minutes']:.1f} min)")
    salman_error = abs(salman_prediction - salman_holdout['actual_time_minutes'])
    print(f"  Error:     {salman_error:.1f} minutes ({'+' if salman_prediction > salman_holdout['actual_time_minutes'] else ''}{salman_prediction - salman_holdout['actual_time_minutes']:.1f})")

    print(f"\n{'='*80}")
    print("Validation Summary")
    print(f"{'='*80}")
    print(f"  Average prediction error: {(your_error + salman_error) / 2:.1f} minutes")
    print(f"  Model CV MAE on training: {results[best_name]['mae']:.1f} minutes")
    print(f"  Actual holdout MAE: {(your_error + salman_error) / 2:.1f} minutes")

    if (your_error + salman_error) / 2 < results[best_name]['mae'] * 1.5:
        print(f"\n  ✓ Model generalizes well! Holdout error within expected range.")
    else:
        print(f"\n  ⚠ Model may be overfitting. Holdout error higher than expected.")

    # Show key training metrics for context
    print(f"\n{'='*80}")
    print("Training Context for Predictions")
    print(f"{'='*80}")

    # Your most recent training
    your_last_training = your_races[-2] if len(your_races) > 1 else your_races[0]
    print(f"\nYour previous marathon ({your_last_training['race_date']}):")
    print(f"  Time: {int(your_last_training['actual_time_minutes']//60)}:{int(your_last_training['actual_time_minutes']%60):02d}")
    your_features_dict = your_holdout['features']
    print(f"  Your recent training:")
    print(f"    Weekly mileage: {your_features_dict.get('total_weekly_mileage', 0):.1f} mi/week")
    print(f"    Zone 1%: {your_features_dict.get('zone1_percent', 0):.1f}%")
    print(f"    Quality workouts%: {your_features_dict.get('quality_workout_percent', 0):.1f}%")

    # Salman's most recent training
    salman_last_training = salman_races[-2] if len(salman_races) > 1 else salman_races[0]
    print(f"\nSalman's previous marathon ({salman_last_training['race_date']}):")
    print(f"  Time: {int(salman_last_training['actual_time_minutes']//60)}:{int(salman_last_training['actual_time_minutes']%60):02d}")
    salman_features_dict = salman_holdout['features']
    print(f"  Salman's recent training:")
    print(f"    Weekly mileage: {salman_features_dict.get('total_weekly_mileage', 0):.1f} mi/week")
    print(f"    Zone 1%: {salman_features_dict.get('zone1_percent', 0):.1f}%")
    print(f"    Quality workouts%: {salman_features_dict.get('quality_workout_percent', 0):.1f}%")

if __name__ == '__main__':
    main()
