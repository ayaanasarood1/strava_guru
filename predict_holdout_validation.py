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

def format_time(minutes):
    """Format minutes as H:MM"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}:{mins:02d}"

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
    bonked_ids = ['marathon_20251012', 'marathon_20231008', 'sara_marathon_20240623']  # Include Sara's 4:30 training run
    clean_races = [r for r in all_races if r['race_id'] not in bonked_ids]

    print(f"Clean races (excluding bonked): {len(clean_races)}")

    # Get unique runners
    runner_ids = sorted(set(r['runner_id'] for r in clean_races))
    runner_names = {
        'my_runner': 'Osman',
        'runner_2': 'Salman',
        'runner_3': 'Azeem',
        'runner_sara': 'Sara'
    }

    print(f"\nRunners: {len(runner_ids)}")

    # Separate by runner and sort by date
    runners_data = {}
    for runner_id in runner_ids:
        races = [r for r in clean_races if r['runner_id'] == runner_id]
        races.sort(key=lambda x: x['race_date'])
        runners_data[runner_id] = races
        name = runner_names.get(runner_id, runner_id)
        print(f"  {name}: {len(races)} races")

    # Identify holdout (most recent) for each runner
    holdouts = {}
    training_races = []

    print(f"\n{'='*80}")
    print("Holdout Races (will predict these):")
    print(f"{'='*80}")

    for runner_id, races in runners_data.items():
        if len(races) < 2:
            print(f"\n{runner_names.get(runner_id, runner_id)}: Skipping (only {len(races)} race)")
            training_races.extend(races)
            continue

        holdout = races[-1]
        holdouts[runner_id] = holdout
        training_races.extend(races[:-1])

        name = runner_names.get(runner_id, runner_id)
        print(f"\n{name}'s most recent:")
        print(f"  Date: {holdout['race_date'][:10]}")
        print(f"  Race: {holdout.get('race_name', 'Marathon')}")
        print(f"  Actual time: {format_time(holdout['actual_time_minutes'])}")

    print(f"\n{'='*80}")
    print(f"Training on {len(training_races)} races")
    print(f"Holding out {len(holdouts)} races for validation")
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

    # Train models with cross-validation
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
        print(f"\n{name}: MAE = {mae:.2f} min ({format_time(mae)})")

    # Best model
    best_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_model = results[best_name]['model']

    print(f"\n{'='*80}")
    print(f"Best Model: {best_name}")
    print(f"  CV MAE: {results[best_name]['mae']:.2f} minutes")
    print(f"{'='*80}")

    # Train best model on all training data
    best_model.fit(X_train, y_train)

    # Make predictions for each holdout
    print(f"\n{'='*80}")
    print("PREDICTIONS vs ACTUAL")
    print(f"{'='*80}")

    total_error = 0
    predictions = []

    for runner_id, holdout in holdouts.items():
        name = runner_names.get(runner_id, runner_id)
        features = [holdout['features'].get(k, 0) or 0 for k in feature_names]
        prediction = best_model.predict([features])[0]
        actual = holdout['actual_time_minutes']
        error = abs(prediction - actual)
        total_error += error

        predictions.append({
            'name': name,
            'date': holdout['race_date'][:10],
            'race_name': holdout.get('race_name', 'Marathon'),
            'predicted': prediction,
            'actual': actual,
            'error': error,
            'diff': prediction - actual
        })

        print(f"\n{name}'s Marathon ({holdout['race_date'][:10]}):")
        print(f"  Race: {holdout.get('race_name', 'Marathon')}")
        print(f"  Predicted: {format_time(prediction)} ({prediction:.1f} min)")
        print(f"  Actual:    {format_time(actual)} ({actual:.1f} min)")
        sign = '+' if prediction > actual else ''
        print(f"  Error:     {error:.1f} minutes ({sign}{prediction - actual:.1f})")

    # Summary
    avg_error = total_error / len(holdouts) if holdouts else 0

    print(f"\n{'='*80}")
    print("Validation Summary")
    print(f"{'='*80}")
    print(f"  Runners validated: {len(holdouts)}")
    print(f"  Average prediction error: {avg_error:.1f} minutes")
    print(f"  Model CV MAE on training: {results[best_name]['mae']:.1f} minutes")

    if avg_error < results[best_name]['mae'] * 1.5:
        print(f"\n  ✓ Model generalizes well! Holdout error within expected range.")
    else:
        print(f"\n  ⚠ Model may be overfitting. Holdout error higher than expected.")

    # Results table
    print(f"\n{'='*80}")
    print("Results Table")
    print(f"{'='*80}")
    print(f"\n{'Runner':<10} {'Race':<25} {'Predicted':<10} {'Actual':<10} {'Error':<10}")
    print("-" * 70)
    for p in predictions:
        race_short = p['race_name'][:23] if len(p['race_name']) > 23 else p['race_name']
        print(f"{p['name']:<10} {race_short:<25} {format_time(p['predicted']):<10} {format_time(p['actual']):<10} {p['error']:.1f} min")

    # Feature importance (if tree-based model)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n{'='*80}")
        print("Top 10 Most Important Features")
        print(f"{'='*80}")
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. {feature_names[idx]:<35}: {importances[idx]*100:.1f}%")

if __name__ == '__main__':
    main()
