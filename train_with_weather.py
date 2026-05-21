#!/usr/bin/env python3
"""
Train model with weather features included
Then predict with different temperature scenarios
"""

import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

def main():
    print("="*80)
    print("Training Model with Weather Features")
    print("="*80)

    # Load dataset
    dataset_path = '/Users/osman/.strava_guru_cache/race_data/combined_41_features.json'
    with open(dataset_path, 'r') as f:
        all_races = json.load(f)

    print(f"\nLoaded {len(all_races)} races")

    # Filter out bonked races
    bonked_ids = ['marathon_20251012', 'marathon_20231008']
    clean_races = [r for r in all_races if r['race_id'] not in bonked_ids]

    # Only use races with weather data for training
    races_with_weather = [r for r in clean_races
                          if r['features'].get('race_temperature') is not None]

    print(f"Clean races: {len(clean_races)}")
    print(f"Races with weather data: {len(races_with_weather)}")

    # Extract features
    X = []
    y = []
    feature_names = None

    for race in races_with_weather:
        features = race.get('features', {})
        if not features:
            continue

        if feature_names is None:
            feature_names = sorted(features.keys())

        # Replace None values with 0
        feature_values = [features.get(k, 0) or 0 for k in feature_names]
        X.append(feature_values)
        y.append(race['actual_time_minutes'])

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {len(feature_names)}")

    # Check if weather features are included
    weather_features = [f for f in feature_names if 'race_' in f and any(x in f for x in ['temperature', 'humidity', 'wind'])]
    print(f"Weather features: {weather_features}")

    # Train models
    print("\n" + "="*80)
    print("Training Models with 5-Fold Cross-Validation")
    print("="*80)

    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.5, max_iter=10000),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        std = cv_scores.std()

        results[name] = {'mae': mae, 'std': std, 'model': model}

        print(f"\n{name}:")
        print(f"  MAE: {mae:.2f} minutes ({mae//60:.0f}:{int(mae%60):02d})")
        print(f"  Std: {std:.2f} minutes")

    # Best model
    best_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_result = results[best_name]

    print("\n" + "="*80)
    print(f"Best Model: {best_name}")
    print(f"  MAE: {best_result['mae']:.2f} minutes")
    print("="*80)

    # Train final model
    final_model = best_result['model']
    final_model.fit(X, y)

    # Save model with weather features
    model_path = 'race_time_model_with_weather.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_names': feature_names,
            'model_name': best_name
        }, f)

    print(f"\n✓ Saved model to {model_path}")

    # Feature importance
    if hasattr(final_model, 'feature_importances_'):
        print("\nTop 20 Most Important Features:")
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. {feature_names[idx]:35s}: {importances[idx]:.3f}")

    # Count by runner
    user_count = len([r for r in races_with_weather if r['runner_id'] == 'my_runner'])
    salman_count = len([r for r in races_with_weather if r['runner_id'] == 'runner_2'])
    azeem_count = len([r for r in races_with_weather if r['runner_id'] == 'runner_3'])

    print("\n" + "="*80)
    print("Model Training Complete!")
    print(f"  Dataset: {len(races_with_weather)} marathons with weather data")
    print(f"    • You: {user_count} marathons")
    print(f"    • Salman: {salman_count} marathons")
    print(f"    • Azeem: {azeem_count} marathons")
    print(f"  Features: {len(feature_names)} (including weather)")
    print(f"  Accuracy: ±{best_result['mae']:.1f} minutes")
    print("="*80)

if __name__ == '__main__':
    main()
