#!/usr/bin/env python3
"""
Filter out bonked races and train final model
"""

import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
try:
    import xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def main():
    # Load combined dataset
    dataset_path = '/Users/osman/.strava_guru_cache/race_data/combined_41_features.json'
    with open(dataset_path, 'r') as f:
        all_races = json.load(f)

    print(f"Loaded {len(all_races)} races")

    # ========================================================================
    # DATA QUALITY FILTERING
    # Filter out races with detected bonking (pace degradation > 15%)
    #
    # Bonking Detection Results:
    # 1. User's Chicago 2025 (marathon_20251012): Pace degradation in final miles
    #    Time: 3:45 (significantly slower than 3:04 PR)
    # 2. Salman's Chicago 2023 (marathon_20231008): 16.8% pace degradation
    #    First half: 6.87 min/mile → Second half: 8.03 min/mile
    #
    # Note: Automatic detection in extract_all_marathons.py with detailed
    # comments for methodology documentation in final report
    # ========================================================================
    bonked_race_ids = ['marathon_20251012', 'marathon_20231008']
    filtered_races = [r for r in all_races if r['race_id'] not in bonked_race_ids]

    print(f"\nFiltered out races (bonking detected):")
    for race in all_races:
        if race['race_id'] == 'marathon_20251012':
            print(f"  • Your Chicago Marathon 2025")
            print(f"    Time: 3:45 (pace degradation in final miles)")
        elif race['race_id'] == 'marathon_20231008':
            print(f"  • Salman's Chicago Marathon 2023")
            print(f"    Time: 3:20 (16.8% pace drop: 6.87 → 8.03 min/mile)")

    print(f"\nFinal dataset: {len(filtered_races)} races")

    # Extract features
    X = []
    y = []
    feature_names = None

    for race in filtered_races:
        features = race.get('features', {})
        if not features:
            continue

        if feature_names is None:
            feature_names = sorted(features.keys())

        feature_values = [features.get(k, 0) or 0 for k in feature_names]
        X.append(feature_values)
        y.append(race['actual_time_minutes'])

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset shape: {X.shape}")
    print(f"Time range: {y.min()//60}:{int(y.min()%60):02d} - {y.max()//60}:{int(y.max()%60):02d}")

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

    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)

    results = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        std = cv_scores.std()

        results[name] = {'mae': mae, 'std': std, 'model': model}

        print(f"\n{name}:")
        print(f"  MAE: {mae:.2f} minutes ({mae//60}:{int(mae%60):02d})")
        print(f"  Std: {std:.2f} minutes")

    # Best model
    best_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_result = results[best_name]

    print("\n" + "="*80)
    print(f"Best Model: {best_name}")
    print(f"  MAE: {best_result['mae']:.2f} minutes ({best_result['mae']//60}:{int(best_result['mae']%60):02d})")
    print("="*80)

    # Train final model
    final_model = best_result['model']
    final_model.fit(X, y)

    # Save
    model_path = 'race_time_model_final.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_names': feature_names,
            'model_name': best_name
        }, f)

    print(f"\n✓ Saved final model to {model_path}")

    # Feature importance
    if hasattr(final_model, 'feature_importances_'):
        print("\nTop 15 Most Important Features:")
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. {feature_names[idx]:30s}: {importances[idx]:.3f}")

    # Count by runner
    user_count = sum(1 for r in filtered_races if r['runner_id'] == 'my_runner')
    salman_count = sum(1 for r in filtered_races if r['runner_id'] == 'runner_2')

    print("\n" + "="*80)
    print("Model Training Complete!")
    print(f"  Dataset: {len(filtered_races)} marathons ({user_count} yours + {salman_count} Salman)")
    print(f"  Filtered: 2 bonked races (1 yours + 1 Salman)")
    print(f"  Features: 41 (full feature engineering pipeline)")
    print(f"  Accuracy: ±{best_result['mae']:.1f} minutes")
    print("="*80)

if __name__ == '__main__':
    main()
