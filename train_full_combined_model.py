#!/usr/bin/env python3
"""
Train model with full 41 features from both runners
"""

import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def load_dataset(filepath):
    """Load race dataset with features"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    X = []
    y = []

    for race in data:
        features = race.get('features', {})
        if not features:
            continue

        # Extract feature values in consistent order
        feature_values = [features.get(k, 0) or 0 for k in sorted(features.keys())]

        X.append(feature_values)
        y.append(race['actual_time_minutes'])

    # Get feature names from first race
    feature_names = sorted(data[0]['features'].keys()) if data else []

    return np.array(X), np.array(y), feature_names


def main():
    print("="*80)
    print("Training Model with Full 41 Features")
    print("="*80)

    # Load datasets
    print("\nLoading datasets...")
    salman_path = '/Users/osman/.strava_guru_cache/race_data/salman_full_features_dataset.json'

    X_salman, y_salman, feature_names = load_dataset(salman_path)
    print(f"  Salman: {len(X_salman)} marathons, {X_salman.shape[1]} features")

    # Note: User's dataset would need to be extracted with full features too
    # For now, we'll train just on Salman's data to test the 41-feature model

    X = X_salman
    y = y_salman

    print(f"\nTotal dataset: {len(X)} marathons")
    print(f"Features: {len(feature_names)}")
    print(f"Time range: {y.min()//60}:{int(y.min()%60):02d} - {y.max()//60}:{int(y.max()%60):02d}")

    # Show some key features
    print(f"\nKey features included:")
    key_features = [f for f in feature_names if any(x in f for x in ['lt_', 'weekly', 'zone', 'efficiency', 'terrain'])]
    for feat in key_features[:10]:
        print(f"  • {feat}")
    if len(key_features) > 10:
        print(f"  ... and {len(key_features) - 10} more")

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

        results[name] = {
            'mae': mae,
            'std': std,
            'model': model
        }

        print(f"\n{name}:")
        print(f"  MAE: {mae:.2f} minutes ({mae//60}:{int(mae%60):02d})")
        print(f"  Std: {std:.2f} minutes")

    # Find best model
    best_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_result = results[best_name]

    print("\n" + "="*80)
    print(f"Best Model: {best_name}")
    print(f"  MAE: {best_result['mae']:.2f} minutes ({best_result['mae']//60}:{int(best_result['mae']%60):02d})")
    print("="*80)

    # Train final model
    print(f"\nTraining final {best_name} model on all data...")
    final_model = best_result['model']
    final_model.fit(X, y)

    # Save model
    model_path = 'race_time_model_41_features.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_names': feature_names,
            'model_name': best_name
        }, f)

    print(f"✓ Saved model to {model_path}")

    # Show feature importance if available
    if hasattr(final_model, 'feature_importances_'):
        print("\nTop 10 Most Important Features:")
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.3f}")

if __name__ == '__main__':
    main()
