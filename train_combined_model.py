#!/usr/bin/env python3
"""
Train model with combined data from both runners
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def load_runner2_data():
    """Load runner 2's data from CSV extraction"""
    with open('race_data/runner2_simple_dataset.json', 'r') as f:
        data = json.load(f)

    X = []
    y = []

    for race in data:
        features = race['features']
        feature_vector = [
            features['total_weekly_mileage'],
            features['peak_weekly_mileage'],
            features['long_run_distance'],
            features['total_runs'],
            features['runs_per_week'],
            features['avg_hr'],
            features['avg_pace'],
            features['num_long_runs']
        ]
        X.append(feature_vector)
        y.append(race['actual_time_minutes'])

    return np.array(X), np.array(y)


def load_your_data_simple():
    """Load your data with extracted features"""
    with open('race_data/your_simple_dataset.json', 'r') as f:
        data = json.load(f)

    X = []
    y = []

    for race in data:
        features = race['features']
        feature_vector = [
            features['total_weekly_mileage'],
            features['peak_weekly_mileage'],
            features['long_run_distance'],
            features['total_runs'],
            features['runs_per_week'],
            features['avg_hr'],
            features['avg_pace'],
            features['num_long_runs']
        ]
        X.append(feature_vector)
        y.append(race['actual_time_minutes'])

    return np.array(X), np.array(y)


def main():
    print("Loading datasets...")

    # Load runner 2's data
    X_r2, y_r2 = load_runner2_data()
    print(f"Runner 2: {len(X_r2)} marathons")

    # Load your data
    X_yours, y_yours = load_your_data_simple()
    print(f"You: {len(X_yours)} marathons")

    # Combine
    X = np.vstack([X_yours, X_r2])
    y = np.concatenate([y_yours, y_r2])

    print(f"\nCombined: {len(X)} total marathons")
    print(f"Time range: {y.min()//60}:{int(y.min()%60):02d} - {y.max()//60}:{int(y.max()%60):02d}")

    # Feature names
    feature_names = [
        'weekly_mileage', 'peak_mileage', 'long_run_dist',
        'total_runs', 'runs_per_week', 'avg_hr', 'avg_pace', 'num_long_runs'
    ]

    print(f"\nFeatures used ({len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")

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
        # Cross-validation
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

    # Train final model on all data
    print(f"\nTraining final {best_name} model on all data...")
    final_model = best_result['model']
    final_model.fit(X, y)

    # Save model
    import pickle
    with open('race_time_model_combined.pkl', 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_names': feature_names,
            'model_name': best_name
        }, f)

    print(f"✓ Saved model to race_time_model_combined.pkl")

    # Show feature importance if available
    if hasattr(final_model, 'feature_importances_'):
        print("\nFeature Importances:")
        importances = final_model.feature_importances_
        for i in np.argsort(importances)[::-1]:
            print(f"  {feature_names[i]}: {importances[i]:.3f}")


if __name__ == '__main__':
    main()
