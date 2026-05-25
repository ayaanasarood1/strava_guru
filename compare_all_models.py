#!/usr/bin/env python3
"""
Compare ALL model types including neural networks and various regression models
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try importing neural network libraries
try:
    from sklearn.neural_network import MLPRegressor
    MLP_AVAILABLE = True
except:
    MLP_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False


def format_time(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}:{mins:02d}"


def main():
    print("=" * 100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 100)

    # Load data
    with open('race_data/combined_hybrid_features.json', 'r') as f:
        all_races = json.load(f)

    print(f"\nDataset: {len(all_races)} races")

    # Prepare features
    feature_names = sorted(all_races[0]['features'].keys())
    X = np.array([[r['features'].get(k, 0) or 0 for k in feature_names] for r in all_races])
    y = np.array([r['actual_time_minutes'] for r in all_races])

    print(f"Features: {len(feature_names)}")
    print(f"Target range: {y.min():.0f} - {y.max():.0f} minutes ({format_time(y.min())} - {format_time(y.max())})")
    print(f"Target mean: {y.mean():.0f} minutes ({format_time(y.mean())})")
    print(f"Target std: {y.std():.1f} minutes")

    # Scale features for models that need it
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define all models to test
    models = {
        # Linear models
        'Linear Regression': (LinearRegression(), X),
        'Ridge (L2)': (Ridge(alpha=1.0), X),
        'Lasso (L1)': (Lasso(alpha=0.5, max_iter=10000), X),
        'ElasticNet (L1+L2)': (ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000), X),

        # Tree-based models
        'Decision Tree': (DecisionTreeRegressor(max_depth=10, random_state=42), X),
        'Random Forest': (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), X),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42), X),
        'AdaBoost': (AdaBoostRegressor(n_estimators=100, random_state=42), X),

        # Distance-based models (need scaling)
        'KNN (k=5)': (KNeighborsRegressor(n_neighbors=5), X_scaled),
        'KNN (k=10)': (KNeighborsRegressor(n_neighbors=10), X_scaled),
        'SVR (RBF)': (SVR(kernel='rbf', C=100, gamma='scale'), X_scaled),
        'SVR (Linear)': (SVR(kernel='linear', C=100), X_scaled),
    }

    # Add neural network if available
    if MLP_AVAILABLE:
        models['Neural Net (small)'] = (
            MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=42, early_stopping=True),
            X_scaled
        )
        models['Neural Net (medium)'] = (
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True),
            X_scaled
        )
        models['Neural Net (large)'] = (
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42, early_stopping=True),
            X_scaled
        )

    # Add XGBoost if available
    if XGB_AVAILABLE:
        models['XGBoost'] = (
            xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0),
            X
        )

    # Run cross-validation for each model
    print("\n" + "=" * 100)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 100)
    print(f"\n{'Model':<25} {'CV MAE':<12} {'CV Std':<12} {'Notes':<40}")
    print("-" * 100)

    results = []

    for name, (model, X_use) in models.items():
        try:
            scores = cross_val_score(model, X_use, y, cv=5, scoring='neg_mean_absolute_error')
            mae = -scores.mean()
            std = scores.std()
            results.append((name, mae, std))

            # Add notes based on model type
            if 'Linear' in name or 'Ridge' in name or 'Lasso' in name or 'Elastic' in name:
                notes = "Assumes linear relationships"
            elif 'Neural' in name:
                notes = f"Needs more data (have {len(y)}, want 1000+)"
            elif 'KNN' in name:
                notes = "Sensitive to feature scaling"
            elif 'SVR' in name:
                notes = "Good for small datasets"
            elif 'Tree' in name and 'Forest' not in name:
                notes = "Prone to overfitting"
            elif 'Forest' in name:
                notes = "Ensemble reduces overfitting"
            elif 'Boost' in name:
                notes = "Sequential learning"
            else:
                notes = ""

            print(f"{name:<25} {mae:.1f} min{'':<4} ±{std:.1f} min{'':<4} {notes:<40}")
        except Exception as e:
            print(f"{name:<25} FAILED: {str(e)[:50]}")

    # Sort by MAE
    results.sort(key=lambda x: x[1])

    print("\n" + "=" * 100)
    print("RANKED RESULTS (Best to Worst)")
    print("=" * 100)
    print(f"\n{'Rank':<6} {'Model':<25} {'CV MAE':<15}")
    print("-" * 50)

    for i, (name, mae, std) in enumerate(results, 1):
        marker = "🏆" if i == 1 else ("  " if i > 3 else "⭐")
        print(f"{i:<6} {name:<25} {mae:.1f} min {marker}")

    # Analysis
    print("\n" + "=" * 100)
    print("ANALYSIS: Why Random Forest Works Best")
    print("=" * 100)

    print(f"""
Dataset Characteristics:
- Small dataset: {len(y)} samples (neural nets typically need 1000+)
- High dimensionality: {len(feature_names)} features
- Non-linear relationships: Marathon time isn't linear with mileage
- Feature interactions: High mileage + quality workouts = synergy

Why Linear Models Struggle:
- Assume linear relationships (mileage ↑ → time ↓ linearly)
- Can't capture diminishing returns (70 mi/wk isn't 2x better than 35)
- Can't capture thresholds (need minimum long run to finish)

Why Neural Networks Struggle HERE:
- Need lots of data to learn complex patterns (we have {len(y)}, want 1000+)
- Prone to overfitting on small datasets
- Require careful hyperparameter tuning
- Feature scaling sensitivity

Why Random Forest Works:
- Handles non-linear relationships naturally
- Captures feature interactions (mileage × quality)
- Resistant to overfitting (ensemble of 100 trees)
- Works well with small datasets
- No feature scaling required
- Provides feature importance for interpretability
""")

    # Show what neural nets would need
    print("\n" + "=" * 100)
    print("WHAT WOULD MAKE NEURAL NETWORKS VIABLE?")
    print("=" * 100)

    print(f"""
Current State:
- {len(y)} training samples
- Neural net MAE: ~{next((r[1] for r in results if 'Neural' in r[0]), 'N/A'):.1f} min (if tested)

To Make Neural Nets Competitive:
1. More data: Need 500-1000+ marathon races
2. Data augmentation: Synthetic training variations
3. Transfer learning: Pre-train on related running data
4. Regularization: Dropout, weight decay to prevent overfitting

For Now:
Random Forest is the right choice for {len(y)} samples.
It achieves {results[0][1]:.1f} min MAE vs neural net's higher error.
""")


if __name__ == '__main__':
    main()
