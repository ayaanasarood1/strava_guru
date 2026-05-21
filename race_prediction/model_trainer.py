"""
Race Time Predictor - ML Model Training
Trains and evaluates models for race time prediction
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class RaceTimePredictor:
    """Train and evaluate race time prediction models"""

    def __init__(self, model_dir: Path = None):
        """Initialize predictor

        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir or Path.home() / ".strava_guru_cache" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = StandardScaler()
        self.feature_names = []
        self.models = {}
        self.best_model_name = None
        self.training_stats = {}

    def prepare_data(
        self,
        X: List[Dict],
        y: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training

        Handles missing values, converts to arrays, scales features

        Args:
            X: List of feature dictionaries
            y: List of target values (race times in minutes)

        Returns:
            Tuple of (X_scaled, y_array, feature_names)
        """
        if not X or not y:
            raise ValueError("No training data provided")

        print(f"\nPreparing data: {len(X)} samples")

        # Get feature names (sorted for consistency)
        self.feature_names = sorted(X[0].keys())
        print(f"  Features: {len(self.feature_names)}")

        # Convert to numpy array, handling None values
        X_array = np.zeros((len(X), len(self.feature_names)))

        for i, sample in enumerate(X):
            for j, feat_name in enumerate(self.feature_names):
                value = sample.get(feat_name)
                # Replace None with 0
                X_array[i, j] = value if value is not None else 0.0

        y_array = np.array(y)

        # Check for any remaining NaN or inf
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Feature statistics
        non_zero_counts = np.count_nonzero(X_array, axis=0)
        print(f"  Non-zero features: {np.sum(non_zero_counts > 0)} / {len(self.feature_names)}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        print(f"  ✓ Data scaled")

        return X_scaled, y_array, self.feature_names

    def train_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, Dict]:
        """Train multiple models and compare performance

        Args:
            X: Scaled feature matrix
            y: Target values
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary of model results
        """
        print(f"\n{'='*60}")
        print("Training Models")
        print(f"{'='*60}")

        results = {}

        # Define models to train
        models_to_train = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.5, max_iter=10000),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
        }

        if HAS_XGBOOST:
            models_to_train['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )

        # Cross-validation setup
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Train and evaluate each model
        for name, model in models_to_train.items():
            print(f"\n{name}:")

            # Cross-validation
            cv_scores = -cross_val_score(
                model,
                X, y,
                cv=kfold,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )

            print(f"  CV MAE: {cv_scores.mean():.2f} ± {cv_scores.std():.2f} minutes")

            # Train on full dataset
            model.fit(X, y)
            y_pred = model.predict(X)

            # Evaluate
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)

            print(f"  Train MAE: {mae:.2f} minutes")
            print(f"  Train RMSE: {rmse:.2f} minutes")
            print(f"  R²: {r2:.3f}")

            # Store results
            results[name] = {
                'model': model,
                'cv_mae': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_mae': mae,
                'train_rmse': rmse,
                'r2': r2
            }

            self.models[name] = model

        # Find best model (lowest CV MAE)
        self.best_model_name = min(results.keys(), key=lambda k: results[k]['cv_mae'])

        print(f"\n{'='*60}")
        print(f"Best Model: {self.best_model_name}")
        print(f"  CV MAE: {results[self.best_model_name]['cv_mae']:.2f} minutes")
        print(f"{'='*60}")

        self.training_stats = results
        return results

    def get_feature_importance(
        self,
        model_name: Optional[str] = None,
        top_n: int = 15
    ) -> List[Tuple[str, float]]:
        """Get feature importance from model

        Args:
            model_name: Model to analyze (default: best model)
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            return []

        model = self.models[model_name]

        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
        else:
            return []

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        feature_importance = [
            (self.feature_names[i], importances[i])
            for i in indices
        ]

        return feature_importance

    def predict(
        self,
        features: Dict[str, float],
        model_name: Optional[str] = None
    ) -> float:
        """Predict race time from features

        Args:
            features: Feature dictionary
            model_name: Model to use (default: best model)

        Returns:
            Predicted race time in minutes
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        # Convert features to array
        X = np.zeros((1, len(self.feature_names)))
        for i, feat_name in enumerate(self.feature_names):
            value = features.get(feat_name)
            X[0, i] = value if value is not None else 0.0

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        model = self.models[model_name]
        prediction = model.predict(X_scaled)[0]

        return prediction

    def save_model(self, filename: str = "race_predictor.pkl"):
        """Save trained model and scaler"""
        output_file = self.model_dir / filename

        model_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'models': self.models,
            'best_model_name': self.best_model_name,
            'training_stats': self.training_stats
        }

        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n✓ Saved model to {output_file}")
        return output_file

    def load_model(self, filename: str = "race_predictor.pkl"):
        """Load trained model and scaler"""
        input_file = self.model_dir / filename

        if not input_file.exists():
            raise FileNotFoundError(f"Model file not found: {input_file}")

        with open(input_file, 'rb') as f:
            model_data = pickle.load(f)

        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.models = model_data['models']
        self.best_model_name = model_data['best_model_name']
        self.training_stats = model_data.get('training_stats', {})

        print(f"✓ Loaded model from {input_file}")
        print(f"  Best model: {self.best_model_name}")
        print(f"  Features: {len(self.feature_names)}")

        return self

    def print_model_comparison(self):
        """Print comparison of all trained models"""
        if not self.training_stats:
            print("No training statistics available")
            return

        print(f"\n{'='*70}")
        print("Model Comparison")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'CV MAE':<15} {'Train MAE':<15} {'R²':<10}")
        print(f"{'-'*70}")

        for name, stats in sorted(
            self.training_stats.items(),
            key=lambda x: x[1]['cv_mae']
        ):
            cv_mae = stats['cv_mae']
            train_mae = stats['train_mae']
            r2 = stats['r2']

            marker = " ⭐" if name == self.best_model_name else ""

            print(f"{name:<20} {cv_mae:>6.2f} ± {stats['cv_std']:>4.2f}   "
                  f"{train_mae:>6.2f} min      {r2:>6.3f}{marker}")

        print(f"{'='*70}")

    def print_feature_importance(self, top_n: int = 15):
        """Print top feature importances"""
        importance = self.get_feature_importance(top_n=top_n)

        if not importance:
            print("Feature importance not available for this model")
            return

        print(f"\n{'='*60}")
        print(f"Top {top_n} Features ({self.best_model_name})")
        print(f"{'='*60}")

        max_importance = importance[0][1]

        for i, (feat_name, imp) in enumerate(importance, 1):
            # Normalize to percentage
            pct = (imp / max_importance) * 100 if max_importance > 0 else 0
            bar = '█' * int(pct / 5)

            print(f"{i:2d}. {feat_name:<30} {imp:>8.4f}  {bar}")

        print(f"{'='*60}")
