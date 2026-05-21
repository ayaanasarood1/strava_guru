# Race Time Prediction - ML Pipeline

Machine learning pipeline for predicting race times from training features.

## Overview

This ML pipeline trains models to predict race finish times based on training characteristics extracted from activity data. It uses the feature engineering pipeline to extract 41 features from training data and trains multiple models to find the best predictor.

## Quick Start

### 1. Create Race Results Dataset

Create a JSON file with historical race results:

```json
[
  {
    "race_id": "boston_2024",
    "runner_id": "runner1",
    "race_date": "2024-04-15",
    "race_distance_miles": 26.2,
    "actual_time_minutes": 195.5,
    "age": 35,
    "sex": "M",
    "max_hr": 185,
    "experience_years": 8
  }
]
```

Or generate synthetic data for testing:

```bash
python create_sample_races.py
```

### 2. Train Model

```bash
python train_race_model.py your_race_results.json
```

This will:
- Extract features for each race
- Train multiple models (Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost)
- Cross-validate and compare performance
- Save the best model

### 3. Make Predictions

```bash
# Predict marathon time
python predict_race_time.py 26.2

# Predict half marathon time
python predict_race_time.py 13.1

# Predict 10K time
python predict_race_time.py 6.2
```

## Architecture

### Components

1. **`RaceDataCollector`** - Collects race results and extracts features
2. **`RaceTimePredictor`** - Trains and evaluates ML models
3. **`predict_race_time()`** - High-level prediction interface

### Models Trained

The pipeline trains and compares multiple models:

- **Ridge Regression** - Linear model with L2 regularization
- **Lasso Regression** - Linear model with L1 regularization (feature selection)
- **Random Forest** - Ensemble of decision trees
- **Gradient Boosting** - Boosted tree ensemble
- **XGBoost** - Advanced gradient boosting (if installed)

The best model is selected based on cross-validation MAE (Mean Absolute Error).

### Feature Preprocessing

- Missing values (None) replaced with 0
- All features standardized using StandardScaler
- Handles 41 features from feature engineering pipeline

## Workflow

### Training Pipeline

```
Race Results (JSON)
    ↓
Extract Features (41 features per race)
    ↓
Prepare Data (handle None, scale)
    ↓
Train Models (5-fold cross-validation)
    ↓
Compare & Select Best Model
    ↓
Save Model (pickle)
```

### Prediction Pipeline

```
Race Parameters + Runner Profile
    ↓
Extract Current Training Features
    ↓
Load Trained Model
    ↓
Predict Race Time
    ↓
Format & Display Results
```

## Usage Examples

### Programmatic Usage

```python
from datetime import datetime
from activity_cache import ActivityCache
from feature_engineering import RunnerContext
from race_prediction import RaceDataCollector, RaceTimePredictor
from race_prediction.predictor import predict_race_time

# === Training ===

# Collect data
cache = ActivityCache()
collector = RaceDataCollector(cache)

# Add race results
runner = RunnerContext(age=35, sex='M', max_hr=185, experience_years=8)

collector.add_race_result(
    race_id="boston_2024",
    runner_id="runner1",
    race_date=datetime(2024, 4, 15),
    race_distance_miles=26.2,
    actual_time_minutes=195.5,  # 3:15:30
    runner_context=runner,
    lookback_weeks=12
)

# Get training data
X, y = collector.get_training_data()

# Train models
predictor = RaceTimePredictor()
X_scaled, y_array, features = predictor.prepare_data(X, y)
results = predictor.train_models(X_scaled, y_array)

# Save model
predictor.save_model()

# === Prediction ===

# Predict race time
result = predict_race_time(
    race_date=datetime(2024, 10, 15),
    race_distance_miles=26.2,
    runner_context=runner,
    lookback_weeks=12
)

print(f"Predicted time: {result['predicted_time_formatted']}")
print(f"Predicted pace: {result['predicted_pace_formatted']} /mile")
```

### Command Line Usage

```bash
# Create sample data
python create_sample_races.py

# Train model
python train_race_model.py synthetic_race_results.json

# Make predictions
python predict_race_time.py 26.2  # Marathon
python predict_race_time.py 13.1  # Half marathon
python predict_race_time.py 6.2   # 10K
```

## Model Performance

### Typical Results (10 race samples)

```
Model Comparison
======================================================================
Model                CV MAE          Train MAE       R²
----------------------------------------------------------------------
Lasso                 18.44 ± 8.96     1.33 min       0.999 ⭐
Ridge                 49.10 ± 21.96     2.04 min       0.999
Random Forest         59.11 ± 12.93    25.99 min       0.848
Gradient Boosting     70.78 ± 10.68     0.00 min       1.000
======================================================================
```

**Metrics:**
- **CV MAE**: Cross-validation Mean Absolute Error (lower is better)
- **Train MAE**: Training set error
- **R²**: Coefficient of determination (higher is better)

### Feature Importance

Top features for race time prediction:

1. `race_distance_miles` - Race distance (strongest predictor)
2. `tempo_workout_count` - Number of tempo workouts
3. `hr_per_grade_downhill` - Downhill performance
4. `zone1_percent` - Easy training volume
5. `hill_recovery_rate` - Recovery after hills
6. `elevation_tolerance` - Hill handling ability
7. `hr_variability_coefficient` - Heart rate consistency
8. `long_run_distance` - Long run quality

## Data Requirements

### Minimum Requirements

- **At least 2 race results** to train a model
- Each race needs:
  - Training data in lookback window (typically 8-12 weeks)
  - At least 10-20 activities with HR data in window
  - Race date, distance, and actual finish time

### Recommended Dataset

For best results:
- **10+ race results** across different distances
- **Multiple runners** (for generalization)
- **Variety of conditions** (different training levels, distances)
- **Recent data** (within last 2-3 years)

### Race Result Format

```json
{
  "race_id": "unique_race_identifier",
  "runner_id": "runner_identifier",
  "race_date": "2024-04-15",
  "race_distance_miles": 26.2,
  "actual_time_minutes": 195.5,
  "age": 35,
  "sex": "M",
  "max_hr": 185,
  "experience_years": 8,
  "resting_hr": 52,
  "recent_injury_flag": false,
  "lookback_weeks": 12
}
```

## Model Files

Models are saved to: `~/.strava_guru_cache/models/race_predictor.pkl`

The saved model includes:
- Trained model(s)
- StandardScaler (fitted)
- Feature names
- Training statistics
- Best model selection

## Cross-Validation

The pipeline uses k-fold cross-validation to evaluate models:

- **Default**: 5-fold CV
- **Adjusts** based on sample size (minimum 2 folds)
- **Scoring**: Negative Mean Absolute Error
- **Shuffle**: Random state = 42 for reproducibility

## Limitations & Considerations

### Current Limitations

1. **Single runner bias** - Model trained on one runner may not generalize
2. **Small sample size** - Need more race results for robust predictions
3. **Distance extrapolation** - Predicting distances far from training data is less accurate
4. **Environmental factors** - Doesn't account for weather, course difficulty (yet)
5. **Recent form** - Assumes training is representative of race-day fitness

### Best Practices

1. **Collect diverse data** - Multiple runners, distances, conditions
2. **Update regularly** - Retrain as you add more race results
3. **Validate predictions** - Compare predictions to actual results
4. **Use appropriate lookback** - Longer races need longer training windows
5. **Check feature importance** - Understand what drives predictions

## Future Improvements

### Planned Enhancements

1. **Ensemble predictions** - Combine multiple models
2. **Confidence intervals** - Provide prediction uncertainty
3. **Weather integration** - Account for race day conditions
4. **Course difficulty** - Incorporate elevation/terrain
5. **Form tracking** - Recent training quality
6. **Personalized models** - Train per runner or runner type
7. **Time series features** - Training progression over time

### Integration Points

- Weather API for race-day conditions
- GPX analysis for course profiles
- Training load metrics (TSS, TRIMP)
- Recovery scores
- Race history database

## Troubleshooting

### "Need at least 2 race results"

Create more race entries in your JSON file or use `create_sample_races.py` to generate synthetic data.

### "No activities in cache"

Run `build_cache.py` to populate the activity cache first.

### "Feature extraction failed"

Check that you have sufficient training data in the lookback window for the race date.

### "Model file not found"

Run `train_race_model.py` first to train and save a model.

### Poor predictions

- Add more diverse training data
- Check feature extraction quality
- Verify race results are accurate
- Consider retraining with more samples

## Dependencies

### Required

- numpy
- scikit-learn
- scipy (for feature engineering)

### Optional

- xgboost (for XGBoost model)

## Performance

- **Feature extraction**: ~5 seconds per race
- **Model training**: ~10-30 seconds for 10 samples
- **Prediction**: < 1 second

## Files

```
race_prediction/
├── __init__.py
├── README.md (this file)
├── data_collector.py       # Race data collection
├── model_trainer.py        # ML model training
└── predictor.py            # Prediction interface

Scripts:
├── train_race_model.py     # Training script
├── predict_race_time.py    # Prediction script
└── create_sample_races.py  # Generate synthetic data
```

## Example Output

### Training

```
============================================================
Model Comparison
============================================================
Lasso                 18.44 ± 8.96     1.33 min       0.999 ⭐
Random Forest         59.11 ± 12.93    25.99 min       0.848
============================================================

Top 15 Features (Lasso)
race_distance_miles             64.1095  ████████████
tempo_workout_count              6.5402  ██
zone1_percent                    3.1798  █
```

### Prediction

```
============================================================
Race Time Prediction
============================================================

Distance: 26.2 miles
Predicted Time: 3:34:11
Predicted Pace: 8:10 /mile

Key Training Metrics:
  Weekly Mileage: 40.5 miles
  Peak Mileage: 54.2 miles
  Training Consistency: 0.88
============================================================
```
