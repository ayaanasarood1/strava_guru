"""
Race Time Prediction Interface
High-level API for making predictions
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from activity_cache import ActivityCache
from feature_engineering import TrainingFeatureExtractor, RunnerContext
from .model_trainer import RaceTimePredictor


def predict_race_time(
    race_date: datetime,
    race_distance_miles: float,
    runner_context: RunnerContext,
    lookback_weeks: int = 12,
    cache: Optional[ActivityCache] = None,
    model_file: str = "race_predictor.pkl"
) -> dict:
    """Predict race time using trained model

    Args:
        race_date: Date of target race
        race_distance_miles: Race distance in miles
        runner_context: Runner personal characteristics
        lookback_weeks: Training lookback period
        cache: ActivityCache instance (creates new if None)
        model_file: Trained model filename

    Returns:
        Dictionary with prediction and details
    """
    # Initialize cache if needed
    if cache is None:
        cache = ActivityCache()

    # Load trained model
    predictor = RaceTimePredictor()
    predictor.load_model(model_file)

    # Extract features
    extractor = TrainingFeatureExtractor(cache)

    features_obj = extractor.extract_features(
        runner_id="prediction",
        race_date=race_date,
        lookback_weeks=lookback_weeks,
        race_distance_miles=race_distance_miles,
        runner_context=runner_context
    )

    features = features_obj.to_dict()

    # Make prediction
    predicted_minutes = predictor.predict(features)

    # Format time
    hours = int(predicted_minutes // 60)
    mins = int(predicted_minutes % 60)
    secs = int((predicted_minutes % 1) * 60)

    if hours > 0:
        time_str = f"{hours}:{mins:02d}:{secs:02d}"
    else:
        time_str = f"{mins}:{secs:02d}"

    # Calculate pace
    pace_minutes = predicted_minutes / race_distance_miles

    # Prepare result
    result = {
        'predicted_time_minutes': predicted_minutes,
        'predicted_time_formatted': time_str,
        'predicted_pace_min_per_mile': pace_minutes,
        'predicted_pace_formatted': f"{int(pace_minutes)}:{int((pace_minutes % 1) * 60):02d}",
        'race_distance_miles': race_distance_miles,
        'model_used': predictor.best_model_name,
        'features_extracted': features_obj.feature_count(),
        'key_features': {
            'weekly_mileage': features_obj.total_weekly_mileage,
            'peak_mileage': features_obj.peak_weekly_mileage,
            'training_consistency': features_obj.training_consistency_score,
            'lt_heart_rate': features_obj.lt_heart_rate,
            'lt_pace': features_obj.lt_pace,
        }
    }

    return result


def format_prediction_report(result: dict) -> str:
    """Format prediction result as readable report

    Args:
        result: Prediction result dictionary

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("Race Time Prediction")
    report.append("=" * 60)

    report.append(f"\nDistance: {result['race_distance_miles']} miles")
    report.append(f"\nPredicted Time: {result['predicted_time_formatted']}")
    report.append(f"Predicted Pace: {result['predicted_pace_formatted']} /mile")

    report.append(f"\nModel: {result['model_used']}")
    report.append(f"Features Used: {result['features_extracted']}")

    report.append("\nKey Training Metrics:")
    kf = result['key_features']
    report.append(f"  Weekly Mileage: {kf['weekly_mileage']:.1f} miles")
    report.append(f"  Peak Mileage: {kf['peak_mileage']:.1f} miles")
    report.append(f"  Training Consistency: {kf['training_consistency']:.2f}")

    if kf.get('lt_heart_rate'):
        report.append(f"  LT Heart Rate: {kf['lt_heart_rate']:.0f} bpm")
        report.append(f"  LT Pace: {kf['lt_pace']:.2f} min/mile")

    report.append("=" * 60)

    return "\n".join(report)
