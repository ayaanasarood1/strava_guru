#!/usr/bin/env python3
"""
Example: Feature Extraction for Race Prediction

This script demonstrates how to use the feature engineering pipeline
to extract training features for race time prediction.
"""

from datetime import datetime, timedelta
from activity_cache import ActivityCache
from feature_engineering import TrainingFeatureExtractor, RunnerContext


def example_marathon_prediction():
    """Extract features for a marathon prediction"""

    print("Example: Marathon Race Prediction")
    print("=" * 60)

    # 1. Initialize cache and extractor
    cache = ActivityCache()
    extractor = TrainingFeatureExtractor(cache)

    # 2. Define runner profile
    runner = RunnerContext(
        age=35,
        sex='M',
        max_hr=185,
        experience_years=8,
        resting_hr=52,
        recent_injury_flag=False
    )

    print("\nRunner Profile:")
    print(f"  Age: {runner.age}, Sex: {runner.sex}")
    print(f"  Max HR: {runner.max_hr} bpm")
    print(f"  Experience: {runner.experience_years} years")

    # 3. Define race parameters
    race_date = datetime.now() + timedelta(weeks=2)  # Race in 2 weeks
    lookback_weeks = 12  # Use last 12 weeks of training
    race_distance = 26.2  # Marathon

    print(f"\nRace Parameters:")
    print(f"  Race date: {race_date.strftime('%Y-%m-%d')}")
    print(f"  Distance: {race_distance} miles")
    print(f"  Training lookback: {lookback_weeks} weeks")

    # 4. Extract features
    print(f"\nExtracting features...")
    features = extractor.extract_features(
        runner_id="example_runner",
        race_date=race_date,
        lookback_weeks=lookback_weeks,
        race_distance_miles=race_distance,
        runner_context=runner
    )

    # 5. Display key features
    print("\n" + "=" * 60)
    print("Key Training Features")
    print("=" * 60)

    print(f"\nTraining Volume:")
    print(f"  Weekly Mileage: {features.total_weekly_mileage:.1f} miles")
    print(f"  Peak Week: {features.peak_weekly_mileage:.1f} miles")
    print(f"  Long Run: {features.long_run_distance:.1f} miles")
    print(f"  Runs/Week: {features.runs_per_week:.1f}")

    print(f"\nTraining Intensity:")
    print(f"  Easy (Z1-Z2): {features.zone1_percent + features.zone2_percent:.1f}%")
    print(f"  Tempo (Z3): {features.zone3_percent:.1f}%")
    print(f"  Threshold+ (Z4-Z5): {features.zone4_percent + features.zone5_percent:.1f}%")
    print(f"  Quality Workouts: {features.tempo_workout_count + features.interval_workout_count}")

    if features.lt_heart_rate:
        print(f"\nLactate Threshold:")
        print(f"  LT HR: {features.lt_heart_rate:.0f} bpm")
        print(f"  LT Pace: {features.lt_pace:.2f} min/mile")

    if features.hr_at_easy_pace:
        print(f"\nRunning Efficiency:")
        print(f"  HR at Easy Pace: {features.hr_at_easy_pace:.0f} bpm")
        print(f"  Cardiac Drift: {features.cardiac_drift:.1f} bpm/hr")

    print(f"\nRace Readiness:")
    print(f"  Training Consistency: {features.training_consistency_score:.2f}")
    print(f"  Taper Quality: {features.taper_quality_score:.2f}")
    print(f"  Days Since Hard Effort: {features.days_since_last_hard_effort}")

    # 6. Validate and export
    print("\n" + "=" * 60)
    print("Export & Validation")
    print("=" * 60)

    is_valid = features.validate()
    print(f"  Validation: {'PASSED ✓' if is_valid else 'FAILED ✗'}")
    print(f"  Features extracted: {features.feature_count()} / 41")

    # Export for ML model
    feature_dict = features.to_dict()
    print(f"  Exported to dict: {len(feature_dict)} features")

    return features


def example_half_marathon_prediction():
    """Extract features for a half marathon prediction"""

    print("\n\nExample: Half Marathon Race Prediction")
    print("=" * 60)

    cache = ActivityCache()
    extractor = TrainingFeatureExtractor(cache)

    # Female runner, younger, less experience
    runner = RunnerContext(
        age=28,
        sex='F',
        max_hr=195,
        experience_years=3,
        resting_hr=58
    )

    print("\nRunner Profile:")
    print(f"  Age: {runner.age}, Sex: {runner.sex}")
    print(f"  Max HR: {runner.max_hr} bpm")
    print(f"  Experience: {runner.experience_years} years")

    # Half marathon, shorter training cycle
    race_date = datetime.now() + timedelta(weeks=1)
    lookback_weeks = 8  # 8-week training plan
    race_distance = 13.1  # Half marathon

    print(f"\nRace Parameters:")
    print(f"  Race date: {race_date.strftime('%Y-%m-%d')}")
    print(f"  Distance: {race_distance} miles")
    print(f"  Training lookback: {lookback_weeks} weeks")

    print(f"\nExtracting features...")
    features = extractor.extract_features(
        runner_id="example_runner_2",
        race_date=race_date,
        lookback_weeks=lookback_weeks,
        race_distance_miles=race_distance,
        runner_context=runner
    )

    print(f"\nKey Features:")
    print(f"  Weekly Mileage: {features.total_weekly_mileage:.1f} miles")
    print(f"  Training Consistency: {features.training_consistency_score:.2f}")
    print(f"  Quality Workouts: {features.tempo_workout_count + features.interval_workout_count}")
    print(f"  Features extracted: {features.feature_count()} / 41")

    return features


if __name__ == '__main__':
    # Run examples
    features_1 = example_marathon_prediction()
    features_2 = example_half_marathon_prediction()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train ML model using extracted features")
    print("  2. Use model to predict race times")
    print("  3. Evaluate predictions against actual results")
