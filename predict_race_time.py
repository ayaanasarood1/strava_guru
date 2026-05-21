#!/usr/bin/env python3
"""
Predict Race Time

Use trained model to predict race time from current training
"""

import sys
from datetime import datetime, timedelta

from activity_cache import ActivityCache
from feature_engineering import RunnerContext
from race_prediction.predictor import predict_race_time, format_prediction_report


def main():
    """Make a race time prediction"""
    print("=" * 60)
    print("Race Time Prediction")
    print("=" * 60)

    # Check if model exists
    from pathlib import Path
    model_dir = Path.home() / ".strava_guru_cache" / "models"
    model_file = model_dir / "race_predictor.pkl"

    if not model_file.exists():
        print("\nERROR: No trained model found!")
        print(f"Expected: {model_file}")
        print("\nPlease run train_race_model.py first to train a model.")
        sys.exit(1)

    # Initialize cache
    cache = ActivityCache()
    stats = cache.get_stats()

    print(f"\nActivity cache: {stats['total_activities']} activities")

    if stats['total_activities'] == 0:
        print("ERROR: No activities in cache.")
        sys.exit(1)

    # Define race parameters
    print("\n" + "=" * 60)
    print("Race Parameters")
    print("=" * 60)

    # Race 2 weeks from most recent activity
    if stats['date_range'][1]:
        most_recent = datetime.fromisoformat(stats['date_range'][1])
        race_date = most_recent + timedelta(weeks=2)
    else:
        race_date = datetime.now() + timedelta(weeks=2)

    print(f"\nRace date: {race_date.strftime('%Y-%m-%d')}")

    # Get race distance from command line or use default
    if len(sys.argv) > 1:
        try:
            race_distance = float(sys.argv[1])
        except ValueError:
            print(f"Invalid distance: {sys.argv[1]}")
            sys.exit(1)
    else:
        race_distance = 26.2  # Default: marathon

    print(f"Race distance: {race_distance} miles")

    # Runner profile
    runner = RunnerContext(
        age=35,
        sex='M',
        max_hr=185,
        experience_years=8,
        resting_hr=52
    )

    print(f"\nRunner: {runner.age}yo {runner.sex}, Max HR: {runner.max_hr}")

    # Make prediction
    print("\n" + "=" * 60)
    print("Extracting Features & Predicting...")
    print("=" * 60)

    try:
        result = predict_race_time(
            race_date=race_date,
            race_distance_miles=race_distance,
            runner_context=runner,
            lookback_weeks=12,
            cache=cache
        )

        # Display results
        print("\n" + format_prediction_report(result))

        # Additional context
        print("\nPrediction Details:")
        print(f"  Based on {result['features_extracted']} training features")
        print(f"  Using {result['model_used']} model")

        print("\nInterpretation:")
        pace = result['predicted_pace_min_per_mile']
        if pace < 6.0:
            print("  Elite performance level")
        elif pace < 7.0:
            print("  Competitive runner")
        elif pace < 8.0:
            print("  Strong recreational runner")
        elif pace < 9.0:
            print("  Average recreational runner")
        else:
            print("  Beginner to intermediate level")

        print("\n" + "=" * 60)
        print("Note: This is a prediction based on training data.")
        print("Actual race performance depends on many factors:")
        print("  - Race day conditions (weather, course)")
        print("  - Nutrition and hydration")
        print("  - Pacing strategy")
        print("  - Mental preparation")
        print("  - Taper quality")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
