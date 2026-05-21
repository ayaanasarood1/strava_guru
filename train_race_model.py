#!/usr/bin/env python3
"""
Train Race Time Prediction Model

This script:
1. Collects race results and extracts features
2. Trains multiple ML models
3. Evaluates model performance
4. Saves the best model for predictions
"""

import sys
from pathlib import Path
from datetime import datetime

from activity_cache import ActivityCache
from feature_engineering import RunnerContext
from race_prediction import RaceDataCollector, RaceTimePredictor


def create_sample_dataset():
    """Create sample race dataset for demonstration

    In practice, you would load real race results from a database or file
    """
    print("\n" + "=" * 60)
    print("Creating Sample Dataset")
    print("=" * 60)

    cache = ActivityCache()
    collector = RaceDataCollector(cache)

    # Get recent activities to create synthetic race results
    # This is for demonstration - in reality you'd have actual race results
    stats = cache.get_stats()
    print(f"\nActivity cache: {stats['total_activities']} activities")

    if stats['total_activities'] == 0:
        print("ERROR: No activities in cache. Please run build_cache.py first.")
        return None

    # Create a synthetic dataset from recent training
    # In practice, you would have actual race results like:
    #
    # collector.add_race_result(
    #     race_id="boston_2024",
    #     runner_id="runner1",
    #     race_date=datetime(2024, 4, 15),
    #     race_distance_miles=26.2,
    #     actual_time_minutes=195.5,  # 3:15:30
    #     runner_context=RunnerContext(age=35, sex='M', max_hr=185, experience_years=8),
    #     lookback_weeks=12
    # )

    print("\nNOTE: This is a demonstration with synthetic data.")
    print("For real predictions, provide actual race results.")

    # Get most recent activity date
    if stats['date_range'][1]:
        most_recent = datetime.fromisoformat(stats['date_range'][1])
    else:
        most_recent = datetime.now()

    # Create runner profile
    runner = RunnerContext(
        age=35,
        sex='M',
        max_hr=185,
        experience_years=8,
        resting_hr=52
    )

    # Create sample race result
    # Use a date that has training data before it
    race_date = most_recent

    print(f"\nAdding sample race result...")
    print(f"  (In production, load real race results from JSON/database)")

    collector.add_race_result(
        race_id=f"sample_race_{race_date.strftime('%Y%m%d')}",
        runner_id="sample_runner",
        race_date=race_date,
        race_distance_miles=26.2,
        actual_time_minutes=210.0,  # 3:30:00 marathon time
        runner_context=runner,
        lookback_weeks=12
    )

    return collector


def train_from_json(json_file: Path):
    """Train model from JSON file of race results

    JSON format:
    [
        {
            "race_id": "boston_2024_runner1",
            "runner_id": "runner1",
            "race_date": "2024-04-15",
            "race_distance_miles": 26.2,
            "actual_time_minutes": 195.5,
            "age": 35,
            "sex": "M",
            "max_hr": 185,
            "experience_years": 8
        },
        ...
    ]
    """
    print("\n" + "=" * 60)
    print("Training from JSON File")
    print("=" * 60)

    cache = ActivityCache()
    collector = RaceDataCollector(cache)

    collector.load_from_json(json_file)

    return collector


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Race Time Prediction - Model Training")
    print("=" * 60)

    # Check for JSON input file
    if len(sys.argv) > 1:
        json_file = Path(sys.argv[1])
        if json_file.exists():
            collector = train_from_json(json_file)
        else:
            print(f"ERROR: File not found: {json_file}")
            print("\nUsage: python train_race_model.py [race_results.json]")
            return
    else:
        # Create sample dataset for demonstration
        collector = create_sample_dataset()

    if collector is None:
        return

    # Get training data
    X, y = collector.get_training_data()

    if len(X) < 2:
        print("\nERROR: Need at least 2 race results to train a model")
        print("\nTo use this tool:")
        print("1. Create a JSON file with race results (see format above)")
        print("2. Run: python train_race_model.py your_races.json")
        print("\nFor demonstration purposes, this created a single sample.")
        print("The model needs multiple race results to train effectively.")
        return

    # Dataset statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    stats = collector.get_dataset_statistics()
    print(f"Total samples: {stats['total_samples']}")
    print(f"With features: {stats['with_features']}")
    print(f"Unique runners: {stats['runners']}")

    print(f"\nRace distances:")
    for dist, count in sorted(stats['race_distances'].items()):
        print(f"  {dist} miles: {count} races")

    print(f"\nTime range:")
    print(f"  Min: {stats['time_range']['min']:.1f} min")
    print(f"  Max: {stats['time_range']['max']:.1f} min")
    print(f"  Mean: {stats['time_range']['mean']:.1f} min")

    # Train models
    predictor = RaceTimePredictor()

    print("\n" + "=" * 60)
    print("Preparing Data")
    print("=" * 60)

    X_scaled, y_array, feature_names = predictor.prepare_data(X, y)

    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)

    # Cross-validation folds (minimum 2, max 5)
    cv_folds = min(5, len(X))

    results = predictor.train_models(X_scaled, y_array, cv_folds=cv_folds)

    # Print comparison
    predictor.print_model_comparison()

    # Feature importance
    predictor.print_feature_importance(top_n=15)

    # Save model
    model_file = predictor.save_model()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest model: {predictor.best_model_name}")
    print(f"Saved to: {model_file}")

    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("1. Use predict_race_time.py to make predictions")
    print("2. Add more race results to improve the model")
    print("3. Experiment with different race distances")

    # Save dataset for future use
    collector.save_dataset()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
