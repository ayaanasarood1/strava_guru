#!/usr/bin/env python3
"""
Test Feature Extraction Pipeline
Verify feature extraction on sample data
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from activity_cache import ActivityCache
from feature_engineering import TrainingFeatureExtractor, RunnerContext


def test_feature_extraction():
    """Test feature extraction with sample runner"""
    print("=" * 60)
    print("Feature Extraction Pipeline Test")
    print("=" * 60)

    # Initialize cache
    cache = ActivityCache()
    stats = cache.get_stats()

    print(f"\nCache Statistics:")
    print(f"  Total activities: {stats['total_activities']}")
    print(f"  With HR data: {stats['with_hr']}")
    print(f"  Date range: {stats['date_range']}")

    if stats['total_activities'] == 0:
        print("\nERROR: No activities in cache. Please run build_cache.py first.")
        sys.exit(1)

    # Create sample runner context
    print("\n" + "=" * 60)
    print("Sample Runner Profile")
    print("=" * 60)

    runner = RunnerContext(
        age=35,
        sex='M',
        max_hr=185,
        experience_years=8,
        resting_hr=52,
        recent_injury_flag=False
    )

    print(f"  Age: {runner.age}")
    print(f"  Sex: {runner.sex}")
    print(f"  Max HR: {runner.max_hr} bpm")
    print(f"  Experience: {runner.experience_years} years")
    print(f"  Resting HR: {runner.resting_hr} bpm")
    print(f"  Recent injury: {runner.recent_injury_flag}")

    # Extract features for hypothetical race
    print("\n" + "=" * 60)
    print("Feature Extraction")
    print("=" * 60)

    extractor = TrainingFeatureExtractor(cache)

    # Use a date based on available data
    # Assume race 2 weeks after most recent activity
    if stats['date_range'][1]:
        race_date = datetime.fromisoformat(stats['date_range'][1]) + timedelta(days=14)
    else:
        race_date = datetime.now() + timedelta(days=14)

    features = extractor.extract_features(
        runner_id="test_runner",
        race_date=race_date,
        lookback_weeks=12,
        race_distance_miles=26.2,  # Marathon
        runner_context=runner
    )

    # Display results
    print("\n" + "=" * 60)
    print("Extracted Features Summary")
    print("=" * 60)

    print("\n1. Lactate Threshold Features:")
    print(f"   LT Heart Rate: {features.lt_heart_rate:.1f} bpm" if features.lt_heart_rate else "   LT Heart Rate: None")
    print(f"   LT Pace: {features.lt_pace:.2f} min/mile" if features.lt_pace else "   LT Pace: None")
    print(f"   LT % Max HR: {features.lt_percent_max_hr:.1%}" if features.lt_percent_max_hr else "   LT % Max HR: None")
    print(f"   AET Heart Rate: {features.aet_heart_rate:.1f} bpm" if features.aet_heart_rate else "   AET Heart Rate: None")
    print(f"   AET Pace: {features.aet_pace:.2f} min/mile" if features.aet_pace else "   AET Pace: None")

    print("\n2. Training Volume Features:")
    print(f"   Total Weekly Mileage: {features.total_weekly_mileage:.1f} miles")
    print(f"   Peak Weekly Mileage: {features.peak_weekly_mileage:.1f} miles")
    print(f"   Long Run Distance: {features.long_run_distance:.1f} miles")
    print(f"   Long Run % Weekly: {features.long_run_percent_weekly:.1%}")
    print(f"   Total Runs: {features.total_runs}")
    print(f"   Runs per Week: {features.runs_per_week:.1f}")
    print(f"   Mileage Consistency: {features.mileage_consistency:.2f}")

    print("\n3. Training Intensity Features:")
    print(f"   Zone 1 %: {features.zone1_percent:.1f}%")
    print(f"   Zone 2 %: {features.zone2_percent:.1f}%")
    print(f"   Zone 3 %: {features.zone3_percent:.1f}%")
    print(f"   Zone 4 %: {features.zone4_percent:.1f}%")
    print(f"   Zone 5 %: {features.zone5_percent:.1f}%")
    print(f"   Tempo Workouts: {features.tempo_workout_count}")
    print(f"   Interval Workouts: {features.interval_workout_count}")
    print(f"   Quality Workout %: {features.quality_workout_percent:.1f}%")

    print("\n4. Running Efficiency Features:")
    print(f"   HR at Easy Pace: {features.hr_at_easy_pace:.1f} bpm" if features.hr_at_easy_pace else "   HR at Easy Pace: None")
    print(f"   HR at Marathon Pace: {features.hr_at_marathon_pace:.1f} bpm" if features.hr_at_marathon_pace else "   HR at Marathon Pace: None")
    print(f"   Cardiac Drift: {features.cardiac_drift:.2f} bpm/hr")
    print(f"   Aerobic Decoupling: {features.aerobic_decoupling:.2f}%")
    print(f"   HR Variability CV: {features.hr_variability_coefficient:.3f}")

    print("\n5. Terrain Handling Features:")
    print(f"   HR per Grade Uphill: {features.hr_per_grade_uphill:.2f} bpm/%")
    print(f"   HR per Grade Downhill: {features.hr_per_grade_downhill:.2f} bpm/%")
    print(f"   Hill Recovery Rate: {features.hill_recovery_rate:.2f} bpm/min")
    print(f"   Elevation Tolerance: {features.elevation_tolerance:.3f}")

    print("\n6. Race Context Features:")
    print(f"   Race Distance: {features.race_distance_miles:.1f} miles")
    print(f"   Taper Quality: {features.taper_quality_score:.2f}")
    print(f"   Days Since Hard Effort: {features.days_since_last_hard_effort}")

    print("\n7. Runner Personalization Features:")
    print(f"   Age Normalized: {features.age_normalized:.3f}")
    print(f"   Sex Encoded: {features.sex_encoded:.1f}")
    print(f"   Max HR Normalized: {features.max_hr_normalized:.3f}")
    print(f"   Experience Years: {features.experience_years:.3f}")
    print(f"   Recent Injury Flag: {features.recent_injury_flag:.1f}")
    print(f"   Training Consistency: {features.training_consistency_score:.3f}")

    # Export to dict and DataFrame
    print("\n" + "=" * 60)
    print("Export Tests")
    print("=" * 60)

    feature_dict = features.to_dict()
    print(f"  Dictionary: {len(feature_dict)} features")

    try:
        feature_df = features.to_dataframe()
        print(f"  DataFrame: {feature_df.shape}")
    except ImportError:
        print(f"  DataFrame: skipped (pandas not installed)")

    # Validation
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    is_valid = features.validate()
    print(f"  Feature validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  Non-None features: {features.feature_count()} / 41")

    # Check feature ranges
    print("\n  Feature Range Checks:")
    issues = []

    # HR features
    hr_features = {
        'lt_heart_rate': features.lt_heart_rate,
        'aet_heart_rate': features.aet_heart_rate,
        'hr_at_easy_pace': features.hr_at_easy_pace,
        'hr_at_marathon_pace': features.hr_at_marathon_pace,
    }
    for name, value in hr_features.items():
        if value is not None and not (50 <= value <= 220):
            issues.append(f"{name} out of range: {value}")

    # Normalized features (should be 0-1)
    normalized = {
        'lt_percent_max_hr': features.lt_percent_max_hr,
        'age_normalized': features.age_normalized,
        'max_hr_normalized': features.max_hr_normalized,
        'training_consistency_score': features.training_consistency_score,
    }
    for name, value in normalized.items():
        if value is not None and not (0 <= value <= 1):
            issues.append(f"{name} out of range [0-1]: {value}")

    if issues:
        print(f"    ⚠ Found {len(issues)} issues:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("    ✓ All features in expected ranges")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

    return features


if __name__ == '__main__':
    try:
        features = test_feature_extraction()
        print("\n✓ Feature extraction test completed successfully")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
