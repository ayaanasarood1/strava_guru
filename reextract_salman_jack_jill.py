#!/usr/bin/env python3
"""
Re-extract features for Salman's Jack & Jill 2025 with fixed cache
"""

import json
from pathlib import Path
from datetime import datetime
from feature_engineering.feature_extractor import TrainingFeatureExtractor
from feature_engineering.runner_context import RunnerContext
from activity_cache import ActivityCache

def main():
    print("="*80)
    print("Re-extracting Features for Salman's Jack & Jill 2025")
    print("="*80)

    # Load Salman's cache
    salman_cache = ActivityCache(
        cache_dir=Path.home() / ".strava_guru_cache" / "salman"
    )

    # Race info
    race_date = datetime(2025, 7, 27)
    race_distance = 26.2

    # Runner context
    runner_context = RunnerContext(
        age=30,
        sex='M',
        max_hr=190,
        resting_hr=50,
        experience_years=10
    )

    print(f"\nRace: Jack & Jill Marathon 2025")
    print(f"Date: {race_date.date()}")
    print(f"Distance: {race_distance} miles")
    print(f"Actual finish time: 2:55")
    print()

    # Create extractor
    extractor = TrainingFeatureExtractor(salman_cache)

    # Extract features
    print("Extracting features from corrected cache...")
    print("(This includes 88 runs from April 27 - July 20, 2025)")
    print()

    features = extractor.extract_features(
        runner_id="runner_2",
        race_date=race_date,
        lookback_weeks=12,
        race_distance_miles=race_distance,
        runner_context=runner_context
    )

    # Show key features
    print("="*80)
    print("Extracted Features (Full 41-feature set)")
    print("="*80)

    print(f"\n📊 Training Volume:")
    print(f"  Total weekly mileage: {features.total_weekly_mileage:.1f} mi/week")
    print(f"  Peak weekly mileage: {features.peak_weekly_mileage:.1f} miles")
    print(f"  Long run distance: {features.long_run_distance:.1f} miles")
    print(f"  Total runs: {features.total_runs}")
    print(f"  Runs per week: {features.runs_per_week:.1f}")

    print(f"\n🏃 Training Intensity:")
    print(f"  Zone 1%: {features.zone1_percent:.1f}%")
    print(f"  Zone 2%: {features.zone2_percent:.1f}%")
    print(f"  Zone 3%: {features.zone3_percent:.1f}%")
    print(f"  Zone 4%: {features.zone4_percent:.1f}%")
    print(f"  Zone 5%: {features.zone5_percent:.1f}%")
    print(f"  Quality workout%: {features.quality_workout_percent:.1f}%")

    print(f"\n💪 Lactate Threshold:")
    lt_hr_str = f"{features.lt_heart_rate}" if features.lt_heart_rate else "N/A"
    lt_pace_str = f"{features.lt_pace:.2f}" if features.lt_pace else "N/A"
    aet_hr_str = f"{features.aet_heart_rate}" if features.aet_heart_rate else "N/A"
    aet_pace_str = f"{features.aet_pace:.2f}" if features.aet_pace else "N/A"
    print(f"  LT heart rate: {lt_hr_str} bpm")
    print(f"  LT pace: {lt_pace_str} min/mile")
    print(f"  AET heart rate: {aet_hr_str} bpm")
    print(f"  AET pace: {aet_pace_str} min/mile")

    # Compare to previous extraction
    print("\n" + "="*80)
    print("Comparison to Previous Extraction")
    print("="*80)
    print(f"  BEFORE (corrupted cache): 8.4 mi/week")
    print(f"  AFTER (fixed cache):      {features.total_weekly_mileage:.1f} mi/week")
    print(f"  Actual from CSV:          69.3 mi/week")
    print()

    if abs(features.total_weekly_mileage - 69.3) < 5.0:
        print("  ✓ Feature extraction is now CORRECT!")
        print("    Fixed cache resolved the issue.")
    else:
        print(f"  ⚠ Still off by {abs(features.total_weekly_mileage - 69.3):.1f} mi/week")
        print("    Additional debugging needed.")

    # Save updated features
    dataset_path = Path.home() / '.strava_guru_cache' / 'race_data' / 'combined_41_features.json'

    with open(dataset_path, 'r') as f:
        all_races = json.load(f)

    # Find and update Jack & Jill 2025
    for race in all_races:
        if race['race_date'] == '2025-07-27' and race['runner_id'] == 'runner_2':
            print(f"\nUpdating race: {race.get('_race_name', 'Jack & Jill 2025')}")

            # Update features
            race['features'] = features.to_dict()

            print(f"  Old weekly mileage: {race['features'].get('total_weekly_mileage', 0):.1f}")
            print(f"  New weekly mileage: {features.total_weekly_mileage:.1f}")
            break

    # Save
    with open(dataset_path, 'w') as f:
        json.dump(all_races, f, indent=2)

    print(f"\n✓ Updated dataset: {dataset_path}")

if __name__ == '__main__':
    main()
