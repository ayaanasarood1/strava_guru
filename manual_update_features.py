#!/usr/bin/env python3
"""
Manually update Salman's Jack & Jill 2025 features with corrected data
"""

import json
from pathlib import Path
from datetime import datetime
from feature_engineering.feature_extractor import TrainingFeatureExtractor
from feature_engineering.runner_context import RunnerContext
from activity_cache import ActivityCache

def main():
    print("="*80)
    print("Manually Updating Features for Salman's Jack & Jill 2025")
    print("="*80)

    # Load dataset
    dataset_path = Path.home() / '.strava_guru_cache' / 'race_data' / 'combined_41_features.json'

    with open(dataset_path, 'r') as f:
        all_races = json.load(f)

    print(f"\nLoaded {len(all_races)} races")

    # Load Salman's cache
    salman_cache = ActivityCache(
        cache_dir=Path.home() / ".strava_guru_cache" / "salman"
    )

    # Extract fresh features
    race_date = datetime(2025, 7, 27)
    runner_context = RunnerContext(
        age=30,
        sex='M',
        max_hr=190,
        resting_hr=50,
        experience_years=10
    )

    extractor = TrainingFeatureExtractor(salman_cache)

    print("\nExtracting fresh features from corrected cache...")
    features = extractor.extract_features(
        runner_id="runner_2",
        race_date=race_date,
        lookback_weeks=12,
        race_distance_miles=26.2,
        runner_context=runner_context
    )

    print(f"  Weekly mileage: {features.total_weekly_mileage:.1f} mi/week")

    # Find and update the race
    updated = False
    for race in all_races:
        if race['race_date'] == '2025-07-27T00:00:00' and race['runner_id'] == 'runner_2':
            print(f"\nUpdating race: {race.get('_race_name', 'Jack & Jill 2025')}")
            print(f"  Old weekly mileage: {race['features'].get('total_weekly_mileage', 0):.1f}")

            # Update features
            race['features'] = features.to_dict()

            print(f"  New weekly mileage: {race['features']['total_weekly_mileage']:.1f}")
            updated = True
            break

    if not updated:
        print("\n❌ Race not found in dataset!")
        return

    # Save
    print(f"\nSaving updated dataset...")
    with open(dataset_path, 'w') as f:
        json.dump(all_races, f, indent=2)

    print(f"✓ Saved to {dataset_path}")

    # Verify
    print(f"\nVerifying update...")
    with open(dataset_path, 'r') as f:
        verify_data = json.load(f)

    for race in verify_data:
        if race['race_date'] == '2025-07-27T00:00:00' and race['runner_id'] == 'runner_2':
            verify_mileage = race['features'].get('total_weekly_mileage', 0)
            print(f"  Verified weekly mileage: {verify_mileage:.1f} mi/week")

            if abs(verify_mileage - 72.3) < 1.0:
                print(f"  ✓ Update successful!")
            else:
                print(f"  ❌ Update failed - mileage is {verify_mileage:.1f}, expected 72.3")

if __name__ == '__main__':
    main()
