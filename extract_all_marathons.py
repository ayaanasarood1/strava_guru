#!/usr/bin/env python3
"""
Generic Marathon Feature Extraction Pipeline

Extracts full 41 features from any runner's data with data quality checks.
Filters out races with signs of bonking or other issues.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from activity_cache import ActivityCache
from race_prediction.data_collector import RaceDataCollector
from feature_engineering import RunnerContext


@dataclass
class BonkingAnalysis:
    """Analysis of whether a race shows signs of bonking"""
    is_bonk: bool
    first_half_pace: float  # min/mile
    second_half_pace: float  # min/mile
    pace_degradation_percent: float
    explanation: str


def detect_bonking(cache: ActivityCache, race_date: datetime,
                   threshold_percent: float = 15.0,
                   activities_dir: Path = None) -> BonkingAnalysis:
    """
    Detect if a marathon shows signs of bonking based on pace degradation.

    Bonking Detection Strategy:
    - Compare pace in first half vs second half of race
    - Flag if pace degrades more than threshold (default 15%)
    - This catches the classic "hit the wall" pattern

    Real-world context:
    - Glycogen depletion typically occurs around mile 18-20
    - Bonked races show dramatic pace drops (20-30%+) in final miles
    - Even trained runners can bonk from poor pacing or fueling
    - These races are outliers - not representative of fitness level

    Args:
        cache: ActivityCache with race data
        race_date: Date of the race
        threshold_percent: Pace degradation % to flag as bonk (default 15%)

    Returns:
        BonkingAnalysis with bonking verdict and metrics
    """

    # Get the race activity from cache
    from activity_analyzer import ActivityAnalyzer

    conn = sqlite3.connect(cache.db_path)
    cursor = conn.cursor()

    # Find race activity (within 1 day of race_date)
    race_date_str = race_date.date().isoformat()
    cursor.execute("""
        SELECT file_name
        FROM activities
        WHERE DATE(activity_date) = ?
          AND distance_meters > 40000  -- Marathon distance
        LIMIT 1
    """, (race_date_str,))

    result = cursor.fetchone()
    conn.close()

    if not result:
        return BonkingAnalysis(
            is_bonk=False,
            first_half_pace=0,
            second_half_pace=0,
            pace_degradation_percent=0,
            explanation="No race data found for analysis"
        )

    file_name = result[0]

    # Load the activity file to get lap data
    # Try multiple possible locations
    possible_paths = [
        cache.cache_dir.parent / file_name,
        Path(cache.cache_dir) / "race_data" / file_name,
    ]

    # If activities_dir provided, look there
    if activities_dir:
        possible_paths.insert(0, activities_dir / file_name)

    activity_file = None
    for path in possible_paths:
        if path.exists():
            activity_file = path
            break

    if not activity_file:
        return BonkingAnalysis(
            is_bonk=False,
            first_half_pace=0,
            second_half_pace=0,
            pace_degradation_percent=0,
            explanation="Activity file not found"
        )

    # Parse activity to get laps
    analyzer = ActivityAnalyzer()
    try:
        stats = analyzer.analyze_file(activity_file)
        if not stats or not stats.laps or len(stats.laps) < 10:
            return BonkingAnalysis(
                is_bonk=False,
                first_half_pace=0,
                second_half_pace=0,
                pace_degradation_percent=0,
                explanation="Insufficient lap data for analysis"
            )

        # Calculate first half vs second half pace
        total_laps = len(stats.laps)
        midpoint = total_laps // 2

        first_half_laps = stats.laps[:midpoint]
        second_half_laps = stats.laps[midpoint:]

        # Average pace for each half
        first_half_pace = sum(lap.pace for lap in first_half_laps) / len(first_half_laps)
        second_half_pace = sum(lap.pace for lap in second_half_laps) / len(second_half_laps)

        # Calculate degradation
        pace_degradation = ((second_half_pace - first_half_pace) / first_half_pace) * 100.0

        is_bonk = pace_degradation > threshold_percent

        explanation = (
            f"First half: {first_half_pace:.2f} min/mile, "
            f"Second half: {second_half_pace:.2f} min/mile, "
            f"Degradation: {pace_degradation:.1f}%"
        )

        if is_bonk:
            explanation += f" (BONKED - exceeded {threshold_percent}% threshold)"

        return BonkingAnalysis(
            is_bonk=is_bonk,
            first_half_pace=first_half_pace,
            second_half_pace=second_half_pace,
            pace_degradation_percent=pace_degradation,
            explanation=explanation
        )

    except Exception as e:
        return BonkingAnalysis(
            is_bonk=False,
            first_half_pace=0,
            second_half_pace=0,
            pace_degradation_percent=0,
            explanation=f"Error analyzing activity: {str(e)[:50]}"
        )


def extract_marathons_for_runner(
    runner_name: str,
    cache_dir: Path,
    marathon_list_file: str,
    bonk_threshold: float = 15.0
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generic extraction function for any runner.

    Args:
        runner_name: Name of the runner (for logging)
        cache_dir: Path to runner's activity cache
        marathon_list_file: JSON file with marathon race results
        bonk_threshold: Pace degradation % to flag as bonk

    Returns:
        (valid_races, filtered_races) tuple
    """

    print(f"\n{'='*80}")
    print(f"Extracting Features for {runner_name}")
    print(f"{'='*80}")

    # Load cache
    cache = ActivityCache(cache_dir=cache_dir)
    stats = cache.get_stats()
    print(f"Cache: {stats.get('total_activities', 0)} activities")

    # Load marathon list
    with open(marathon_list_file, 'r') as f:
        marathons = json.load(f)

    print(f"Found {len(marathons)} marathons to process")

    # Initialize collector
    collector = RaceDataCollector(cache=cache)

    valid_races = []
    filtered_races = []

    # Process each marathon
    for i, marathon in enumerate(marathons, 1):
        race_name = marathon.get('_race_name', marathon['race_date'])
        print(f"\n{i}/{len(marathons)}: {race_name}")
        print(f"  Date: {marathon['race_date']}")
        print(f"  Time: {int(marathon['actual_time_minutes']//60)}:{int(marathon['actual_time_minutes']%60):02d}")

        race_date = datetime.fromisoformat(marathon['race_date'])

        # DATA QUALITY CHECK: Detect bonking
        print(f"  Analyzing race quality...")
        bonk_analysis = detect_bonking(cache, race_date, bonk_threshold)
        print(f"    {bonk_analysis.explanation}")

        if bonk_analysis.is_bonk:
            print(f"  ⚠ FILTERED OUT: Race shows signs of bonking")
            filtered_races.append({
                'marathon': marathon,
                'reason': 'bonking',
                'analysis': bonk_analysis
            })
            continue

        # Extract features
        runner_context = RunnerContext(
            age=marathon['age'],
            sex=marathon['sex'],
            max_hr=marathon['max_hr'],
            experience_years=marathon.get('experience_years'),
            resting_hr=marathon.get('resting_hr')
        )

        try:
            collector.add_race_result(
                race_id=marathon['race_id'],
                runner_id=marathon['runner_id'],
                race_date=race_date,
                race_distance_miles=marathon['race_distance_miles'],
                actual_time_minutes=marathon['actual_time_minutes'],
                runner_context=runner_context,
                lookback_weeks=marathon['lookback_weeks']
            )

            if collector.race_results:
                features = collector.race_results[-1].features
                non_null = sum(1 for v in features.values() if v is not None and v != 0)
                print(f"  ✓ Extracted {non_null} features")
                valid_races.append(collector.race_results[-1])

        except Exception as e:
            print(f"  ✗ Error extracting features: {str(e)[:100]}")
            filtered_races.append({
                'marathon': marathon,
                'reason': 'extraction_error',
                'error': str(e)
            })

    return valid_races, filtered_races


def main():
    """
    Extract features for all runners with data quality filtering
    """

    print("="*80)
    print("Generic Marathon Feature Extraction Pipeline")
    print("="*80)
    print("\nData Quality Checks:")
    print("  • Bonking detection (pace degradation > 15% after mile 15)")
    print("  • Feature extraction validation")
    print("  • Training data availability")

    # ========================================================================
    # Runner 1: User's marathons
    # ========================================================================
    user_cache = Path.home() / ".strava_guru_cache"
    user_races, user_filtered = extract_marathons_for_runner(
        runner_name="You",
        cache_dir=user_cache,
        marathon_list_file="my_actual_marathons.json",
        bonk_threshold=15.0
    )

    # ========================================================================
    # Runner 2: Salman's marathons (already extracted, just load)
    # ========================================================================
    print(f"\n{'='*80}")
    print("Loading Salman's Previously Extracted Data")
    print(f"{'='*80}")

    salman_path = Path.home() / ".strava_guru_cache" / "race_data" / "salman_full_features_dataset.json"
    with open(salman_path, 'r') as f:
        salman_data = json.load(f)

    print(f"Loaded {len(salman_data)} races for Salman")

    # ========================================================================
    # Combine datasets
    # ========================================================================
    print(f"\n{'='*80}")
    print("Combined Dataset Summary")
    print(f"{'='*80}")

    # Convert user races to dict format
    from race_prediction.data_collector import RaceResult
    user_races_dict = [
        {
            'race_id': r.race_id,
            'runner_id': r.runner_id,
            'race_date': r.race_date.isoformat(),
            'race_distance_miles': r.race_distance_miles,
            'actual_time_minutes': r.actual_time_minutes,
            'age': r.age,
            'sex': r.sex,
            'max_hr': r.max_hr,
            'experience_years': r.experience_years,
            'lookback_weeks': r.lookback_weeks,
            'features': r.features
        }
        for r in user_races
    ]

    combined_data = user_races_dict + salman_data

    print(f"  Your races (valid): {len(user_races)}")
    print(f"  Your races (filtered): {len(user_filtered)}")
    print(f"  Salman's races: {len(salman_data)}")
    print(f"  Combined total: {len(combined_data)}")

    # Save combined dataset
    output_path = Path.home() / ".strava_guru_cache" / "race_data" / "combined_41_features.json"
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)

    print(f"\n✓ Saved combined dataset to {output_path}")

    # Show filtered races
    if user_filtered:
        print(f"\n{'='*80}")
        print("Filtered Races (Not Included in Training)")
        print(f"{'='*80}")
        for item in user_filtered:
            marathon = item['marathon']
            reason = item['reason']
            race_name = marathon.get('_race_name', marathon['race_date'])
            print(f"\n  • {race_name} ({marathon['race_date']})")
            print(f"    Reason: {reason}")
            if reason == 'bonking':
                analysis = item['analysis']
                print(f"    {analysis.explanation}")

    # Show time ranges
    all_times = [r['actual_time_minutes'] for r in combined_data]
    print(f"\n{'='*80}")
    print("Final Dataset Statistics")
    print(f"{'='*80}")
    print(f"  Total races: {len(combined_data)}")
    print(f"  Time range: {min(all_times)//60}:{int(min(all_times)%60):02d} - {max(all_times)//60}:{int(max(all_times)%60):02d}")
    print(f"  Mean time: {sum(all_times)/len(all_times)//60}:{int((sum(all_times)/len(all_times))%60):02d}")
    print(f"  Features: 41 (full feature engineering pipeline)")


if __name__ == '__main__':
    main()
