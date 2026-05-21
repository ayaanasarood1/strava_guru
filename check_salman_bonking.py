#!/usr/bin/env python3
"""
Check Salman's marathons for bonking
"""

import json
from pathlib import Path
from datetime import datetime
from activity_cache import ActivityCache
from extract_all_marathons import detect_bonking

def main():
    print("="*80)
    print("Checking Salman's Marathons for Bonking")
    print("="*80)

    # Load Salman's cache
    salman_cache = ActivityCache(
        cache_dir=Path.home() / ".strava_guru_cache" / "salman"
    )

    # Salman's activities directory
    salman_activities_dir = Path('/Users/osman/Downloads/export_1884062_salman/activities')

    # Load his marathon data
    with open('runner2_marathons.json', 'r') as f:
        marathons = json.load(f)

    print(f"\nAnalyzing {len(marathons)} marathons...\n")

    bonked_races = []
    clean_races = []
    unable_to_analyze = []

    for i, marathon in enumerate(marathons, 1):
        race_name = marathon.get('_race_name', marathon['race_date'])
        time = marathon['actual_time_minutes']
        race_date = datetime.fromisoformat(marathon['race_date'])

        print(f"{i:2d}. {race_name:40s} ({marathon['race_date']}) - {int(time//60)}:{int(time%60):02d}")

        # Detect bonking
        analysis = detect_bonking(
            salman_cache,
            race_date,
            threshold_percent=15.0,
            activities_dir=salman_activities_dir
        )

        if analysis.first_half_pace == 0:
            print(f"    ⚠ {analysis.explanation}")
            unable_to_analyze.append(marathon)
        elif analysis.is_bonk:
            print(f"    🚨 BONK DETECTED: {analysis.explanation}")
            bonked_races.append({
                'marathon': marathon,
                'analysis': analysis
            })
        else:
            print(f"    ✓ Clean: {analysis.explanation}")
            clean_races.append(marathon)
        print()

    # Summary
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"  Clean races: {len(clean_races)}")
    print(f"  Bonked races: {len(bonked_races)}")
    print(f"  Unable to analyze: {len(unable_to_analyze)}")

    if bonked_races:
        print(f"\n{'='*80}")
        print("Bonked Races (Should Filter Out)")
        print(f"{'='*80}")
        for item in bonked_races:
            m = item['marathon']
            a = item['analysis']
            print(f"\n  • {m.get('_race_name', m['race_date'])} ({m['race_date']})")
            print(f"    Time: {int(m['actual_time_minutes']//60)}:{int(m['actual_time_minutes']%60):02d}")
            print(f"    {a.explanation}")
            print(f"    Pace drop: {a.pace_degradation_percent:.1f}%")

        # Save list of race IDs to filter
        bonked_ids = [item['marathon']['race_id'] for item in bonked_races]
        print(f"\n  Race IDs to filter: {bonked_ids}")

if __name__ == '__main__':
    main()
