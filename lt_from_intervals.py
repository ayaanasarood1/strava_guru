#!/usr/bin/env python3
"""
Find lactate threshold from interval workouts (e.g., 3 x 3 miles)
"""

from activity_cache import ActivityCache
from datetime import datetime, timedelta
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from activity_analyzer import format_pace


def find_interval_workouts(cache, months_back=None, min_activity_hr=140):
    """
    Find interval workouts like 3 x 3 miles, 4 x 1 mile, etc.

    Looks for:
    - Multiple laps of similar distance (e.g., 3 miles each)
    - Fast pace (6-9 min/mile)
    - High activity-level HR
    """
    if months_back:
        start_date = datetime.now() - timedelta(days=months_back * 30)
        activities = cache.get_activities_by_date_range(start_date, datetime.now())
    else:
        activities = cache.get_activities_with_hr()

    workouts = []

    for activity in activities:
        if not activity['laps_json']:
            continue

        if not activity['avg_heart_rate'] or activity['avg_heart_rate'] < min_activity_hr:
            continue

        laps = json.loads(activity['laps_json'])

        # Group laps by similar distance (within 0.3 miles)
        distance_groups = {}

        for lap in laps:
            dist_mi = lap['distance'] * 0.000621371
            pace = lap['pace']

            # Must be reasonable interval distance (0.8-5 miles) and pace (5-9 min/mi)
            if not (0.8 <= dist_mi <= 5.0):
                continue
            if not (5 < pace < 9):
                continue

            # Find matching distance group
            found_group = False
            for group_dist in distance_groups:
                if abs(dist_mi - group_dist) < 0.3:  # Within 0.3 miles
                    distance_groups[group_dist].append(lap)
                    found_group = True
                    break

            if not found_group:
                distance_groups[dist_mi] = [lap]

        # Find groups with 2+ intervals (e.g., 3 x 3 miles, 4 x 1 mile)
        for group_dist, group_laps in distance_groups.items():
            if len(group_laps) >= 2:  # At least 2 reps
                # Filter to only FAST laps (threshold reps, not warm-up/recovery)
                gaps = [lap['gap'] if lap['gap'] and lap['gap'] > 0 else lap['pace'] for lap in group_laps]

                # Find the fast laps - exclude anything > 8:00 pace (likely warm-up/cool-down)
                fast_laps = [lap for i, lap in enumerate(group_laps) if gaps[i] < 8.0]

                if len(fast_laps) < 2:
                    continue

                # Check pace consistency across FAST reps only
                fast_gaps = [lap['gap'] if lap['gap'] and lap['gap'] > 0 else lap['pace'] for lap in fast_laps]
                pace_cv = np.std(fast_gaps) / np.mean(fast_gaps)

                if pace_cv < 0.15:  # Consistent pace (< 15% variation)
                    avg_threshold_pace = np.mean(fast_gaps)

                    workouts.append({
                        'date': activity['activity_date'],
                        'activity_hr': activity['avg_heart_rate'],
                        'distance': group_dist,
                        'reps': len(fast_laps),  # Number of threshold reps only
                        'avg_pace': avg_threshold_pace,
                        'laps': fast_laps  # Only the fast threshold laps
                    })
                    break  # Only count each activity once

    return workouts


def plot_interval_progression(workouts, lt_hr, lt_pace):
    """Create visualization of interval workout progression"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from sklearn.linear_model import LinearRegression

    sorted_workouts = sorted(workouts, key=lambda w: w['date'])
    dates = [datetime.fromisoformat(w['date']) for w in sorted_workouts]
    hrs = [w['activity_hr'] for w in sorted_workouts]
    paces = [w['avg_pace'] for w in sorted_workouts]

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Lactate Threshold from Interval Workouts', fontsize=16, fontweight='bold')

    # 1. Pace progression over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(dates, paces, s=100, alpha=0.6, color='#4AB4DC')
    ax1.plot(dates, paces, alpha=0.3, linestyle='-', linewidth=1, color='#4AB4DC')

    # Trend line
    X = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)
    y_pace = np.array(paces)
    model = LinearRegression()
    model.fit(X, y_pace)
    trend_pace = model.predict(X)
    ax1.plot(dates, trend_pace, 'r--', linewidth=2, label='Trend')

    # LT line
    ax1.axhline(y=lt_pace, color='orange', linestyle='--', linewidth=2,
                label=f'Current LT ({format_pace(lt_pace)})')

    ax1.set_ylabel('Pace (min/mile)', fontweight='bold')
    ax1.set_title('Threshold Pace Progression', fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Heart rate progression
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(dates, hrs, s=100, alpha=0.6, color='#FC4C02')
    ax2.plot(dates, hrs, alpha=0.3, linestyle='-', linewidth=1, color='#FC4C02')

    # Trend line
    y_hr = np.array(hrs)
    model.fit(X, y_hr)
    trend_hr = model.predict(X)
    ax2.plot(dates, trend_hr, 'r--', linewidth=2, label='Trend')

    # LT line
    ax2.axhline(y=lt_hr, color='red', linestyle='--', linewidth=2,
                label=f'Activity Avg LT ({lt_hr} bpm)')
    ax2.axhline(y=lt_hr + 8, color='darkred', linestyle=':', linewidth=2,
                label=f'Rep HR (~{lt_hr + 8} bpm)')

    ax2.set_ylabel('Heart Rate (bpm)', fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_title('Heart Rate Progression', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Workout distribution
    ax3 = fig.add_subplot(gs[1, 1])
    workout_types = {}
    for w in sorted_workouts:
        key = f"{w['reps']} x {w['distance']:.1f}mi"
        workout_types[key] = workout_types.get(key, 0) + 1

    ax3.bar(workout_types.keys(), workout_types.values(), color='#4AB4DC', alpha=0.7)
    ax3.set_ylabel('Number of Workouts', fontweight='bold')
    ax3.set_xlabel('Workout Type', fontweight='bold')
    ax3.set_title('Workout Distribution', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Summary stats
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    summary = f"""
LACTATE THRESHOLD ESTIMATE (from interval workouts)

Threshold Metrics:
  LT Heart Rate (activity avg):     {lt_hr} bpm
  LT Heart Rate (during reps):      ~{lt_hr + 8} bpm
  LT Pace:                           {format_pace(lt_pace)}

Data Summary:
  Interval workouts analyzed:        {len(workouts)}
  Date range:                        {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}

Recent Workouts:
"""
    for workout in sorted_workouts[-5:]:
        w_date = datetime.fromisoformat(workout['date']).strftime('%Y-%m-%d')
        summary += f"  {w_date}: {workout['reps']} x {workout['distance']:.1f}mi @ {format_pace(workout['avg_pace'])}, {workout['activity_hr']} bpm\n"

    summary += f"""
Training Zones (based on rep HR ~{lt_hr + 8} bpm):
  Zone 1 (Recovery):    < {int((lt_hr + 8) * 0.70)} bpm
  Zone 2 (Easy):        {int((lt_hr + 8) * 0.70)}-{int((lt_hr + 8) * 0.85)} bpm
  Zone 3 (Tempo):       {int((lt_hr + 8) * 0.85)}-{int((lt_hr + 8) * 0.95)} bpm
  Zone 4 (Threshold):   {int((lt_hr + 8) * 0.95)}-{int((lt_hr + 8) * 1.05)} bpm  ← Your threshold zone
  Zone 5 (VO2max):      > {int((lt_hr + 8) * 1.05)} bpm

Note: Activity HR includes warm-up/cool-down. Actual HR during reps is typically 5-10 bpm higher.
"""

    ax4.text(0.1, 0.95, summary, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = "./charts/lactate_threshold_intervals.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    plt.close()


def main():
    """Main entry point"""
    import sys

    cache = ActivityCache()

    # Parse arguments
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    min_hr = 140

    if '--min-hr' in sys.argv:
        hr_idx = sys.argv.index('--min-hr')
        if hr_idx + 1 < len(sys.argv):
            min_hr = int(sys.argv[hr_idx + 1])

    print(f"Finding interval workouts (last {months} months)")
    print(f"Minimum activity HR: {min_hr} bpm")
    print()

    workouts = find_interval_workouts(cache, months_back=months, min_activity_hr=min_hr)

    if not workouts:
        print("No interval workouts found!")
        print("Try lowering --min-hr or checking more months")
        return

    print(f"Found {len(workouts)} interval workouts:")
    print("="*80)

    for workout in sorted(workouts, key=lambda w: w['date']):
        print(f"\n📅 {workout['date']}")
        print(f"   {workout['reps']} x {workout['distance']:.1f} miles")
        print(f"   Average pace: {format_pace(workout['avg_pace'])}")
        print(f"   Activity HR: {workout['activity_hr']} bpm")
        print(f"   Individual reps:")
        for lap in workout['laps']:
            gap = lap['gap'] if lap['gap'] and lap['gap'] > 0 else lap['pace']
            print(f"      Lap {lap['number']}: {format_pace(gap)} GAP")

    print()
    print("="*80)
    print("🎯 LACTATE THRESHOLD ESTIMATE (from interval workouts)")
    print("="*80)

    # Calculate LT from these workouts
    # Use activity HR as proxy (actual rep HR is likely 5-10 bpm higher)
    hrs = [w['activity_hr'] for w in workouts]
    paces = [w['avg_pace'] for w in workouts]

    # Weight recent workouts more
    sorted_workouts = sorted(workouts, key=lambda w: w['date'])
    if len(sorted_workouts) >= 3:
        weights = np.linspace(0.5, 1.0, len(sorted_workouts))
    else:
        weights = np.ones(len(sorted_workouts))

    lt_hr = int(np.average(hrs, weights=weights))
    lt_pace = np.average(paces, weights=weights)

    print(f"  LT Heart Rate: {lt_hr} bpm (activity avg)")
    print(f"                 ~{lt_hr + 8} bpm (estimated during reps)")
    print(f"  LT Pace:       {format_pace(lt_pace)}")
    print(f"  Based on:      {len(workouts)} interval workouts")
    print("="*80)
    print()
    print("Note: Activity HR is averaged over entire run (including warm-up/cool-down).")
    print("Actual HR during threshold reps is typically 5-10 bpm higher.")

    # Generate visualization
    print("\nGenerating visualization...")
    plot_interval_progression(workouts, lt_hr, lt_pace)


if __name__ == '__main__':
    main()
