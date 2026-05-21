#!/usr/bin/env python3
"""
Estimate marathon predictions with +10F warmer weather
Based on research on temperature impact on marathon performance
"""

def calculate_temperature_impact(base_time_minutes, temp_increase_f):
    """
    Calculate performance degradation due to temperature increase

    Research-based temperature impact on marathon performance:
    - Optimal temp: 45-55°F
    - Every 5°F above 55°F adds ~1.5-2.5% to finish time
    - 10°F increase typically adds 3-5% to finish time

    Using conservative estimate: 3.5% for 10°F increase

    Source: Research by Matthew Ely et al. (2007, 2008) on temperature
    effects in marathon running
    """

    # Temperature impact factor
    # 10°F increase = ~3.5% slower (conservative estimate)
    impact_percent = (temp_increase_f / 10.0) * 3.5

    # Calculate adjusted time
    adjusted_time = base_time_minutes * (1 + impact_percent / 100.0)

    return adjusted_time, impact_percent

def format_time(minutes):
    """Format minutes to H:MM"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}:{mins:02d}"

def main():
    print("="*80)
    print("Temperature Impact on Marathon Predictions")
    print("="*80)

    # Base predictions (from holdout validation)
    predictions = [
        {
            'runner': 'You',
            'race': 'Dec 2024 Marathon',
            'base_prediction': 193.9,  # 3:13
            'actual': 202.8  # 3:22
        },
        {
            'runner': 'Salman',
            'race': 'Jack & Jill 2025',
            'base_prediction': 174.5,  # 2:54
            'actual': 175.4  # 2:55
        },
        {
            'runner': 'Azeem',
            'race': 'Houston Marathon 2026',
            'base_prediction': 205.6,  # 3:25
            'actual': 202.0  # 3:22
        }
    ]

    temp_increase = 10  # +10°F warmer

    print(f"\nScenario: Race temperature is +{temp_increase}°F warmer than optimal")
    print(f"\nResearch basis:")
    print(f"  • Optimal marathon temperature: 45-55°F")
    print(f"  • Every 5°F above optimal adds ~1.5-2.5% to finish time")
    print(f"  • +10°F typically adds 3-5% to finish time")
    print(f"  • Using conservative estimate: 3.5% for +10°F")
    print()

    print("="*80)
    print("Adjusted Predictions with +10°F Warmer Weather")
    print("="*80)

    for i, pred in enumerate(predictions, 1):
        adjusted_time, impact_percent = calculate_temperature_impact(
            pred['base_prediction'],
            temp_increase
        )

        time_added = adjusted_time - pred['base_prediction']

        print(f"\n{i}. {pred['runner']}'s Marathon ({pred['race']})")
        print(f"   Base prediction (normal temp):  {format_time(pred['base_prediction'])}")
        print(f"   Adjusted (+10°F warmer):        {format_time(adjusted_time)}")
        print(f"   Time added by heat:             +{time_added:.1f} minutes")
        print(f"   Impact:                         +{impact_percent:.1f}%")
        print(f"   Actual finish time:             {format_time(pred['actual'])}")

        # Compare adjusted to actual
        adjusted_error = abs(adjusted_time - pred['actual'])
        print(f"   Error (adjusted vs actual):     {adjusted_error:.1f} minutes")

    # Summary
    print("\n" + "="*80)
    print("Summary: Heat Impact on Performance")
    print("="*80)

    total_time_added = sum([
        calculate_temperature_impact(p['base_prediction'], temp_increase)[0] - p['base_prediction']
        for p in predictions
    ])
    avg_time_added = total_time_added / len(predictions)

    print(f"\nAverage time added by +10°F heat: {avg_time_added:.1f} minutes")
    print(f"Range: {calculate_temperature_impact(predictions[1]['base_prediction'], temp_increase)[0] - predictions[1]['base_prediction']:.1f} - {calculate_temperature_impact(predictions[0]['base_prediction'], temp_increase)[0] - predictions[0]['base_prediction']:.1f} minutes")

    print("\nKey Insights:")
    print("  • Faster runners lose more absolute time to heat")
    print("    (same % impact, but more minutes)")
    print("  • +10°F can cost 6-7 minutes for a 3:20 marathoner")
    print("  • Heat acclimatization can reduce impact by 30-50%")
    print("  • Proper hydration and pacing are critical in heat")

    print("\n" + "="*80)
    print("Temperature Ranges and Expected Impact")
    print("="*80)

    temp_ranges = [
        ("45-55°F", "Optimal", "0%", "Baseline"),
        ("55-60°F", "Good", "0-1%", "Minimal impact"),
        ("60-65°F", "Warm", "1-2%", "2-4 min for 3:20"),
        ("65-70°F", "Hot", "2-4%", "4-8 min for 3:20"),
        ("70-75°F", "Very Hot", "4-6%", "8-12 min for 3:20"),
        ("75-80°F", "Extreme", "6-10%", "12-20 min for 3:20"),
        ("80°F+", "Dangerous", "10-15%+", "20-30+ min for 3:20")
    ]

    print(f"\n{'Temperature':<15} {'Conditions':<12} {'Impact':<10} {'Example (3:20 base)'}")
    print("-" * 70)
    for temp, cond, impact, example in temp_ranges:
        print(f"{temp:<15} {cond:<12} {impact:<10} {example}")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
