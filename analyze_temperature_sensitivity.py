#!/usr/bin/env python3
"""
Analyze temperature sensitivity for each runner
"""

import json
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

def predict_with_temp(model, feature_names, features, temp):
    """Predict with specific temperature"""
    feat_copy = features.copy()
    feat_copy['race_temperature'] = temp
    feat_copy['race_apparent_temperature'] = temp
    values = [feat_copy.get(k, 0) or 0 for k in feature_names]
    return model.predict([values])[0]

def format_time(minutes):
    return f"{int(minutes//60)}:{int(minutes%60):02d}"

def main():
    print("="*80)
    print("Temperature Sensitivity Analysis by Runner")
    print("="*80)

    # Load model
    with open('race_time_model_with_weather.pkl', 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_names = model_data['feature_names']

    # Load dataset
    dataset_path = Path.home() / '.strava_guru_cache' / 'race_data' / 'combined_41_features.json'
    with open(dataset_path, 'r') as f:
        all_races = json.load(f)

    bonked_ids = ['marathon_20251012', 'marathon_20231008']
    clean_races = [r for r in all_races if r['race_id'] not in bonked_ids]

    # Get most recent races
    your_races = sorted([r for r in clean_races if r['runner_id'] == 'my_runner'],
                       key=lambda x: x['race_date'])
    salman_races = sorted([r for r in clean_races if r['runner_id'] == 'runner_2'],
                         key=lambda x: x['race_date'])
    azeem_races = sorted([r for r in clean_races if r['runner_id'] == 'runner_3'],
                        key=lambda x: x['race_date'])

    runners = [
        ('You', your_races[-1], 'blue'),
        ('Salman', salman_races[-1], 'red'),
        ('Azeem', azeem_races[-1], 'green')
    ]

    # Temperature range to test
    temps = np.arange(40, 81, 5)

    print("\nAnalyzing temperature sensitivity across 40-80°F range\n")

    # Calculate predictions for each runner at each temperature
    sensitivity_data = {}

    for runner_name, race, color in runners:
        predictions = []
        base_features = race['features']

        for temp in temps:
            pred = predict_with_temp(model, feature_names, base_features, temp)
            predictions.append(pred)

        sensitivity_data[runner_name] = {
            'temps': temps,
            'predictions': predictions,
            'color': color,
            'actual_time': race['actual_time_minutes']
        }

        # Calculate sensitivity metrics
        pred_at_50 = predictions[2]  # 50°F
        pred_at_70 = predictions[6]  # 70°F
        absolute_change = pred_at_70 - pred_at_50
        percent_change = (absolute_change / pred_at_50) * 100

        print(f"{runner_name}:")
        print(f"  Actual marathon time: {format_time(race['actual_time_minutes'])}")
        print(f"  Predicted at 50°F: {format_time(pred_at_50)}")
        print(f"  Predicted at 70°F: {format_time(pred_at_70)}")
        print(f"  Change (50→70°F): +{absolute_change:.1f} min (+{percent_change:.2f}%)")
        print(f"  Sensitivity: {absolute_change/20:.2f} min per °F")
        print()

    # Summary comparison
    print("="*80)
    print("Temperature Sensitivity Comparison")
    print("="*80)

    print(f"\n{'Runner':<10} {'Time @ 50°F':<15} {'Time @ 70°F':<15} {'Change':<12} {'% Change':<12} {'Min/°F'}")
    print("-" * 85)

    sensitivities = []
    for runner_name, race, _ in runners:
        data = sensitivity_data[runner_name]
        pred_50 = data['predictions'][2]
        pred_70 = data['predictions'][6]
        change = pred_70 - pred_50
        pct = (change / pred_50) * 100
        min_per_f = change / 20

        sensitivities.append((runner_name, min_per_f, pct))

        print(f"{runner_name:<10} {format_time(pred_50):<15} {format_time(pred_70):<15} "
              f"+{change:.1f} min    +{pct:.2f}%      {min_per_f:.2f}")

    # Rank by sensitivity
    sensitivities.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*80)
    print("Sensitivity Ranking (Most to Least Affected by Heat)")
    print("="*80)

    for i, (runner_name, min_per_f, pct) in enumerate(sensitivities, 1):
        print(f"{i}. {runner_name}: {min_per_f:.2f} min/°F (+{pct:.2f}% per 20°F)")

    # Check if differences are meaningful
    print("\n" + "="*80)
    print("Statistical Notes")
    print("="*80)

    min_sens = sensitivities[-1][1]
    max_sens = sensitivities[0][1]
    diff = max_sens - min_sens

    print(f"\nSensitivity range: {min_sens:.2f} - {max_sens:.2f} min/°F")
    print(f"Difference: {diff:.2f} min/°F")
    print(f"At 20°F increase: {diff * 20:.1f} minutes difference between most/least sensitive")

    if diff * 20 > 3:
        print(f"\n✓ Meaningful difference in heat sensitivity between runners")
        print(f"  The most heat-sensitive runner slows {diff * 20:.1f} min more than")
        print(f"  the least sensitive over a 20°F temperature increase")
    else:
        print(f"\n⚠ Small difference - may be within model uncertainty")

    # Explain why
    print("\n" + "="*80)
    print("Possible Explanations for Differences")
    print("="*80)

    # Count weather data points per runner
    weather_counts = {
        'You': len([r for r in your_races if r['features'].get('race_temperature')]),
        'Salman': len([r for r in salman_races if r['features'].get('race_temperature')]),
        'Azeem': len([r for r in azeem_races if r['features'].get('race_temperature')])
    }

    print(f"\nTraining data with weather:")
    for runner, count in weather_counts.items():
        print(f"  {runner}: {count} races")

    print(f"\nFactors that affect heat sensitivity:")
    print(f"  • Training data availability (more data = better learned sensitivity)")
    print(f"  • Actual physiological differences (heat acclimatization, body composition)")
    print(f"  • Pacing strategy in hot weather")
    print(f"  • Hydration habits")

    if weather_counts['Salman'] > weather_counts['You'] + weather_counts['Azeem']:
        print(f"\n→ Salman's sensitivity is better learned (10 races with weather data)")
        print(f"  Your and Azeem's sensitivities are based on fewer data points")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
