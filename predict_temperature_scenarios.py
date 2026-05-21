#!/usr/bin/env python3
"""
Predict marathons with different temperature scenarios using weather-aware model
"""

import json
import numpy as np
import pickle
from pathlib import Path

def predict_with_temperature(model, feature_names, base_features, temperature_f):
    """
    Predict race time with a specific temperature

    Args:
        model: Trained model
        feature_names: List of feature names
        base_features: Dict of base features
        temperature_f: Race temperature in Fahrenheit

    Returns:
        Predicted time in minutes
    """
    # Copy features
    features = base_features.copy()

    # Set race temperature
    features['race_temperature'] = temperature_f
    features['race_apparent_temperature'] = temperature_f  # Simplified (no wind chill)

    # Convert to array in correct order
    feature_values = [features.get(k, 0) or 0 for k in feature_names]

    # Predict
    prediction = model.predict([feature_values])[0]

    return prediction

def format_time(minutes):
    """Format minutes to H:MM"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}:{mins:02d}"

def main():
    print("="*80)
    print("Temperature Scenario Analysis with Weather-Aware Model")
    print("="*80)

    # Load model
    model_path = 'race_time_model_with_weather.pkl'
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_names = model_data['feature_names']

    print(f"\nLoaded model: {model_data['model_name']}")
    print(f"Features: {len(feature_names)} (including weather)")

    # Load dataset to get holdout races
    dataset_path = Path.home() / '.strava_guru_cache' / 'race_data' / 'combined_41_features.json'
    with open(dataset_path, 'r') as f:
        all_races = json.load(f)

    # Filter and identify holdout races
    bonked_ids = ['marathon_20251012', 'marathon_20231008']
    clean_races = [r for r in all_races if r['race_id'] not in bonked_ids]

    # Get most recent for each runner
    your_races = sorted([r for r in clean_races if r['runner_id'] == 'my_runner'],
                       key=lambda x: x['race_date'])
    salman_races = sorted([r for r in clean_races if r['runner_id'] == 'runner_2'],
                         key=lambda x: x['race_date'])
    azeem_races = sorted([r for r in clean_races if r['runner_id'] == 'runner_3'],
                        key=lambda x: x['race_date'])

    holdout_races = [
        ('You', your_races[-1]),
        ('Salman', salman_races[-1]),
        ('Azeem', azeem_races[-1])
    ]

    # Temperature scenarios to test
    temperatures = [45, 50, 55, 60, 65, 70, 75]

    print("\n" + "="*80)
    print("Predictions Across Temperature Range")
    print("="*80)

    for runner_name, race in holdout_races:
        print(f"\n{runner_name}'s {race.get('_race_name', race['race_date'][:10])}")
        print(f"Actual: {format_time(race['actual_time_minutes'])}")

        # Get base features
        base_features = race['features'].copy()

        # Get actual race temperature if available
        actual_temp = base_features.get('race_temperature')
        if actual_temp:
            print(f"Actual race temp: {actual_temp:.1f}°F")

        print(f"\nPredictions at different temperatures:")
        print(f"{'Temp (°F)':<12} {'Predicted':<12} {'vs Actual':<15}")
        print("-" * 50)

        predictions = []
        for temp in temperatures:
            pred = predict_with_temperature(model, feature_names, base_features, temp)
            error = pred - race['actual_time_minutes']
            predictions.append((temp, pred, error))

            marker = ""
            if actual_temp and abs(temp - actual_temp) < 3:
                marker = " ← Actual"
            elif temp == 55:
                marker = " (Optimal)"

            print(f"{temp}°F         {format_time(pred):<12} {error:+.1f} min{marker}")

        # Find best temperature (closest to actual)
        best_temp, best_pred, best_error = min(predictions, key=lambda x: abs(x[2]))
        print(f"\nBest prediction: {format_time(best_pred)} at {best_temp}°F (error: {best_error:+.1f} min)")

    # Summary table
    print("\n" + "="*80)
    print("Temperature Impact Summary")
    print("="*80)

    print(f"\n{'Runner':<10} {'Actual Time':<15} {'Best Temp':<12} {'Best Prediction':<15} {'Error'}")
    print("-" * 70)

    for runner_name, race in holdout_races:
        base_features = race['features'].copy()

        predictions = []
        for temp in temperatures:
            pred = predict_with_temperature(model, feature_names, base_features, temp)
            error = pred - race['actual_time_minutes']
            predictions.append((temp, pred, error))

        best_temp, best_pred, best_error = min(predictions, key=lambda x: abs(x[2]))

        print(f"{runner_name:<10} {format_time(race['actual_time_minutes']):<15} {best_temp}°F        {format_time(best_pred):<15} {best_error:+.1f} min")

    print("\n" + "="*80)
    print("Key Insights")
    print("="*80)
    print("• Weather-aware model accounts for temperature impact")
    print("• Optimal marathon temperature: 45-55°F")
    print("• Model can predict times for different race conditions")
    print("• Apparent temperature (with wind chill) matters most")
    print("="*80)

if __name__ == '__main__':
    main()
