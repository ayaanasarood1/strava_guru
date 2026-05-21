#!/usr/bin/env python3
"""
Add weather features to the feature engineering pipeline
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class WeatherData:
    """Weather data for a race"""
    temperature: Optional[float] = None  # °F
    apparent_temperature: Optional[float] = None  # °F (feels-like)
    humidity: Optional[float] = None  # %
    wind_speed: Optional[float] = None  # mph
    weather_condition: Optional[str] = None

def extract_weather_from_csv(csv_path: Path, race_date: datetime) -> WeatherData:
    """
    Extract weather data for a specific race date from activities.csv
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # Parse activity date
                date_str = row.get('Activity Date', '')
                if not date_str:
                    continue

                activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")

                # Check if this is the race (same day, long distance)
                if activity_date.date() == race_date.date():
                    distance_m = float(row.get('Distance', 0) or 0)
                    distance_mi = distance_m / 1609.34

                    # Marathon distance check
                    if 26.0 <= distance_mi <= 26.5:
                        # Extract weather data
                        weather = WeatherData()

                        # Temperature (convert Celsius to Fahrenheit)
                        temp_str = row.get('Weather Temperature', '')
                        if temp_str:
                            try:
                                temp_c = float(temp_str)
                                weather.temperature = temp_c * 9/5 + 32  # C to F
                            except:
                                pass

                        # Apparent temperature (convert Celsius to Fahrenheit)
                        apparent_temp_str = row.get('Apparent Temperature', '')
                        if apparent_temp_str:
                            try:
                                apparent_c = float(apparent_temp_str)
                                weather.apparent_temperature = apparent_c * 9/5 + 32  # C to F
                            except:
                                pass

                        # Humidity (convert fraction to percentage)
                        humidity_str = row.get('Humidity', '')
                        if humidity_str:
                            try:
                                humidity_frac = float(humidity_str)
                                weather.humidity = humidity_frac * 100  # Fraction to %
                            except:
                                pass

                        # Wind speed (convert m/s to mph)
                        wind_str = row.get('Wind Speed', '')
                        if wind_str:
                            try:
                                wind_ms = float(wind_str)
                                weather.wind_speed = wind_ms * 2.237  # m/s to mph
                            except:
                                pass

                        weather.weather_condition = row.get('Weather Condition', '')

                        return weather
            except:
                continue

    return WeatherData()

def add_weather_to_dataset():
    """Add weather features to all races in the dataset"""
    print("="*80)
    print("Adding Weather Features to Dataset")
    print("="*80)

    # Load dataset
    dataset_path = Path.home() / '.strava_guru_cache' / 'race_data' / 'combined_41_features.json'

    with open(dataset_path, 'r') as f:
        all_races = json.load(f)

    print(f"\nLoaded {len(all_races)} races")

    # CSV paths for each runner
    csv_paths = {
        'my_runner': Path('/Users/osman/Downloads/export_40402578/activities.csv'),
        'runner_2': Path('/Users/osman/Downloads/export_1884062_salman/activities.csv'),
        'runner_3': Path('/Users/osman/Downloads/export_52983191_azeem/activities.csv')
    }

    updated_count = 0

    for race in all_races:
        runner_id = race['runner_id']
        race_date_str = race['race_date']
        race_name = race.get('_race_name', race_date_str[:10])

        # Parse race date
        race_date = datetime.fromisoformat(race_date_str.replace('T00:00:00', ''))

        # Get CSV path
        csv_path = csv_paths.get(runner_id)
        if not csv_path or not csv_path.exists():
            continue

        # Extract weather
        weather = extract_weather_from_csv(csv_path, race_date)

        # Add weather features to race
        if weather.temperature is not None:
            race['features']['race_temperature'] = weather.temperature
            race['features']['race_apparent_temperature'] = weather.apparent_temperature or weather.temperature
            race['features']['race_humidity'] = weather.humidity or 0.0
            race['features']['race_wind_speed'] = weather.wind_speed or 0.0

            updated_count += 1

            print(f"✓ {race_name:30s} - {weather.temperature:.1f}°F, Humidity: {weather.humidity or 0:.0f}%")
        else:
            # Set defaults if no weather data
            race['features']['race_temperature'] = None
            race['features']['race_apparent_temperature'] = None
            race['features']['race_humidity'] = None
            race['features']['race_wind_speed'] = None

            print(f"⚠ {race_name:30s} - No weather data")

    # Save updated dataset
    with open(dataset_path, 'w') as f:
        json.dump(all_races, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Added weather features to {updated_count}/{len(all_races)} races")
    print(f"✓ Saved to {dataset_path}")
    print(f"{'='*80}")

    # Summary statistics
    temps = [r['features']['race_temperature'] for r in all_races
             if r['features'].get('race_temperature') is not None]

    if temps:
        print(f"\nWeather Statistics:")
        print(f"  Temperature range: {min(temps):.1f}°F - {max(temps):.1f}°F")
        print(f"  Average temperature: {sum(temps)/len(temps):.1f}°F")

if __name__ == '__main__':
    add_weather_to_dataset()
