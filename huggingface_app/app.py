#!/usr/bin/env python3
"""
Strava Marathon Predictor - Hugging Face Spaces App
Predicts marathon finish time based on Strava training data
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import zipfile
import tempfile
import os
from datetime import datetime, timedelta
from io import StringIO
import statistics

# Load model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

MODEL = model_data['model']
FEATURE_NAMES = model_data['feature_names']


def parse_strava_csv(csv_content):
    """Parse Strava activities.csv content"""
    df = pd.read_csv(StringIO(csv_content))

    activities = []
    for _, row in df.iterrows():
        if row.get('Activity Type') != 'Run':
            continue

        try:
            # Parse distance
            distance_m = float(row.get('Distance', 0) or 0)
            distance_miles = distance_m / 1609.34
            if distance_miles < 0.5:
                continue

            # Parse time
            moving_sec = float(row.get('Moving Time', 0) or 0)
            elapsed_sec = float(row.get('Elapsed Time', 0) or 0)
            duration_min = (moving_sec if moving_sec > 0 else elapsed_sec) / 60

            # Parse date
            date_str = row.get('Activity Date', '')
            try:
                activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
            except:
                try:
                    activity_date = datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
                except:
                    continue

            # Parse HR
            avg_hr = float(row.get('Average Heart Rate', 0) or 0) or None

            # Calculate pace
            pace = duration_min / distance_miles if distance_miles > 0 else None

            activities.append({
                'date': activity_date,
                'distance_miles': distance_miles,
                'duration_min': duration_min,
                'avg_hr': avg_hr,
                'pace': pace,
                'elevation_gain': float(row.get('Elevation Gain', 0) or 0),
            })
        except Exception as e:
            continue

    return activities


def extract_features(activities, race_date, race_distance, runner_context, weather):
    """Extract features from training activities"""
    features = {name: 0 for name in FEATURE_NAMES}

    if not activities:
        return features

    # Filter to training window (4 months before race, excluding last week taper)
    lookback_start = race_date - timedelta(weeks=16)
    taper_start = race_date - timedelta(days=7)

    training = [a for a in activities
                if lookback_start <= a['date'] < taper_start]

    if not training:
        return features

    # Basic volume features
    total_distance = sum(a['distance_miles'] for a in training)
    total_runs = len(training)

    dates = [a['date'] for a in training]
    weeks = max(1, (max(dates) - min(dates)).days / 7)

    weekly_mileage = total_distance / weeks
    runs_per_week = total_runs / weeks

    # Long runs (15+ miles)
    long_runs = [a for a in training if a['distance_miles'] >= 15]
    long_run_distance = max([a['distance_miles'] for a in long_runs], default=0)

    # Weekly breakdown for peak and consistency
    weekly_distances = {}
    for a in training:
        week_num = a['date'].isocalendar()[1]
        weekly_distances[week_num] = weekly_distances.get(week_num, 0) + a['distance_miles']

    peak_weekly_mileage = max(weekly_distances.values()) if weekly_distances else 0

    # Mileage consistency
    mileage_values = list(weekly_distances.values())
    if len(mileage_values) > 1:
        mileage_consistency = 1 - (statistics.stdev(mileage_values) / statistics.mean(mileage_values)) if statistics.mean(mileage_values) > 0 else 0
    else:
        mileage_consistency = 1.0

    # Quality workouts (tempo < 8:00 pace, fast < 7:30 pace)
    tempo_runs = [a for a in training if a['pace'] and 7.0 <= a['pace'] < 8.0]
    fast_runs = [a for a in training if a['pace'] and a['pace'] < 7.5]
    quality_runs = [a for a in training if a['pace'] and a['pace'] < 8.0]

    # HR features
    hr_activities = [a for a in training if a['avg_hr']]
    avg_hr = sum(a['avg_hr'] for a in hr_activities) / len(hr_activities) if hr_activities else None

    # Populate features
    features['race_distance_miles'] = race_distance
    features['total_weekly_mileage'] = round(weekly_mileage, 2)
    features['peak_weekly_mileage'] = round(peak_weekly_mileage, 2)
    features['long_run_distance'] = round(long_run_distance, 2)
    features['long_run_count'] = len(long_runs)
    features['total_runs'] = total_runs
    features['runs_per_week'] = round(runs_per_week, 2)
    features['mileage_consistency'] = round(max(0, min(1, mileage_consistency)), 3)
    features['long_run_percent_weekly'] = round(long_run_distance / weekly_mileage * 100, 1) if weekly_mileage > 0 else 0

    # Runner context
    features['age_normalized'] = runner_context.get('age_normalized', 0.9)
    features['sex_encoded'] = runner_context.get('sex_encoded', 1)
    features['max_hr_normalized'] = runner_context.get('max_hr_normalized', 0.9)
    features['experience_years'] = runner_context.get('experience_years', 3)
    features['training_consistency_score'] = features['mileage_consistency']

    # Quality workouts
    features['tempo_workout_count'] = len(tempo_runs)
    features['fast_workout_count'] = len(fast_runs)
    features['quality_workout_percent'] = round(100 * len(quality_runs) / total_runs, 1) if total_runs > 0 else 0

    # HR features
    if avg_hr:
        features['avg_hr'] = avg_hr
        features['hr_at_easy_pace'] = avg_hr
        features['hr_at_marathon_pace'] = avg_hr

    # Weather
    features['race_temperature'] = weather.get('temperature', 50)
    features['race_humidity'] = weather.get('humidity', 0.6)
    features['race_apparent_temperature'] = weather.get('temperature', 50)
    features['race_wind_speed'] = weather.get('wind_speed', 5)

    # Historical PR
    features['historical_pr_minutes'] = runner_context.get('historical_pr', None)

    # Defaults for unused features
    features['elevation_tolerance'] = 1.0
    features['taper_quality_score'] = 0.5
    features['days_since_last_hard_effort'] = 7

    return features


def format_time(minutes):
    """Format minutes as H:MM:SS"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes % 1) * 60)
    return f"{hours}:{mins:02d}:{secs:02d}"


def predict_marathon(
    strava_zip,
    race_date_str,
    age,
    sex,
    historical_pr_hours,
    historical_pr_minutes,
    temperature,
    humidity
):
    """Main prediction function"""

    try:
        # Parse race date
        race_date = datetime.strptime(race_date_str, "%Y-%m-%d")
    except:
        return "Error: Invalid race date format. Use YYYY-MM-DD", "", ""

    # Extract activities from zip
    activities = []
    try:
        with zipfile.ZipFile(strava_zip.name, 'r') as z:
            # Find activities.csv
            csv_file = None
            for name in z.namelist():
                if name.endswith('activities.csv'):
                    csv_file = name
                    break

            if not csv_file:
                return "Error: No activities.csv found in zip file", "", ""

            with z.open(csv_file) as f:
                csv_content = f.read().decode('utf-8')
                activities = parse_strava_csv(csv_content)
    except Exception as e:
        return f"Error reading zip file: {str(e)}", "", ""

    if not activities:
        return "Error: No running activities found in the data", "", ""

    # Prepare runner context
    age_normalized = 1 - (abs(age - 30) / 50)  # Peak at 30
    historical_pr = None
    if historical_pr_hours > 0 or historical_pr_minutes > 0:
        historical_pr = historical_pr_hours * 60 + historical_pr_minutes

    runner_context = {
        'age_normalized': age_normalized,
        'sex_encoded': 1 if sex == "Male" else 0,
        'max_hr_normalized': 0.9,
        'experience_years': 3,
        'historical_pr': historical_pr,
    }

    weather = {
        'temperature': temperature,
        'humidity': humidity / 100,  # Convert to decimal
        'wind_speed': 5,
    }

    # Extract features
    features = extract_features(activities, race_date, 26.2, runner_context, weather)

    # Create feature vector
    X = np.array([[features.get(name, 0) or 0 for name in FEATURE_NAMES]])

    # Predict
    prediction = MODEL.predict(X)[0]

    # Calculate confidence range (using model's training std)
    std_estimate = 12  # ~12 min standard deviation based on our CV
    low_estimate = prediction - std_estimate
    high_estimate = prediction + std_estimate

    # Format results
    prediction_str = f"**Predicted Marathon Time: {format_time(prediction)}**"

    range_str = f"Confidence Range: {format_time(low_estimate)} - {format_time(high_estimate)}"

    # Training insights
    insights = f"""
### Training Summary (4 months before race)

| Metric | Value |
|--------|-------|
| Weekly Mileage | {features['total_weekly_mileage']:.1f} miles |
| Peak Week | {features['peak_weekly_mileage']:.1f} miles |
| Runs per Week | {features['runs_per_week']:.1f} |
| Long Runs (15+ mi) | {features['long_run_count']} |
| Longest Run | {features['long_run_distance']:.1f} miles |
| Tempo Workouts | {features['tempo_workout_count']} |
| Quality Run % | {features['quality_workout_percent']:.1f}% |
| Consistency Score | {features['mileage_consistency']:.2f} |

### Race Conditions
- Temperature: {temperature}°F
- Humidity: {humidity}%
"""

    if historical_pr:
        insights += f"- Historical PR: {format_time(historical_pr)}\n"

    # Add warnings
    warnings = []
    if features['total_weekly_mileage'] < 30:
        warnings.append("Low weekly mileage - consider increasing training volume")
    if features['long_run_count'] < 3:
        warnings.append("Few long runs - more 15+ mile runs recommended")
    if humidity > 80:
        warnings.append("High humidity may slow you down 5-15 minutes")
    if temperature > 65:
        warnings.append("Warm conditions - expect slower times")

    if warnings:
        insights += "\n### Warnings\n"
        for w in warnings:
            insights += f"- {w}\n"

    return prediction_str, range_str, insights


# Create Gradio interface
with gr.Blocks(title="Marathon Time Predictor") as app:
    gr.Markdown("""
    # Marathon Time Predictor

    Upload your Strava data export to predict your marathon finish time based on your training.

    ## How to get your Strava data:
    1. Go to [Strava Settings](https://www.strava.com/settings/profile)
    2. Click "Download or Delete Your Account"
    3. Click "Request Your Archive"
    4. Wait for email, download the zip file
    5. Upload the zip file here
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Data")
            strava_file = gr.File(
                label="Strava Export (zip file)",
                file_types=[".zip"]
            )

            gr.Markdown("### Race Details")
            race_date = gr.Textbox(
                label="Race Date (YYYY-MM-DD)",
                placeholder="2026-10-15",
                value=datetime.now().strftime("%Y-%m-%d")
            )

            gr.Markdown("### Runner Profile")
            with gr.Row():
                age = gr.Number(label="Age", value=35, minimum=18, maximum=80)
                sex = gr.Radio(["Male", "Female"], label="Sex", value="Male")

            gr.Markdown("### Historical PR (optional)")
            with gr.Row():
                pr_hours = gr.Number(label="Hours", value=0, minimum=0, maximum=6)
                pr_minutes = gr.Number(label="Minutes", value=0, minimum=0, maximum=59)

            gr.Markdown("### Expected Race Day Weather")
            with gr.Row():
                temperature = gr.Slider(
                    label="Temperature (°F)",
                    minimum=30, maximum=90, value=55, step=1
                )
                humidity = gr.Slider(
                    label="Humidity (%)",
                    minimum=20, maximum=100, value=60, step=5
                )

            predict_btn = gr.Button("Predict Marathon Time", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Prediction Results")
            prediction_output = gr.Markdown(label="Predicted Time")
            range_output = gr.Markdown(label="Confidence Range")
            insights_output = gr.Markdown(label="Training Insights")

    predict_btn.click(
        fn=predict_marathon,
        inputs=[
            strava_file, race_date, age, sex,
            pr_hours, pr_minutes, temperature, humidity
        ],
        outputs=[prediction_output, range_output, insights_output]
    )

    gr.Markdown("""
    ---
    ### About This Model

    This model was trained on marathon data from 5 runners (37 races total) using a Random Forest algorithm.

    **Key Features Used:**
    - Weekly mileage and peak mileage
    - Number and quality of tempo/speed workouts
    - Long run frequency and distance
    - Training consistency
    - Historical PR (if available)
    - Race day weather conditions

    **Model Performance:**
    - Mean Absolute Error: ~6 minutes on holdout validation
    - Works best for runners training 40-70 miles/week

    **Limitations:**
    - Limited training data (37 races)
    - May underpredict very fast runners (sub-3:00)
    - Weather sensitivity varies by individual

    Built with Strava data + scikit-learn + Gradio
    """)


if __name__ == "__main__":
    app.launch()
