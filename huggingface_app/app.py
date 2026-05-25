#!/usr/bin/env python3
"""
Strava Marathon Predictor - Hugging Face Spaces App
Accepts pre-extracted features.json OR activities.csv for prediction
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from io import StringIO
import statistics

# Load model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

MODEL = model_data['model']
FEATURE_NAMES = model_data['feature_names']


def parse_strava_csv(csv_content):
    """Parse Strava activities.csv"""
    df = pd.read_csv(StringIO(csv_content))

    activities = []
    for _, row in df.iterrows():
        if row.get('Activity Type') != 'Run':
            continue

        try:
            distance_m = float(row.get('Distance.1', 0) or 0)
            if distance_m > 0:
                distance_miles = distance_m / 1609.34
            else:
                distance_val = float(row.get('Distance', 0) or 0)
                distance_miles = distance_val / 1609.34 if distance_val > 100 else distance_val

            if distance_miles < 0.5:
                continue

            moving_sec = float(row.get('Moving Time', 0) or 0)
            elapsed_sec = float(row.get('Elapsed Time', 0) or 0)
            duration_min = (moving_sec if moving_sec > 0 else elapsed_sec) / 60

            date_str = row.get('Activity Date', '')
            try:
                activity_date = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
            except:
                try:
                    activity_date = datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
                except:
                    continue

            avg_hr = float(row.get('Average Heart Rate', 0) or 0) or None
            pace = duration_min / distance_miles if distance_miles > 0 else None

            activities.append({
                'date': activity_date,
                'distance_miles': distance_miles,
                'duration_min': duration_min,
                'avg_hr': avg_hr,
                'pace': pace,
            })
        except:
            continue

    return activities


def extract_csv_features(activities, race_date):
    """Extract features from CSV activities"""
    lookback_start = race_date - timedelta(weeks=16)
    taper_start = race_date - timedelta(days=7)

    training = [a for a in activities if lookback_start <= a['date'] < taper_start]

    if not training:
        return None, lookback_start, taper_start

    total_distance = sum(a['distance_miles'] for a in training)
    total_runs = len(training)

    dates = [a['date'] for a in training]
    weeks = max(1, (max(dates) - min(dates)).days / 7)

    weekly_mileage = total_distance / weeks
    runs_per_week = total_runs / weeks

    long_runs = [a for a in training if a['distance_miles'] >= 15]
    long_run_distance = max([a['distance_miles'] for a in long_runs], default=0)

    weekly_distances = {}
    for a in training:
        week_num = a['date'].isocalendar()[1]
        weekly_distances[week_num] = weekly_distances.get(week_num, 0) + a['distance_miles']

    peak_weekly_mileage = max(weekly_distances.values()) if weekly_distances else 0

    mileage_values = list(weekly_distances.values())
    if len(mileage_values) > 1:
        mileage_consistency = 1 - (statistics.stdev(mileage_values) / statistics.mean(mileage_values)) if statistics.mean(mileage_values) > 0 else 0
    else:
        mileage_consistency = 1.0

    tempo_runs = [a for a in training if a['pace'] and 7.0 <= a['pace'] < 8.0]
    fast_runs = [a for a in training if a['pace'] and a['pace'] < 7.5]
    quality_runs = [a for a in training if a['pace'] and a['pace'] < 8.0]

    hr_activities = [a for a in training if a['avg_hr']]
    avg_hr = sum(a['avg_hr'] for a in hr_activities) / len(hr_activities) if hr_activities else 0

    return {
        'total_weekly_mileage': round(weekly_mileage, 2),
        'peak_weekly_mileage': round(peak_weekly_mileage, 2),
        'runs_per_week': round(runs_per_week, 2),
        'total_runs': total_runs,
        'long_run_distance': round(long_run_distance, 2),
        'long_run_count': len(long_runs),
        'long_run_percent_weekly': round(long_run_distance / weekly_mileage * 100, 1) if weekly_mileage > 0 else 0,
        'mileage_consistency': round(max(0, min(1, mileage_consistency)), 3),
        'training_consistency_score': round(max(0, min(1, mileage_consistency)), 3),
        'tempo_workout_count': len(tempo_runs),
        'fast_workout_count': len(fast_runs),
        'quality_workout_percent': round(100 * len(quality_runs) / total_runs, 1) if total_runs > 0 else 0,
        'avg_hr': avg_hr,
        'hr_at_easy_pace': avg_hr,
        'hr_at_marathon_pace': avg_hr,
    }, lookback_start, taper_start


def format_time(minutes):
    """Format minutes as H:MM:SS"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes % 1) * 60)
    return f"{hours}:{mins:02d}:{secs:02d}"


def predict_from_features(
    features_file,
    age,
    sex,
    historical_pr_hours,
    historical_pr_minutes,
    temperature,
    humidity
):
    """Predict from pre-extracted features.json file"""

    if features_file is None:
        return "Please upload a features.json file", "", ""

    try:
        with open(features_file.name, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading features.json: {str(e)}", "", ""

    extracted_features = data.get('features', {})
    race_date = data.get('race_date', 'Unknown')
    training_window = data.get('training_window', {})
    extraction_info = data.get('extraction_info', {})

    # Prepare runner context
    age_normalized = 1 - (abs(age - 30) / 50)
    historical_pr = None
    if historical_pr_hours > 0 or historical_pr_minutes > 0:
        historical_pr = historical_pr_hours * 60 + historical_pr_minutes

    # Build full feature set
    features = {name: 0 for name in FEATURE_NAMES}

    # Copy extracted features
    for k, v in extracted_features.items():
        if k in features:
            features[k] = v or 0

    # Add runner context
    features['age_normalized'] = age_normalized
    features['sex_encoded'] = 1 if sex == "Male" else 0
    features['max_hr_normalized'] = 0.9
    features['experience_years'] = 3
    features['historical_pr_minutes'] = historical_pr or 0

    # Add weather
    features['race_temperature'] = temperature
    features['race_humidity'] = humidity / 100
    features['race_apparent_temperature'] = temperature
    features['race_wind_speed'] = 5

    # Add defaults
    features['race_distance_miles'] = 26.2
    features['elevation_tolerance'] = 1.0
    features['taper_quality_score'] = 0.5
    features['days_since_last_hard_effort'] = 7

    # Create feature vector and predict
    X = np.array([[features.get(name, 0) or 0 for name in FEATURE_NAMES]])
    prediction = MODEL.predict(X)[0]

    # Format results
    std_estimate = 10
    prediction_str = f"**Predicted Marathon Time: {format_time(prediction)}**"
    range_str = f"Confidence Range: {format_time(prediction - std_estimate)} - {format_time(prediction + std_estimate)}"

    # Data source info
    csv_count = extraction_info.get('csv_activities', 0)
    fit_count = extraction_info.get('fit_activities', 0)
    data_source = f"CSV ({csv_count})"
    if fit_count > 0:
        data_source = f"Hybrid (CSV: {csv_count}, FIT: {fit_count})"

    insights = f"""
### Training Summary
**Race Date:** {race_date}
**Training Window:** {training_window.get('start', 'N/A')} to {training_window.get('end', 'N/A')}
**Data Source:** {data_source}

| Metric | Value |
|--------|-------|
| Weekly Mileage | {extracted_features.get('total_weekly_mileage', 0):.1f} miles |
| Peak Week | {extracted_features.get('peak_weekly_mileage', 0):.1f} miles |
| Runs per Week | {extracted_features.get('runs_per_week', 0):.1f} |
| Total Runs | {extracted_features.get('total_runs', 0)} |
| Long Runs (15+ mi) | {extracted_features.get('long_run_count', 0)} |
| Longest Run | {extracted_features.get('long_run_distance', 0):.1f} miles |
| Tempo Workouts | {extracted_features.get('tempo_workout_count', 0)} |
| Quality Run % | {extracted_features.get('quality_workout_percent', 0):.1f}% |
"""

    if extracted_features.get('avg_hr', 0) > 0:
        insights += f"| Avg Heart Rate | {extracted_features['avg_hr']:.0f} bpm |\n"

    insights += f"""
### Race Conditions
- Temperature: {temperature}°F
- Humidity: {humidity}%
"""

    if historical_pr:
        insights += f"- Historical PR: {format_time(historical_pr)}\n"

    return prediction_str, range_str, insights


def predict_from_csv(
    csv_file,
    race_date_str,
    age,
    sex,
    historical_pr_hours,
    historical_pr_minutes,
    temperature,
    humidity
):
    """Predict from activities.csv file (simpler, faster)"""

    if csv_file is None:
        return "Please upload an activities.csv file", "", ""

    try:
        race_date = datetime.strptime(race_date_str, "%Y-%m-%d")
    except:
        return "Error: Invalid race date format. Use YYYY-MM-DD", "", ""

    try:
        with open(csv_file.name, 'r') as f:
            csv_content = f.read()
        activities = parse_strava_csv(csv_content)
    except Exception as e:
        return f"Error reading CSV: {str(e)}", "", ""

    if not activities:
        return "Error: No running activities found in the CSV", "", ""

    # Extract features
    result = extract_csv_features(activities, race_date)
    if result[0] is None:
        return "Error: No training activities found in the 4-month window before race date", "", ""

    extracted_features, lookback_start, taper_start = result

    # Prepare runner context
    age_normalized = 1 - (abs(age - 30) / 50)
    historical_pr = None
    if historical_pr_hours > 0 or historical_pr_minutes > 0:
        historical_pr = historical_pr_hours * 60 + historical_pr_minutes

    # Build full feature set
    features = {name: 0 for name in FEATURE_NAMES}

    for k, v in extracted_features.items():
        if k in features:
            features[k] = v or 0

    features['age_normalized'] = age_normalized
    features['sex_encoded'] = 1 if sex == "Male" else 0
    features['max_hr_normalized'] = 0.9
    features['experience_years'] = 3
    features['historical_pr_minutes'] = historical_pr or 0
    features['race_temperature'] = temperature
    features['race_humidity'] = humidity / 100
    features['race_apparent_temperature'] = temperature
    features['race_wind_speed'] = 5
    features['race_distance_miles'] = 26.2
    features['elevation_tolerance'] = 1.0
    features['taper_quality_score'] = 0.5
    features['days_since_last_hard_effort'] = 7

    # Predict
    X = np.array([[features.get(name, 0) or 0 for name in FEATURE_NAMES]])
    prediction = MODEL.predict(X)[0]

    std_estimate = 10
    prediction_str = f"**Predicted Marathon Time: {format_time(prediction)}**"
    range_str = f"Confidence Range: {format_time(prediction - std_estimate)} - {format_time(prediction + std_estimate)}"

    insights = f"""
### Training Summary
**Race Date:** {race_date_str}
**Training Window:** {lookback_start.strftime('%Y-%m-%d')} to {taper_start.strftime('%Y-%m-%d')}
**Data Source:** CSV only ({len(activities)} total runs)

| Metric | Value |
|--------|-------|
| Weekly Mileage | {extracted_features['total_weekly_mileage']:.1f} miles |
| Peak Week | {extracted_features['peak_weekly_mileage']:.1f} miles |
| Runs per Week | {extracted_features['runs_per_week']:.1f} |
| Total Runs | {extracted_features['total_runs']} |
| Long Runs (15+ mi) | {extracted_features['long_run_count']} |
| Longest Run | {extracted_features['long_run_distance']:.1f} miles |
| Tempo Workouts | {extracted_features['tempo_workout_count']} |
| Quality Run % | {extracted_features['quality_workout_percent']:.1f}% |
"""

    if extracted_features.get('avg_hr', 0) > 0:
        insights += f"| Avg Heart Rate | {extracted_features['avg_hr']:.0f} bpm |\n"

    insights += f"""
### Race Conditions
- Temperature: {temperature}°F
- Humidity: {humidity}%
"""

    if historical_pr:
        insights += f"- Historical PR: {format_time(historical_pr)}\n"

    # Warnings
    warnings = []
    if extracted_features['total_weekly_mileage'] < 30:
        warnings.append("⚠️ Low weekly mileage")
    if extracted_features['long_run_count'] < 3:
        warnings.append("⚠️ Few long runs")
    if humidity > 80:
        warnings.append("⚠️ High humidity may slow you down")

    if warnings:
        insights += "\n### Warnings\n" + "\n".join(f"- {w}" for w in warnings)

    return prediction_str, range_str, insights


# Create Gradio interface with tabs
with gr.Blocks(title="Marathon Time Predictor") as app:
    gr.Markdown("""
    # 🏃 Marathon Time Predictor

    Predict your marathon finish time based on your Strava training data.

    **Two options:**
    1. **Quick (CSV only):** Upload just your `activities.csv` file
    2. **Full (with FIT data):** Run the extraction script locally, then upload `features.json`
    """)

    with gr.Tabs():
        # Tab 1: Quick CSV upload
        with gr.TabItem("📊 Quick (CSV Only)"):
            gr.Markdown("""
            ### Upload activities.csv

            Extract `activities.csv` from your Strava export zip and upload it here.
            This is fast but doesn't include detailed HR data from FIT files.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    csv_file = gr.File(label="activities.csv", file_types=[".csv"])
                    csv_race_date = gr.Textbox(
                        label="Race Date (YYYY-MM-DD)",
                        value=datetime.now().strftime("%Y-%m-%d")
                    )

                    gr.Markdown("### Runner Profile")
                    with gr.Row():
                        csv_age = gr.Number(label="Age", value=35, minimum=18, maximum=80)
                        csv_sex = gr.Radio(["Male", "Female"], label="Sex", value="Male")

                    gr.Markdown("### Historical PR (optional)")
                    with gr.Row():
                        csv_pr_hours = gr.Number(label="Hours", value=0, minimum=0, maximum=6)
                        csv_pr_minutes = gr.Number(label="Minutes", value=0, minimum=0, maximum=59)

                    gr.Markdown("### Race Day Weather")
                    with gr.Row():
                        csv_temp = gr.Slider(label="Temperature (°F)", minimum=30, maximum=90, value=55)
                        csv_humidity = gr.Slider(label="Humidity (%)", minimum=20, maximum=100, value=60)

                    csv_predict_btn = gr.Button("Predict Marathon Time", variant="primary", size="lg")

                with gr.Column(scale=1):
                    csv_prediction = gr.Markdown()
                    csv_range = gr.Markdown()
                    csv_insights = gr.Markdown()

            csv_predict_btn.click(
                fn=predict_from_csv,
                inputs=[csv_file, csv_race_date, csv_age, csv_sex, csv_pr_hours, csv_pr_minutes, csv_temp, csv_humidity],
                outputs=[csv_prediction, csv_range, csv_insights]
            )

        # Tab 2: Features.json upload
        with gr.TabItem("🔬 Full (Pre-extracted Features)"):
            gr.Markdown("""
            ### Upload features.json

            For more accurate predictions with FIT file data:

            1. Download the extraction script: [extract_my_features.py](https://github.com/your-repo/extract_my_features.py)
            2. Run locally: `python extract_my_features.py your_strava_export.zip --race-date 2026-10-15`
            3. Upload the generated `features.json` here

            This includes detailed HR data from FIT files for better accuracy.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    features_file = gr.File(label="features.json", file_types=[".json"])

                    gr.Markdown("### Runner Profile")
                    with gr.Row():
                        feat_age = gr.Number(label="Age", value=35, minimum=18, maximum=80)
                        feat_sex = gr.Radio(["Male", "Female"], label="Sex", value="Male")

                    gr.Markdown("### Historical PR (optional)")
                    with gr.Row():
                        feat_pr_hours = gr.Number(label="Hours", value=0, minimum=0, maximum=6)
                        feat_pr_minutes = gr.Number(label="Minutes", value=0, minimum=0, maximum=59)

                    gr.Markdown("### Race Day Weather")
                    with gr.Row():
                        feat_temp = gr.Slider(label="Temperature (°F)", minimum=30, maximum=90, value=55)
                        feat_humidity = gr.Slider(label="Humidity (%)", minimum=20, maximum=100, value=60)

                    feat_predict_btn = gr.Button("Predict Marathon Time", variant="primary", size="lg")

                with gr.Column(scale=1):
                    feat_prediction = gr.Markdown()
                    feat_range = gr.Markdown()
                    feat_insights = gr.Markdown()

            feat_predict_btn.click(
                fn=predict_from_features,
                inputs=[features_file, feat_age, feat_sex, feat_pr_hours, feat_pr_minutes, feat_temp, feat_humidity],
                outputs=[feat_prediction, feat_range, feat_insights]
            )

    gr.Markdown("""
    ---
    ### About This Model

    **Hybrid model** trained on 68 marathon races from 6 runners using Random Forest.

    **Model Performance:**
    - Cross-validation MAE: ~15 minutes
    - Holdout validation MAE: ~5 minutes

    **Limitations:**
    - May underpredict very fast runners (sub-3:00)
    - Weather sensitivity varies by individual
    """)


if __name__ == "__main__":
    app.launch()
