#!/usr/bin/env python3
"""
Strava Marathon Predictor - Hugging Face Spaces App
Auto-detects marathon PR from CSV, uses unanchored model with PR decay
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from io import StringIO

# Load model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

MODEL = model_data['model']
FEATURE_NAMES = model_data['feature_names']


def is_actual_marathon_race(activity_name):
    """Check if activity name indicates an actual marathon race (not training run)"""
    if pd.isna(activity_name):
        return False

    name = str(activity_name).lower()

    # Exclude generic training run names
    training_patterns = ['morning run', 'afternoon run', 'evening run', 'lunch run',
                         'easy run', 'long run', 'recovery run']
    for pattern in training_patterns:
        if name == pattern or name.startswith(pattern + ' '):
            return False

    # Exclude trail runs (not road marathons)
    if 'trail' in name:
        return False

    # Exclude virtual runs and pacing runs
    if 'virtual' in name or 'pacing' in name or 'pacer' in name:
        return False

    # Include if contains 'marathon'
    if 'marathon' in name:
        return True

    # Include known race indicators
    race_patterns = [
        'cim', 'boston', 'nyc', 'chicago', 'berlin', 'london', 'tokyo',
        'jack & jill', 'jack and jill', 'napa', 'la marathon', 'l.a.',
        'bq', 'qualifier', 'official', 'race', 'sub-', 'sub 3',
        'vancouver', 'sf marathon', 'san francisco', 'long beach', 'honolulu'
    ]
    for pattern in race_patterns:
        if pattern in name:
            return True

    return False


def parse_strava_date(date_str):
    """Parse Strava activity date string"""
    try:
        return datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
    except:
        try:
            return datetime.strptime(date_str[:12].strip().rstrip(','), "%b %d, %Y")
        except:
            return None


def parse_num(x):
    """Parse number that might have commas"""
    if pd.isna(x):
        return 0
    try:
        return float(str(x).replace(',', ''))
    except:
        return 0


def detect_csv_format(df):
    """Detect if this is a raw Strava CSV or an enriched CSV"""
    columns = set(df.columns)
    if 'Activity Type' in columns and 'Distance' in columns:
        return 'strava_raw'
    elif 'distance_miles' in columns and 'duration_min' in columns:
        return 'enriched'
    else:
        return 'unknown'


def parse_strava_csv(csv_content):
    """Parse Strava activities.csv or enriched CSV and return activities + detected marathons"""
    df = pd.read_csv(StringIO(csv_content))

    csv_format = detect_csv_format(df)

    activities = []
    marathons = []

    for _, row in df.iterrows():
        try:
            if csv_format == 'strava_raw':
                # Raw Strava export format
                if row.get('Activity Type') != 'Run':
                    continue

                # Parse distance (handle commas in numbers)
                distance_m = parse_num(row.get('Distance.1', 0))
                if distance_m > 0:
                    distance_miles = distance_m / 1609.34
                else:
                    distance_val = parse_num(row.get('Distance', 0))
                    distance_miles = distance_val / 1609.34 if distance_val > 100 else distance_val

                # Parse duration (seconds -> minutes)
                moving_sec = parse_num(row.get('Moving Time', 0))
                elapsed_sec = parse_num(row.get('Elapsed Time', 0))
                duration_min = (moving_sec if moving_sec > 0 else elapsed_sec) / 60

                # Parse elevation
                elev_gain = parse_num(row.get('Elevation Gain', 0))
                elev_loss = parse_num(row.get('Elevation Loss', 0))
                net_elevation = elev_gain - elev_loss

                # Parse HR
                avg_hr = parse_num(row.get('Average Heart Rate', 0)) or None

                # Check if marathon using is_actual_marathon_race
                activity_name = row.get('Activity Name', '')
                is_marathon_distance = 25.0 <= distance_miles <= 27.5
                is_marathon = is_marathon_distance and is_actual_marathon_race(activity_name)

            elif csv_format == 'enriched':
                # Enriched CSV format (already processed)
                distance_miles = parse_num(row.get('distance_miles', 0))
                duration_min = parse_num(row.get('duration_min', 0))

                # Elevation from enriched CSV - try FIT data first, then CSV
                # Note: enriched CSVs may not have elevation loss, so net_elevation may be unreliable
                fit_gain_m = parse_num(row.get('fit_elevation_gain_m', 0))
                fit_loss_m = parse_num(row.get('fit_elevation_loss_m', 0))
                if fit_gain_m > 0 or fit_loss_m > 0:
                    # Convert meters to feet
                    net_elevation = (fit_gain_m - fit_loss_m) * 3.28084
                else:
                    # Fallback to CSV (only has gain, not loss - can't detect downhill)
                    net_elevation = 0  # Can't reliably calculate without loss data

                # HR from enriched CSV (prefer fit data)
                avg_hr = parse_num(row.get('fit_avg_hr', 0)) or parse_num(row.get('avg_hr', 0)) or None

                # Use is_marathon column if available, otherwise check distance
                activity_name = row.get('Activity Name', '')
                if 'is_marathon' in row:
                    is_marathon = row.get('is_marathon', False) == True
                    # Still filter for actual races
                    if is_marathon:
                        is_marathon = is_actual_marathon_race(activity_name)
                else:
                    is_marathon_distance = 25.0 <= distance_miles <= 27.5
                    is_marathon = is_marathon_distance and is_actual_marathon_race(activity_name)

            else:
                continue

            if distance_miles < 0.5:
                continue

            # Parse date
            date_str = row.get('Activity Date', '')
            activity_date = parse_strava_date(date_str)
            if not activity_date:
                continue

            pace = duration_min / distance_miles if distance_miles > 0 else None

            activity = {
                'date': activity_date,
                'name': activity_name,
                'distance_miles': distance_miles,
                'duration_min': duration_min,
                'avg_hr': avg_hr,
                'pace': pace,
                'net_elevation': net_elevation,
            }
            activities.append(activity)

            # Check if this is a marathon race
            if is_marathon and 120 <= duration_min <= 360:
                marathons.append(activity)

        except Exception as e:
            continue

    return activities, marathons


def detect_pr_from_marathons(marathons, race_date):
    """Find PR from marathons before the race date, excluding downhill courses"""
    prior_marathons = [m for m in marathons if m['date'] < race_date]

    if not prior_marathons:
        return None, None, [], []

    # Filter out downhill courses (net drop > 500ft) for PR consideration
    # These are aided times and shouldn't count as true PRs
    DOWNHILL_THRESHOLD = -500  # feet

    pr_eligible = [m for m in prior_marathons if m.get('net_elevation', 0) >= DOWNHILL_THRESHOLD]
    excluded_downhill = [m for m in prior_marathons if m.get('net_elevation', 0) < DOWNHILL_THRESHOLD]

    if not pr_eligible:
        # All marathons were downhill - use fastest anyway but warn
        pr_marathon = min(prior_marathons, key=lambda m: m['duration_min'])
    else:
        # Find fastest from eligible (non-downhill) marathons
        pr_marathon = min(pr_eligible, key=lambda m: m['duration_min'])

    pr_time = pr_marathon['duration_min']
    pr_date = pr_marathon['date']

    return pr_time, pr_date, prior_marathons, excluded_downhill


def extract_features(activities, race_date, marathon_pr, pr_date, prior_marathon_time):
    """Extract features for prediction"""
    # Training window: 16 weeks before race, excluding 7-day taper
    window_end = race_date - timedelta(days=7)
    window_start = race_date - timedelta(weeks=16)

    # Filter training activities (exclude marathon-distance)
    training = [a for a in activities
                if window_start <= a['date'] < window_end
                and a['distance_miles'] < 25]

    if len(training) < 5:
        return None, window_start, window_end

    features = {}

    # Volume features
    features['total_runs'] = len(training)
    features['total_mileage'] = sum(a['distance_miles'] for a in training)
    features['avg_weekly_mileage'] = features['total_mileage'] / 16

    # Weekly mileage stats
    weekly_miles = {}
    for a in training:
        week = a['date'].isocalendar()[1]
        weekly_miles[week] = weekly_miles.get(week, 0) + a['distance_miles']

    features['peak_weekly_mileage'] = max(weekly_miles.values()) if weekly_miles else 0
    if len(weekly_miles) > 1:
        features['avg_weekly_mileage_std'] = np.std(list(weekly_miles.values()))
    else:
        features['avg_weekly_mileage_std'] = 0

    # Long runs
    long_runs = [a for a in training if a['distance_miles'] >= 15]
    features['long_run_count'] = len(long_runs)
    features['long_run_max_distance'] = max([a['distance_miles'] for a in long_runs], default=0)
    features['long_run_avg_distance'] = np.mean([a['distance_miles'] for a in long_runs]) if long_runs else 0

    # Very long runs (20+)
    very_long = [a for a in training if a['distance_miles'] >= 20]
    features['very_long_run_count'] = len(very_long)

    # Pace features
    valid_pace = [a for a in training if a['pace'] and a['pace'] > 0]
    features['avg_pace'] = np.mean([a['pace'] for a in valid_pace]) if valid_pace else 10.0
    features['pace_std'] = np.std([a['pace'] for a in valid_pace]) if len(valid_pace) > 1 else 0

    # Tempo runs (7-8 min/mile)
    tempo = [a for a in training if a['pace'] and 7.0 <= a['pace'] < 8.0]
    features['tempo_run_count'] = len(tempo)
    features['tempo_mileage'] = sum(a['distance_miles'] for a in tempo)

    # Speed work (<7 min/mile)
    speed = [a for a in training if a['pace'] and a['pace'] < 7.0]
    features['speed_work_count'] = len(speed)

    # Easy runs (9+ min/mile)
    easy = [a for a in training if a['pace'] and a['pace'] >= 9.0]
    features['easy_run_count'] = len(easy)
    features['easy_run_pct'] = len(easy) / len(training) * 100 if training else 0

    # Heart rate (from CSV - limited)
    hr_runs = [a for a in training if a['avg_hr']]
    features['avg_hr'] = np.mean([a['avg_hr'] for a in hr_runs]) if hr_runs else 0
    features['max_hr_recorded'] = max([a['avg_hr'] for a in hr_runs], default=0)

    # HR zones (set to 0 - CSV doesn't have this detail)
    features['zone1_pct'] = 0
    features['zone2_pct'] = 0
    features['zone3_pct'] = 0
    features['zone4_pct'] = 0
    features['zone5_pct'] = 0

    # Cadence (not in CSV)
    features['avg_cadence'] = 0
    features['avg_pace_variability'] = 0

    # Training consistency
    days_with_runs = len(set(a['date'].date() for a in training))
    features['training_frequency'] = days_with_runs / (16 * 7) * 100

    # Recent form (last 4 weeks before taper)
    recent_start = race_date - timedelta(weeks=4) - timedelta(days=7)
    recent = [a for a in training if a['date'] >= recent_start]
    features['recent_mileage'] = sum(a['distance_miles'] for a in recent)
    recent_pace = [a for a in recent if a['pace'] and a['pace'] > 0]
    features['recent_avg_pace'] = np.mean([a['pace'] for a in recent_pace]) if recent_pace else features['avg_pace']

    # PR-related features (the key ones!)
    features['marathon_pr'] = marathon_pr if marathon_pr else 210.0
    features['prior_marathon_time'] = prior_marathon_time if prior_marathon_time else 210.0

    # PR age and decay
    if marathon_pr and pr_date:
        pr_age_days = (race_date - pr_date).days
        pr_age_years = max(0, pr_age_days / 365.25)
    else:
        pr_age_years = 0.0

    decay_factor = 0.5 ** (pr_age_years / 3.0)  # Half-life of 3 years
    features['pr_age_years'] = pr_age_years
    features['pr_decay_factor'] = decay_factor
    features['decayed_pr'] = features['marathon_pr'] * decay_factor + 210.0 * (1 - decay_factor)

    return features, window_start, window_end


def format_time(minutes):
    """Format minutes as H:MM"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}:{mins:02d}"


def analyze_csv(csv_file, race_date_str):
    """Analyze CSV and return detected PR info"""
    if csv_file is None:
        return "Upload a CSV file first", "", 3, 30, ""

    try:
        race_date = datetime.strptime(race_date_str, "%Y-%m-%d")
    except:
        return "Invalid date format", "", 3, 30, ""

    try:
        with open(csv_file.name, 'r') as f:
            csv_content = f.read()
        activities, marathons = parse_strava_csv(csv_content)
    except Exception as e:
        return f"Error: {str(e)}", "", 3, 30, ""

    # Detect CSV format for warning
    df = pd.read_csv(StringIO(csv_content))
    csv_format = detect_csv_format(df)

    pr_time, pr_date, prior_marathons, excluded_downhill = detect_pr_from_marathons(marathons, race_date)

    if pr_time:
        pr_info = f"**Detected PR: {format_time(pr_time)}** (set on {pr_date.strftime('%Y-%m-%d')})"
        pr_hours = int(pr_time // 60)
        pr_mins = int(pr_time % 60)
        pr_date_str = pr_date.strftime("%Y-%m-%d")

        # List all prior marathons with elevation info
        marathon_lines = []
        for m in sorted(prior_marathons, key=lambda x: x['date']):
            net_elev = m.get('net_elevation', 0)
            elev_str = f" ({net_elev:+.0f}ft)" if net_elev != 0 else ""
            excluded_marker = " ⚠️ *excluded from PR (downhill)*" if m in excluded_downhill else ""
            marathon_lines.append(
                f"- {m['date'].strftime('%Y-%m-%d')}: {format_time(m['duration_min'])}{elev_str} - {m['name'][:35]}{excluded_marker}"
            )
        marathon_list = "\n".join(marathon_lines)

        if excluded_downhill:
            marathon_info = f"**Found {len(prior_marathons)} prior marathon(s)** ({len(excluded_downhill)} downhill excluded from PR):\n{marathon_list}"
        else:
            marathon_info = f"**Found {len(prior_marathons)} prior marathon(s):**\n{marathon_list}"

        # Add note about enriched CSV limitations
        if csv_format == 'enriched' and not excluded_downhill:
            marathon_info += "\n\n*Note: Using enriched CSV - downhill detection limited. Use override below if your PR was on a downhill course.*"
    else:
        pr_info = "**No prior marathons detected.** Please enter your PR manually below."
        pr_hours = 3
        pr_mins = 30
        pr_date_str = ""
        marathon_info = ""

    return pr_info, marathon_info, pr_hours, pr_mins, pr_date_str


def predict_marathon(csv_file, race_date_str, override_pr, pr_hours, pr_minutes, pr_date_str):
    """Main prediction function"""
    if csv_file is None:
        return "Please upload an activities.csv file", "", ""

    try:
        race_date = datetime.strptime(race_date_str, "%Y-%m-%d")
    except:
        return "Error: Invalid race date format. Use YYYY-MM-DD", "", ""

    # Parse CSV
    try:
        with open(csv_file.name, 'r') as f:
            csv_content = f.read()
        activities, marathons = parse_strava_csv(csv_content)
    except Exception as e:
        return f"Error reading CSV: {str(e)}", "", ""

    if not activities:
        return "Error: No running activities found", "", ""

    # Get PR (auto-detected or override)
    auto_pr_time, auto_pr_date, prior_marathons, excluded_downhill = detect_pr_from_marathons(marathons, race_date)

    if override_pr and pr_hours >= 0 and pr_minutes >= 0:
        # Use manual override
        marathon_pr = pr_hours * 60 + pr_minutes
        if pr_date_str:
            try:
                pr_date = datetime.strptime(pr_date_str, "%Y-%m-%d")
            except:
                pr_date = auto_pr_date or (race_date - timedelta(days=365))
        else:
            pr_date = auto_pr_date or (race_date - timedelta(days=365))
        pr_source = "manual override"
    elif auto_pr_time:
        marathon_pr = auto_pr_time
        pr_date = auto_pr_date
        pr_source = "auto-detected"
    else:
        # No PR available - use default
        marathon_pr = 210.0  # 3:30 default
        pr_date = race_date - timedelta(days=365)
        pr_source = "default (no prior marathons found)"

    # Get most recent prior marathon time
    if prior_marathons:
        most_recent = max(prior_marathons, key=lambda m: m['date'])
        prior_marathon_time = most_recent['duration_min']
    else:
        prior_marathon_time = marathon_pr

    # Extract features
    result = extract_features(activities, race_date, marathon_pr, pr_date, prior_marathon_time)
    if result[0] is None:
        return "Error: Not enough training data in the 16-week window", "", ""

    features, window_start, window_end = result

    # Build feature vector
    X = np.array([[features.get(name, 0) or 0 for name in FEATURE_NAMES]])

    # Predict
    prediction = MODEL.predict(X)[0]
    # Cap to realistic range (2:30 to 5:00)
    prediction = max(150, min(300, prediction))

    # Calculate confidence range
    std_estimate = 8  # Based on holdout MAE of ~5 min
    pred_low = prediction - std_estimate
    pred_high = prediction + std_estimate

    prediction_str = f"## Predicted Marathon Time: {format_time(prediction)}"
    range_str = f"**Confidence Range:** {format_time(pred_low)} - {format_time(pred_high)}"

    # PR age info
    pr_age_years = features['pr_age_years']
    decay_factor = features['pr_decay_factor']

    insights = f"""
### Training Summary
**Race Date:** {race_date_str}
**Training Window:** {window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}

### PR Information
- **Marathon PR:** {format_time(marathon_pr)} ({pr_source})
- **PR Age:** {pr_age_years:.1f} years
- **PR Decay Factor:** {decay_factor:.2f} (older PRs weighted less)

### Training Metrics
| Metric | Value |
|--------|-------|
| Weekly Mileage | {features['avg_weekly_mileage']:.1f} miles |
| Peak Week | {features['peak_weekly_mileage']:.1f} miles |
| Total Runs | {features['total_runs']} |
| Long Runs (15+ mi) | {features['long_run_count']} |
| Longest Run | {features['long_run_max_distance']:.1f} miles |
| Tempo Runs | {features['tempo_run_count']} |
| Recent 4-Week Mileage | {features['recent_mileage']:.1f} miles |
| Training Frequency | {features['training_frequency']:.0f}% of days |
"""

    # Warnings
    warnings = []
    if features['avg_weekly_mileage'] < 30:
        warnings.append("Low weekly mileage - consider more volume")
    if features['long_run_count'] < 3:
        warnings.append("Few long runs - important for marathon endurance")
    if pr_age_years > 3:
        warnings.append(f"PR is {pr_age_years:.1f} years old - prediction has more uncertainty")

    if warnings:
        insights += "\n### Warnings\n" + "\n".join(f"- ⚠️ {w}" for w in warnings)

    return prediction_str, range_str, insights


# Create Gradio interface
with gr.Blocks(title="Marathon Time Predictor") as app:
    gr.Markdown("""
    # Marathon Time Predictor

    Predict your marathon finish time based on your Strava training data.

    **How it works:**
    1. Upload your CSV (either raw `activities.csv` from Strava export, or enriched CSV)
    2. Enter your race date
    3. Review auto-detected PR (override if it was on a downhill course)
    4. Get your prediction!

    *Tip: Raw Strava `activities.csv` can auto-detect downhill PRs. Enriched CSVs need manual override.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Upload Data")
            csv_file = gr.File(label="activities.csv", file_types=[".csv"])
            race_date = gr.Textbox(
                label="Race Date (YYYY-MM-DD)",
                value=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                placeholder="2026-10-15"
            )

            analyze_btn = gr.Button("Analyze CSV", variant="secondary")

            gr.Markdown("### Step 2: Review PR")
            pr_info = gr.Markdown("Upload CSV to detect your marathon PR")
            marathon_info = gr.Markdown("")

            override_pr = gr.Checkbox(label="Override detected PR", value=False)
            with gr.Row():
                pr_hours = gr.Number(label="PR Hours", value=3, minimum=2, maximum=6, visible=True)
                pr_minutes = gr.Number(label="PR Minutes", value=30, minimum=0, maximum=59, visible=True)
            pr_date_input = gr.Textbox(label="PR Date (YYYY-MM-DD)", value="", placeholder="2024-03-15", visible=True)

            gr.Markdown("### Step 3: Predict")
            predict_btn = gr.Button("Predict Marathon Time", variant="primary", size="lg")

        with gr.Column(scale=1):
            prediction_output = gr.Markdown()
            range_output = gr.Markdown()
            insights_output = gr.Markdown()

    # Wire up analyze button
    def on_analyze(csv_file, race_date_str):
        result = analyze_csv(csv_file, race_date_str)
        if len(result) == 4:
            # Error case
            return result[0], "", 3, 30, ""
        return result

    analyze_btn.click(
        fn=on_analyze,
        inputs=[csv_file, race_date],
        outputs=[pr_info, marathon_info, pr_hours, pr_minutes, pr_date_input]
    )

    # Wire up predict button
    predict_btn.click(
        fn=predict_marathon,
        inputs=[csv_file, race_date, override_pr, pr_hours, pr_minutes, pr_date_input],
        outputs=[prediction_output, range_output, insights_output]
    )

    gr.Markdown("""
    ---
    ### About This Model

    **Unanchored model** trained on 43 marathon races from 5 runners using Random Forest.

    **Key Features:**
    - Auto-detects your marathon PR from Strava data
    - PR decay: older PRs are weighted less (3-year half-life)
    - Uses 16-week training window before race

    **Model Performance:**
    - Cross-validation MAE: ~14 minutes
    - Holdout validation MAE: ~5 minutes

    **Limitations:**
    - Cannot detect downhill course PRs (override if needed)
    - Limited HR data from CSV (FIT files have more detail)
    """)


if __name__ == "__main__":
    app.launch()
