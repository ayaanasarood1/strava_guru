# Strava Marathon Predictor

> Predict your marathon finish time using machine learning on Strava training data

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/marathon-predictor)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Understanding](#2-dataset-understanding)
3. [Data Cleaning and Preprocessing](#3-data-cleaning-and-preprocessing)
4. [Algorithm and Architecture](#4-algorithm-and-architecture)
5. [Metrics and Evaluation](#5-metrics-and-evaluation)
6. [Deployment](#6-deployment)
7. [LLM Conversation](#7-llm-conversation)
8. [Quick Start](#quick-start)
9. [Activity Analyzer (Original Tool)](#activity-analyzer-original-tool)

---

## 1. Project Overview

This project builds a machine learning model to predict marathon finish times based on training data from Strava. Users can upload their Strava data export and receive a predicted finish time along with training insights.

**Key Results:**
- **Model:** Random Forest Regressor
- **Accuracy:** 6.1 minutes average error on holdout validation
- **Features:** 48 training metrics including mileage, quality workouts, and race conditions
- **Deployment:** Hugging Face Spaces web application

---

## 2. Dataset Understanding

### Data Source

Training data was collected from **5 recreational marathon runners** via Strava's data export feature:

| Runner | Marathons | Years | Best Time | Avg Weekly Miles |
|--------|-----------|-------|-----------|------------------|
| Osman | 7 | 2021-2025 | 3:04 | 54.8 |
| Salman | 17 | 2012-2025 | 2:55 | 61.6 (recent) |
| Azeem | 5 | 2022-2026 | 3:22 | 53.0 |
| Sara | 11 | 2022-2026 | 3:17 | 35.5 |
| Qazi | 1 | 2025 | 3:53 | 41.8 |

**Total Dataset:** 41 marathons → 37 after filtering "bonked" races

### Target Variable

- **Marathon finish time in minutes**
- Range: 175 - 271 minutes (2:55 - 4:31)
- Mean: 206 minutes (3:26)
- Distribution: Skewed toward 3:00-3:30 range

### Feature Categories (48 features)

| Category | Count | Examples |
|----------|-------|----------|
| **Volume** | 6 | weekly_mileage, peak_mileage, total_runs |
| **Quality** | 4 | tempo_workout_count, quality_percent |
| **Long Runs** | 3 | long_run_distance, long_run_count |
| **Consistency** | 2 | mileage_consistency, training_consistency_score |
| **Runner Profile** | 5 | age, sex, experience_years, historical_pr |
| **Race Conditions** | 4 | temperature, humidity, wind_speed |
| **Physiological** | 12 | avg_hr, hr_zones, cardiac_drift |
| **Other** | 12 | Reserved/placeholder features |

### Data Concerns

1. **Small sample size** (37 races) limits generalization
2. **Data imbalance:** Only 4 sub-3:00 races (all from one runner Salman)
3. **Individual variation:** Same training produces different results per runner
4. **Missing HR data:** ~30% of activities lack heart rate information

---

## 3. Data Cleaning and Preprocessing

### Pipeline Overview

```
Raw Strava Export (ZIP)
    │
    ▼
Extract activities.csv
    │
    ▼
Filter to Running Activities
    │
    ▼
Identify Marathon Races (25-27.5 miles)
    │
    ▼
Remove Bonked/Invalid Races
    │
    ▼
Extract 4-Month Training Window
    │
    ▼
Calculate 48 Features
    │
    ▼
Clean Dataset (37 races)
```

### Key Cleaning Steps

#### 1. FIT File Corruption Handling
**Problem:** 90% of FIT files from one runner were corrupted
```
fitparse error: Invalid field size
```
**Solution:** Created CSV-based fallback parser (`build_cache_from_csv.py`)

#### 2. Bonked Race Filtering
Removed races with significant underperformance:
```python
bonked_races = [
    ('my_runner', 'marathon_20251012'),   # 3:45 - injury
    ('runner_sara', 'sara_marathon_20240623'),  # Training run
    # ... 3 more
]
```

#### 3. Humidity Normalization
**Bug Found:** Inconsistent units across runners
```python
# Fix: Normalize all to 0-1 scale
if humidity > 1:
    humidity = humidity / 100
```

#### 4. Feature Extraction Window
```python
# Training window: 16 weeks before race, excluding 7-day taper
lookback_start = race_date - timedelta(weeks=16)
taper_start = race_date - timedelta(days=7)
training_activities = filter(lookback_start <= date < taper_start)
```

---

## 4. Algorithm and Architecture

### Model Comparison

| Algorithm | 5-Fold CV MAE | Notes |
|-----------|---------------|-------|
| Ridge Regression | 30.5 min | Poor - linear assumptions fail |
| Lasso Regression | 28.6 min | Poor - over-regularized |
| **Random Forest** | **13.9 min** | **Best - captures non-linearity** |
| Gradient Boosting | 16.0 min | Good but tends to overfit |

### Why Random Forest?

1. **Non-linear relationships:** Diminishing returns on mileage
2. **Feature interactions:** Captures high mileage × quality workouts
3. **Robust:** No scaling needed, handles missing values
4. **Interpretable:** Provides feature importance rankings

### Final Model Configuration

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,    # 100 trees in forest
    max_depth=10,        # Prevent overfitting
    random_state=42      # Reproducibility
)
```

### Top 10 Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | total_weekly_mileage | 15.1% |
| 2 | historical_pr_minutes | 13.0% |
| 3 | race_wind_speed | 11.0% |
| 4 | peak_weekly_mileage | 9.5% |
| 5 | runs_per_week | 7.2% |
| 6 | avg_hr | 4.6% |
| 7 | total_runs | 4.6% |
| 8 | tempo_workout_count | 4.1% |
| 9 | training_consistency_score | 3.6% |
| 10 | race_apparent_temperature | 3.4% |

---

## 5. Metrics and Evaluation

### Evaluation Strategy

1. **5-Fold Cross-Validation** on training set
2. **Holdout Validation** on each runner's most recent race

### Results

#### Cross-Validation (Training)
- **CV MAE: 13.9 minutes**
- Indicates expected error on unseen data from similar distribution

#### Holdout Validation (Test)

| Runner | Race | Predicted | Actual | Error |
|--------|------|-----------|--------|-------|
| Salman | Oct 2025 | 3:08 | 2:56 | +12.3 min |
| Osman | Dec 2024 | 3:15 | 3:22 | -7.1 min |
| Sara | Boston 2025 | 3:28 | 3:24 | +3.5 min |
| Sara | London 2026 | 3:27 | 3:22 | +4.4 min |
| Azeem | Houston 2026 | 3:25 | 3:22 | +3.1 min |

**Holdout MAE: 6.1 minutes**

### Error Analysis

**Why Salman's error is large (+12 min):**
- Only 4 sub-3:00 races in entire dataset (all Salman)
- Model averages toward mean (3:26), underpredicts fast runners
- This is a data imbalance problem, not a feature problem

**Why Osman was overpredicted (-7 min):**
- Race had 90% humidity
- Model doesn't capture individual weather sensitivity
- Osman struggles in humidity; Salman handles it well

---

## 6. Deployment

### Hugging Face Spaces App

**Live Demo:** [https://huggingface.co/spaces/YOUR_USERNAME/marathon-predictor](https://huggingface.co/spaces/YOUR_USERNAME/marathon-predictor)

### Features

- Upload Strava export (zip file)
- Automatic feature extraction from activities.csv
- Input race date, profile, expected weather
- Get predicted finish time with confidence range
- View training insights and warnings

### User Flow

```
1. User downloads Strava export (zip)
        ↓
2. Uploads to app
        ↓
3. App extracts activities.csv
        ↓
4. User enters race date + profile
        ↓
5. App calculates 4-month training features
        ↓
6. Model predicts finish time
        ↓
7. Display prediction + insights + warnings
```

### Tech Stack

- **Frontend:** Gradio 4.x
- **Backend:** scikit-learn, pandas, numpy
- **Hosting:** Hugging Face Spaces (free)

---

## 7. LLM Conversation

### Development Process

This project was developed collaboratively with **Claude** (Anthropic's AI) over multiple sessions. The full conversation context is documented in `SESSION_NOTES.md`.

### LLM Contributions

| Task | LLM Role |
|------|----------|
| Data pipeline | Generated CSV parser when FIT files corrupted |
| Bug discovery | Found humidity unit inconsistency |
| Model analysis | Investigated prediction errors, identified data imbalance |
| Documentation | Generated README, docstrings, comments |
| Deployment | Created Gradio app and Hugging Face setup |

### Reflection on LLM Use

**Effective Uses:**
- Rapid iteration on code
- Systematic debugging approach
- Documentation generation
- Exploring multiple hypotheses

**Limitations Encountered:**
- Context loss between sessions (mitigated with SESSION_NOTES.md)
- Initial wrong conclusions (corrected by user domain expertise)
- Tendency to over-engineer solutions

**Key Learning:**
> "LLM initially suggested the training data was wrong. User correctly identified it was a data imbalance problem - same training volume means different things for different runners. Only 4 sub-3:00 races in 37 total races meant the model was pulled toward the slower mean."

Human expertise remained critical for domain-specific insights.

### Ethical Considerations

- **Privacy:** Strava data contains location information - app processes locally, doesn't store data
- **Bias:** Model trained on 5 specific runners may not generalize to all populations
- **Limitations:** Clearly communicated in app interface

---

## Quick Start

### Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/strava_guru.git
cd strava_guru

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run holdout validation
python predict_holdout_validation.py

# Run Hugging Face app locally
cd huggingface_app
pip install -r requirements.txt
python app.py
```

### Get Your Strava Data

1. Go to [Strava Settings](https://www.strava.com/settings/profile)
2. Click "Download or Delete Your Account"
3. Click "Request Your Archive"
4. Wait for email (can take up to 24 hours)
5. Download and upload to the app

---

## Project Structure

```
strava_guru/
├── README.md                      # This file
├── SESSION_NOTES.md               # LLM conversation context
├── requirements.txt               # Python dependencies
│
├── predict_holdout_validation.py  # Model evaluation script
├── build_cache_from_csv.py        # Data preprocessing
├── activity_analyzer.py           # Core analysis module
│
├── race_data/
│   ├── combined_41_features.json  # Training dataset
│   └── *_marathons.json           # Per-runner data
│
└── huggingface_app/
    ├── app.py                     # Gradio web application
    ├── model.pkl                  # Trained model
    ├── requirements.txt           # App dependencies
    └── README.md                  # Hugging Face metadata
```

---

## Activity Analyzer (Original Tool)

This repository also includes the original Strava activity analyzer tool:

### Features

- **Parse FIT and GPX files** (including gzipped versions)
- **Calculate comprehensive stats**: Distance, pace, GAP, elevation, heart rate
- **Mile splits** with per-split statistics
- **Visualizations**: Pace charts, elevation profiles, route maps

### Usage

```bash
# Analyze a single activity
python activity_analyzer.py ~/Downloads/strava_export/activities/12345.fit.gz

# Generate visualizations
python visualizer.py ~/Downloads/strava_export/activities/12345.fit.gz ./charts/
```

### Output Example

```
================================================================================
ACTIVITY SUMMARY
================================================================================
Distance:       18.12 mi
Moving Time:    2:21:55
Pace:           7:50 /mi
GAP:            7:47 /mi
Elevation Gain: 436 ft
Avg Heart Rate: 156 bpm
================================================================================
```

---

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- Training data from 5 volunteer runners
- Built with [scikit-learn](https://scikit-learn.org/), [Gradio](https://gradio.app/), [Hugging Face](https://huggingface.co/)
- Developed with assistance from [Claude](https://www.anthropic.com/claude) (Anthropic)

---

*Last updated: May 2026*
