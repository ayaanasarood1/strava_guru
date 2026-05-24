---
title: Marathon Time Predictor
emoji: 🏃
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.29.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# Marathon Time Predictor

Predict your marathon finish time based on your Strava training data using machine learning.

**Live App:** [Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/marathon-predictor)

---

## 1. Dataset Understanding

### Data Source
- **Primary Source:** Strava activity exports from 5 recreational marathon runners
- **Collection Method:** Downloaded via Strava's "Download Your Data" feature
- **Time Period:** 2012-2026 (varies by runner)
- **Geographic Region:** United States (various locations)

### Dataset Composition

| Runner | Marathons | Training Period | Best Time |
|--------|-----------|-----------------|-----------|
| Osman | 7 | 2021-2025 | 3:04 |
| Salman | 17 | 2012-2025 | 2:55 |
| Azeem | 5 | 2022-2026 | 3:22 |
| Sara | 11 | 2022-2026 | 3:17 |
| Qazi | 1 | 2025 | 3:53 |
| **Total** | **41** | | |

After filtering "bonked" races (DNF or significant underperformance): **37 clean races**

### Target Variable
- **Marathon finish time in minutes** (continuous, regression task)
- Range: 175 - 271 minutes (2:55 - 4:31)
- Mean: 206 minutes (3:26)
- Standard Deviation: 22 minutes

### Features (48 total)

#### Training Volume Features
| Feature | Description | Importance |
|---------|-------------|------------|
| `total_weekly_mileage` | Average miles per week over 4-month training block | 15.1% |
| `peak_weekly_mileage` | Highest single week mileage | 9.5% |
| `runs_per_week` | Average running frequency | 7.2% |
| `total_runs` | Total number of runs in training block | 4.6% |
| `long_run_distance` | Longest single run (miles) | 2.5% |
| `long_run_count` | Number of runs 15+ miles | - |

#### Training Quality Features
| Feature | Description | Importance |
|---------|-------------|------------|
| `tempo_workout_count` | Runs at 7:00-8:00 min/mile pace | 4.1% |
| `fast_workout_count` | Runs faster than 7:30 min/mile | 2.5% |
| `quality_workout_percent` | Percentage of quality workouts | 2.8% |
| `training_consistency_score` | Week-to-week mileage consistency (0-1) | 3.6% |

#### Runner Profile Features
| Feature | Description |
|---------|-------------|
| `historical_pr_minutes` | Best prior marathon time | 13.0% importance |
| `age_normalized` | Age factor (peak at 30) |
| `sex_encoded` | Male=1, Female=0 |
| `experience_years` | Years of running experience |

#### Race Day Features
| Feature | Description | Importance |
|---------|-------------|------------|
| `race_temperature` | Temperature in °F | 2.7% |
| `race_humidity` | Humidity (0-1 scale) | - |
| `race_wind_speed` | Wind speed | 11.0% |
| `race_apparent_temperature` | Feels-like temperature | 3.4% |

### Data Concerns & Limitations

1. **Small Sample Size:** Only 37 races from 5 runners limits generalization
2. **Data Imbalance:** Only 4 sub-3:00 races (all from one runner)
3. **Selection Bias:** All runners are recreational athletes who self-selected to track data
4. **Missing Data:** Some features (HR zones, cardiac drift) are incomplete for CSV-based imports
5. **Individual Variation:** Same training produces different results for different runners

---

## 2. Data Cleaning and Preprocessing

### Raw Data Processing Pipeline

```
Strava Export (ZIP)
    │
    ├── activities.csv (structured data)
    │   └── Parse: distance, time, HR, date, weather
    │
    └── FIT/GPX files (detailed track data)
        └── Note: Many corrupted, fallback to CSV
```

### Cleaning Steps

#### Step 1: Activity Parsing
```python
# Filter to running activities only
if row.get('Activity Type') != 'Run':
    continue

# Convert units
distance_miles = distance_meters / 1609.34

# Handle missing time data
duration = moving_time if moving_time > 0 else elapsed_time
```

#### Step 2: Marathon Identification
- Filter activities with distance 25.0 - 27.5 miles
- Exclude obvious training runs (> 5:00 finish time)
- Manual review for borderline cases

#### Step 3: Bonked Race Filtering
Races with significant underperformance due to injury, illness, or conditions:
```python
bonked_races = [
    ('my_runner', 'marathon_20251012'),   # Osman's 3:45 (injury)
    ('my_runner', 'marathon_20231008'),   # Bad conditions
    ('runner_2', 'marathon_20231008'),    # Same race
    ('runner_3', 'marathon_20231008'),    # Same race
    ('runner_sara', 'sara_marathon_20240623'),  # Training run
]
```

#### Step 4: Feature Normalization

**Humidity Inconsistency Found & Fixed:**
```python
# Some data had humidity as percentage (67), others as decimal (0.67)
if humidity > 1:
    humidity = humidity / 100  # Normalize to 0-1
```

#### Step 5: Training Window Extraction
- Look back 16 weeks (4 months) from race date
- Exclude final 7 days (taper period)
- Calculate aggregate statistics

### Data Quality Checks

| Check | Result | Action |
|-------|--------|--------|
| Missing distances | 0 | N/A |
| Invalid dates | 12 | Excluded |
| Missing HR data | ~30% | Used available data |
| Duplicate activities | 3 | Removed |
| Outlier finish times | 5 | Manually reviewed |

---

## 3. Algorithm and Architecture

### Model Selection Process

We evaluated 4 regression algorithms:

| Model | CV MAE | Notes |
|-------|--------|-------|
| Ridge Regression | 30.5 min | Poor - assumes linear relationships |
| Lasso Regression | 28.6 min | Poor - too much regularization |
| **Random Forest** | **13.9 min** | **Best - handles non-linearity** |
| Gradient Boosting | 16.0 min | Good but prone to overfitting |

### Why Random Forest?

1. **Non-linear Relationships:** Marathon time isn't linearly related to mileage - there are diminishing returns and thresholds
2. **Feature Interactions:** Captures interactions (e.g., high mileage + good quality workouts)
3. **Robust to Outliers:** Doesn't require feature scaling
4. **Interpretable:** Provides feature importance rankings
5. **No Overfitting:** Ensemble of trees with max_depth=10 prevents memorization

### Final Model Architecture

```python
RandomForestRegressor(
    n_estimators=100,      # 100 decision trees
    max_depth=10,          # Limit tree depth to prevent overfitting
    random_state=42,       # Reproducibility
    min_samples_split=2,   # Default splitting
    min_samples_leaf=1     # Default leaf size
)
```

### Feature Engineering Decisions

1. **Historical PR as Feature:** Added runner's best prior marathon time - became #2 most important feature (13%)
2. **Quality Workout Detection:** Identified tempo runs by pace threshold (<8:00 min/mile)
3. **Consistency Score:** Week-to-week mileage variance penalizes erratic training
4. **Dropped Features:** Zone distribution features were mostly zeros (CSV data limitation)

---

## 4. Metrics and Evaluation

### Primary Metric: Mean Absolute Error (MAE)

**Why MAE?**
- Interpretable: "Model is off by X minutes on average"
- Robust to outliers (vs. MSE/RMSE)
- Same units as target variable

### Evaluation Strategy

#### 1. 5-Fold Cross-Validation (Training Data)
```
Fold 1: Train on 80%, Test on 20% → MAE
Fold 2: Train on 80%, Test on 20% → MAE
...
Final CV MAE: 13.9 minutes (average across folds)
```

#### 2. Holdout Validation (Unseen Races)
Hold out each runner's most recent race for final evaluation:

| Runner | Race | Predicted | Actual | Error |
|--------|------|-----------|--------|-------|
| Salman | Oct 2025 | 3:08 | 2:56 | +12.3 min |
| Osman | Dec 2024 | 3:15 | 3:22 | -7.1 min |
| Sara | Boston 2025 | 3:28 | 3:24 | +3.5 min |
| Sara | London 2026 | 3:27 | 3:22 | +4.4 min |
| Azeem | Houston 2026 | 3:25 | 3:22 | +3.1 min |

**Holdout MAE: 6.1 minutes**

### Interpretation of Results

**What the metrics tell us:**
- Model predicts within ~6 minutes for most runners
- Larger errors for extreme cases (Salman's sub-3:00 times)
- CV MAE > Holdout MAE suggests good generalization (not overfitting)

**Error Analysis:**

| Error Source | Impact | Mitigation |
|--------------|--------|------------|
| Data imbalance (few sub-3:00) | +12 min for fast runners | Need more fast runner data |
| Individual weather sensitivity | ±10 min | Would need per-runner coefficients |
| Race execution variability | ±5 min | Cannot be captured in training data |

### Feature Importance Analysis

Top 5 features explain 56% of predictions:
1. **Weekly Mileage (15.1%)** - More miles = faster times
2. **Historical PR (13.0%)** - Past performance predicts future
3. **Wind Speed (11.0%)** - Weather matters significantly
4. **Peak Mileage (9.5%)** - Big training weeks indicate fitness
5. **Runs/Week (7.2%)** - Consistency and frequency

---

## 5. Deployment

### Hugging Face Spaces App

**URL:** https://huggingface.co/spaces/YOUR_USERNAME/marathon-predictor

### App Features

1. **Strava Data Upload:** Users upload their Strava export zip file
2. **Automatic Feature Extraction:** Parses activities.csv, calculates training metrics
3. **User Input:** Race date, age, sex, expected weather, historical PR
4. **Prediction Output:** Finish time with confidence range
5. **Training Insights:** Summary of user's training metrics with warnings

### Technology Stack

```
Frontend: Gradio 4.x (Python-native web UI)
Backend: scikit-learn Random Forest model
Hosting: Hugging Face Spaces (free tier)
```

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

### Deployment Files

```
huggingface_app/
├── app.py           # Gradio application
├── model.pkl        # Trained Random Forest model
├── requirements.txt # Python dependencies
└── README.md        # This file (Hugging Face metadata)
```

---

## 6. LLM Conversation & Reflection

### How LLM (Claude) Was Used

This project was developed collaboratively with Claude (Anthropic's AI assistant) over multiple sessions. The conversation is documented in `SESSION_NOTES.md`.

### Key LLM Contributions

1. **Data Pipeline Development**
   - Created CSV parsing scripts when FIT file parsing failed
   - Debugged feature extraction issues

2. **Bug Discovery**
   - Found humidity normalization inconsistency (0-1 vs 0-100)
   - Identified bonked race filter removing valid races (same date, different runners)

3. **Model Analysis**
   - Investigated why Salman's predictions were off → data imbalance
   - Analyzed individual weather sensitivity patterns

4. **Documentation**
   - Generated this README and project documentation
   - Maintained SESSION_NOTES.md for context preservation

### Responsible Use Reflection

**What went well:**
- LLM accelerated debugging and analysis
- Code generation saved significant time
- Systematic approach to investigating errors

**Challenges:**
- Context loss between sessions required careful documentation
- Some initial analyses were wrong (had to correct after user feedback)
- LLM suggested adding runner encoding which could overfit

**Lessons Learned:**
1. Always verify LLM outputs against domain knowledge
2. Document context for long-running projects
3. User expertise is critical - LLM missed that training data was correct
4. Iterative refinement works better than one-shot solutions

### Ethical Considerations

- **Privacy:** Strava data contains location information - app processes locally, doesn't store data
- **Bias:** Model trained on 5 specific runners may not generalize to all populations
- **Limitations:** Clearly communicated in app interface

---

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/strava_guru.git
cd strava_guru/huggingface_app

# Install dependencies
pip install -r requirements.txt

# Run app locally
python app.py
```

### Using the App

1. Export your Strava data (Settings → Download Your Data)
2. Upload the zip file to the app
3. Enter your race date and profile information
4. Get your prediction!

---

## Files in Repository

```
strava_guru/
├── README.md                    # Project overview
├── SESSION_NOTES.md             # LLM conversation context
├── predict_holdout_validation.py # Model evaluation script
├── race_data/
│   └── combined_41_features.json # Training dataset
└── huggingface_app/
    ├── app.py                   # Gradio web app
    ├── model.pkl                # Trained model
    ├── requirements.txt         # Dependencies
    └── README.md                # Hugging Face metadata
```

---

## License

MIT License - See LICENSE file

## Acknowledgments

- Training data provided by 5 volunteer runners
- Built with scikit-learn, Gradio, and Hugging Face Spaces
- Developed with assistance from Claude (Anthropic)
