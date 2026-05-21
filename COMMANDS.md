# Strava Guru - Command Reference

All commands to run the marathon prediction pipeline.

---

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start (If Data Already Exists)

```bash
# Run holdout validation (trains model + predicts)
python predict_holdout_validation.py

# Train weather-aware model
python train_with_weather.py
```

---

## Full Pipeline

### Step 1: Build Activity Cache

**Option A: From FIT/GPX files**
```bash
python build_cache.py /path/to/strava/activities
```

**Option B: From Strava CSV export (recommended - handles corrupted FIT files)**
```bash
python build_cache_from_csv.py
```

For additional runners:
```bash
python build_salman_cache.py    # Build cache for Salman's data
```

### Step 2: Extract Race Features

**Find marathon races in activity data:**
```bash
python extract_all_marathons.py
python extract_races_smart.py
```

**Extract 41 training features for each race:**
```bash
python test_feature_extraction.py           # Test feature extraction
python extract_salman_features_full.py      # Extract for Salman
python extract_your_simple_features.py      # Extract for yourself
```

### Step 3: Create Training Dataset

```bash
# Combine all runners into one dataset
python train_combined_model.py

# Or train on 3 runners with holdout
python train_final_3_runners.py
```

### Step 4: Train Models

**Basic training (compares 5 models):**
```bash
python train_race_model.py race_data/combined_41_features.json
```

**Weather-aware training:**
```bash
python train_with_weather.py
```

**3-runner model with filtering:**
```bash
python filter_and_train_final.py
```

### Step 5: Make Predictions

**Holdout validation (leave-one-out per runner):**
```bash
python predict_holdout_validation.py
```

**Predict your next race:**
```bash
python predict_race_time.py 26.2    # Marathon
python predict_race_time.py 13.1    # Half marathon
```

**Temperature impact analysis:**
```bash
python predict_temperature_scenarios.py
python predict_with_temperature.py
```

---

## Debugging & Investigation

```bash
# Debug feature extraction issues
python debug_feature_extraction.py

# Diagnose missing activities in cache
python diagnose_missing_activities.py

# Verify training data integrity
python verify_salman_training.py

# Check for bonked races (outliers)
python check_azeem_bonking.py
python check_salman_bonking.py
python check_long_runs.py
```

---

## Data Files

| File | Description |
|------|-------------|
| `race_data/combined_41_features.json` | Main dataset: 21+ marathons, 45 features |
| `~/.strava_guru_cache/activities.db` | SQLite cache of parsed activities |
| `race_time_model_with_weather.pkl` | Trained Random Forest model |

---

## Typical Workflow

```bash
# 1. Setup
source venv/bin/activate

# 2. If you have new Strava data, rebuild cache
python build_cache_from_csv.py

# 3. Extract features for new races
python extract_all_marathons.py

# 4. Train and validate
python predict_holdout_validation.py

# 5. Check temperature impact
python train_with_weather.py
```

---

## Model Output

Current best model: **Random Forest** with weather features
- Accuracy: ±11.5 minutes MAE
- Top feature: Apparent temperature (17.8% importance)
- Dataset: 15 marathons with weather data from 3 runners
