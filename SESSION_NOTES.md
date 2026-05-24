# Claude Session Notes

This file tracks conversation context so Claude can pick up where we left off.

---

## Project Summary

**Strava Guru** - Race time prediction system using ML on Strava training data.

### Core Pipeline
1. **Activity Cache** - SQLite database of parsed .fit/.gpx files
2. **Feature Engineering** - 41 features (volume, intensity, efficiency, terrain, etc.)
3. **ML Models** - Ridge, Lasso, Random Forest, GBM, XGBoost
4. **Prediction** - Race time estimates from current training

### Runners in Dataset (41 marathons total)
- **Osman** - 7 marathons
- **Salman** - 17 marathons
- **Azeem** - 5 marathons
- **Sara** - 11 marathons (female runner, added May 2026)
- **Qazi** - 1 marathon (Philadelphia, added May 2026)

### Current Model Performance (May 2026)
- **Best Model:** Random Forest
- **CV MAE:** ±13.9 minutes (5-fold cross-validation)
- **Holdout MAE:** ±6.6 minutes (5 races, 4 runners)
- **Dataset:** 41 marathons, 5 runners (31 training, 5 holdout)
- **Top Features:** Weekly mileage (18%), runs/week (13%), peak mileage (9.5%), tempo workouts (8.1%)

### Holdout Predictions (5 races, 4 runners)
| Runner | Race | Predicted | Actual | Error | Notes |
|--------|------|-----------|--------|-------|-------|
| Osman | Dec 2024 | 3:12 | 3:22 | -10.7 min | 90% humidity, struggles in heat |
| Salman | Oct 2025 | 3:13 | 2:56 | +16.4 min | Exceptional fitness, model can't extrapolate |
| Azeem | Houston 2026 | 3:26 | 3:22 | +4.1 min | Good prediction |
| Sara | Boston 2025 | 3:27 | 3:24 | +2.6 min | Good prediction |
| Sara | London 2026 | 3:26 | 3:22 | +3.8 min | Good prediction |

**Average error: 7.5 minutes**
Note: Sara is only female runner - sex_encoded feature present but needs more female data
Note: Qazi only has 1 race so used for training only (no holdout)

---

## Previous Session Summary

### Major Investigation Completed
**Problem:** Salman's marathon prediction was 45 minutes off (predicted 3:40, actual 2:55)

**Root Cause:** FIT file parsing silently failed for 90% of files (79/88 corrupted)
- `fitparse` library rejected files with `Invalid field size` errors
- Cache only had 8 running activities when 88 existed
- Feature extraction showed 8.4 mi/week instead of actual 69.3 mi/week

**Solution:** Created `build_cache_from_csv.py` to bypass corrupted FIT files
- CSV has all activity data (distance, time, HR, elevation)
- Rebuilt cache with 4,280 activities
- Prediction error dropped from 45.4 → 8.8 minutes

### Files Created in Previous Sessions
- `build_cache_from_csv.py` - CSV-based cache builder
- `debug_feature_extraction.py` - Investigation script
- `diagnose_missing_activities.py` - FIT parsing failure diagnosis
- Multiple extraction/training scripts

---

## Session Log

### 2026-05-23 (continued) - Added Salman Khan (6th runner)
- Processed new runner "Salman Khan" from downloads/salman_khan
- 1604 runs, 31 marathons (2017-2026)
- PR: 3:05 (Mesa Phoenix 2020)
- 6 Major Marathon Stars completed (Tokyo 2023)
- Updated combined dataset: now 72 races from 6 runners
- Retrained model: holdout MAE improved 7.5 → 5.5 minutes
- Sara's Boston prediction now 0.1 min error (nearly perfect!)
- Updated Hugging Face model.pkl

### 2026-05-23 (continued) - Hugging Face Deployment & Documentation
- Created Gradio web app for marathon prediction (huggingface_app/)
  - Upload Strava export zip, get prediction
  - Shows training insights, confidence range, warnings
- Created comprehensive README.md covering all rubric criteria:
  - Dataset understanding (5 runners, 37 races, 48 features)
  - Data cleaning (FIT corruption, humidity normalization)
  - Algorithm (Random Forest, why chosen)
  - Metrics (MAE, CV, holdout validation)
  - Deployment (Hugging Face ready)
  - LLM conversation reflection
- Saved trained model as model.pkl for deployment
- All code committed and pushed

### 2026-05-23 (continued) - Deep Investigation of Prediction Errors
- Investigated why Salman predicted 16 min slow (3:13 vs 2:56 actual)
  - Found bonked race filter was incorrectly filtering Salman's good Oct 2025 race
  - Fixed to use (runner_id, race_id) tuples instead of just race_id
  - Found humidity stored inconsistently (0-1 vs 0-100) - normalized all to 0-1
- Investigated why Osman predicted 10 min fast (3:12 vs 3:22 actual)
  - Found individual weather sensitivity varies dramatically by runner
  - Salman handles 90% humidity fine; Osman slows 18+ min
- Added `historical_pr_minutes` feature (now #3 importance at 12%)
- Documented model limitations for final report

### 2026-05-23 (continued) - Quality Features & Qazi
- Added 5th runner Qazi (1 marathon - Philadelphia 3:53)
- Investigated why quality workout features were all zeros
- Root cause: CSV cache had pace data but extraction wasn't using it
- Added quality feature extraction from CSV for all runners:
  - tempo runs (<8:00 pace), fast runs (<7:30 pace)
  - `tempo_workout_count`, `quality_workout_percent`
- Sara's quality features updated: Boston 26%, London 22%
- Salman's quality features: 39% quality, 21 tempo runs
- Model improved: 9.4 → 6.6 min avg error (30% improvement)
- `tempo_workout_count` now #4 most important feature (8.1%)

### 2026-05-23 - Added Sara (4th runner)
- Processed Sara's Strava export (1004 runs, 11 marathons)
- Built cache from CSV, extracted features
- Updated holdout validation to support all runners dynamically
- Model accuracy improved: 7.0 min avg error (was 21.5)
- Best model: Random Forest, CV MAE 12 min

### 2026-05-20 - Session Recovery
- User frustrated about lost context (no memory between sessions)
- Read all summary files to restore context
- Created this SESSION_NOTES.md file for future reference

---

## Quick Reference

### Key Commands
```bash
# Build cache from activities
python build_cache.py /path/to/strava/activities
python build_cache_from_csv.py  # CSV fallback

# Extract features
python test_feature_extraction.py

# Train model
python train_race_model.py my_races.json

# Predict
python predict_race_time.py 26.2  # Marathon
```

### Known Issues
1. FIT file parsing fails for some Garmin firmware versions
2. CSV cache lacks track points (no zone distribution, cardiac drift)
3. Need to rebuild user's cache from CSV (may have similar issues)

---

## Model Limitations & Analysis (May 2026)

### Why Salman's Prediction is 16 min Off (predicted 3:13, actual 2:56)

**Root Cause: Severe data imbalance**

Dataset composition:
- Sub-3:00 races: **4 total** (all Salman)
- 3:00-3:30 races: 23
- 3:30+ races: 14
- Average finish time: **3:26**

The model has only 4 sub-3 examples to learn from! Even on Salman's TRAINING data, it predicts 5-6 min slow:
- Jul 2024: Actual 2:57, Predicted 3:03 (+6 min)
- Dec 2024: Actual 2:55, Predicted 3:00 (+5 min)
- Jul 2025: Actual 2:55, Predicted 3:00 (+5 min)

Random Forest averages across trees trained on 90% slower data, pulling predictions toward the mean (3:26).

**Salman's training volume is correct (61.6 mi/wk recent avg)** - the issue is the model doesn't have enough fast-runner examples to learn what that training means for a 2:55 runner vs a 3:22 runner.

**Key insight:** Same training volume means different things for different runners. Salman runs 2:55 with 60 mi/wk while Osman runs 3:22 with similar volume. The model can't distinguish this with only 4 sub-3 examples.

### Why Osman's Prediction is 10 min Off (predicted 3:12, actual 3:22)

**Root Cause: Individual weather sensitivity not captured**
- Dec 2024 race had **90% humidity**
- Model predicts ~8 min slowdown, but Osman actually slowed 18 min from his PR
- Osman consistently struggles in high humidity; Salman handles it well

**Weather sensitivity by runner (high humidity >85%):**
| Runner | Typical Slowdown from PR |
|--------|-------------------------|
| Salman | -2 to +3 min (heat tolerant) |
| Osman | +18 to +41 min (struggles) |
| Sara | +13 to +24 min (struggles) |

### Features Added to Address These Issues
- `historical_pr_minutes` - Runner's best prior marathon (now #3 importance at 12%)
- Fixed humidity normalization (was inconsistent 0-1 vs 0-100 across runners)
- Fixed bonked race filter to use (runner_id, race_id) tuples

### Fundamental Limitations
1. **Small dataset per runner** - Only 5-17 races per person, not enough to learn individual patterns
2. **No runner-specific weather sensitivity** - Would need more data to estimate
3. **Cross-runner generalization** - Model assumes similar training → similar results, but baseline fitness varies
4. **Race execution** - Pacing strategy, nutrition, mental state not captured in training data

### Recommendations for Future Improvement
- [ ] **Add more fast runners** - Currently only 4 sub-3 races (all Salman). Need more data at the fast end.
- [ ] **Use sample weighting** - Upweight rare sub-3 examples during training
- [ ] **Try quantile regression** - Predict a range rather than point estimate
- [ ] Add runner-specific baseline pace from easy runs
- [ ] Calculate individual weather sensitivity coefficient (needs more data)
- [ ] Consider separate models per runner (if dataset grows)
- [ ] Add race elevation profile as a feature
- [ ] Track taper quality more precisely

---

### Next Steps (from previous sessions)
- [ ] Rebuild user's cache from CSV
- [ ] Add logging to cache builder for failed files
- [ ] Enhance feature engineering for CSV data
- [ ] Investigate FIT file corruption root cause
