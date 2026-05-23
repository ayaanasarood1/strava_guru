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
| Runner | Race | Predicted | Actual | Error |
|--------|------|-----------|--------|-------|
| Osman | Dec 2024 Marathon | 3:12 | 3:22 | 10.4 min |
| Salman | Jul 2025 Marathon | 3:05 | 2:55 | 10.1 min |
| Azeem | Houston 2026 | 3:27 | 3:22 | 5.4 min |
| Sara | Boston 2025 | 3:28 | 3:24 | 3.4 min |
| Sara | London 2026 | 3:26 | 3:22 | 3.9 min |

**Average error: 6.6 minutes**
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

### Next Steps (from previous sessions)
- [ ] Rebuild user's cache from CSV
- [ ] Add logging to cache builder for failed files
- [ ] Enhance feature engineering for CSV data
- [ ] Investigate FIT file corruption root cause
