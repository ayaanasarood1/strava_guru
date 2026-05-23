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

### Runners in Dataset (40 marathons total)
- **Osman** - 7 marathons
- **Salman** - 17 marathons
- **Azeem** - 5 marathons
- **Sara** - 11 marathons (female runner, added May 2026)

### Current Model Performance
- **Best Model:** Random Forest
- **CV MAE:** ±12.0 minutes
- **Holdout MAE:** ±7.0 minutes (4 runners)
- **Dataset:** 40 marathons, 4 runners
- **Top Features:** Weekly mileage (23%), runs/week (15%), consistency (8%)

### Holdout Predictions (4 runners)
| Runner | Race | Predicted | Actual | Error |
|--------|------|-----------|--------|-------|
| Osman | Dec 2024 Marathon | 3:12 | 3:22 | 10.4 min |
| Salman | Jack & Jill 2025 | 3:04 | 2:55 | 9.2 min |
| Azeem | Houston 2026 | 3:23 | 3:22 | 1.1 min |
| Sara | London 2026 | 3:29 | 3:22 | 7.2 min |

**Average error: 7.0 minutes** (improved from 21.5 with Sara's data)

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
