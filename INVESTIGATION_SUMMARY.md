# Investigation Summary: Feature Extraction Bug Resolution

## Problem

Holdout validation showed massive prediction error for Salman's Jack & Jill 2025 marathon:
- **Predicted:** 3:40 (220 minutes)
- **Actual:** 2:55 (175 minutes)
- **Error:** 45.4 minutes (26% off)

Meanwhile, your marathon prediction was accurate (11.6 min error).

## Root Cause Investigation

### Step 1: Validate Training Data from CSV
Checked `activities.csv` directly to verify actual training volume:
- **Feature extraction claimed:** 8.4 mi/week
- **Actual from CSV:** 69.3 mi/week
- **Discrepancy:** OFF BY 60.9 MILES PER WEEK (8x underestimate!)

### Step 2: Debug Cache Database
Created `debug_feature_extraction.py` to trace the issue:
- Cache had 1,869 total activities
- Only **17 activities** in training window (expected 88)
- Only **8 running activities** cached (expected 88)
- **ROOT CAUSE:** Cache building process failed to cache 71 out of 88 activities

### Step 3: Diagnose File Parsing Failures
Created `diagnose_missing_activities.py` to check individual files:
- All 88 files exist on disk ✓
- **79 out of 88 files (90%) failed to parse**
- Error: `FitParseError: Invalid field size 1 for type 'uint32' (expected a multiple of 4)`

**Conclusion:** Strava FIT files from April-July 2025 had corrupted/non-standard binary structures that the `fitparse` library rejected.

## Solution

### Approach: CSV-Based Cache Fallback
Since FIT files were corrupted but `activities.csv` had all the data, created alternative cache builder:

**File:** `build_cache_from_csv.py`
- Parses `activities.csv` directly
- Extracts distance, duration, date, heart rate, elevation
- Bypasses corrupted FIT files entirely

**Results:**
- Cached **4,280 activities** (vs 1,869 from FIT parsing)
- 2,752 with HR data
- Date range: 2010-2026

### Re-extraction Results

**Before (corrupted cache):**
```
2024-07-27: 2:58 - 16.2 mi/wk
2024-12-08: 2:56 - 11.8 mi/wk
2025-04-21: 3:19 - 12.2 mi/wk
2025-07-27: 2:55 - 8.4 mi/wk   ← WORST
2025-10-12: 2:57 - 18.3 mi/wk
```

**After (CSV-based cache):**
```
2024-07-27: 2:58 - 58.6 mi/wk (+42.5)
2024-12-08: 2:56 - 63.5 mi/wk (+51.6)
2025-04-21: 3:19 - 53.8 mi/wk (+41.5)
2025-07-27: 2:55 - 72.3 mi/wk (+64.0) ← FIXED!
2025-10-12: 2:57 - 63.2 mi/wk (+44.9)
```

## Impact on Model Performance

### Holdout Validation Results

**Before Cache Fix:**
- Your marathon: 11.6 min error
- Salman's marathon: **45.4 min error** ❌

**After Cache Fix:**
- Your marathon: **9.6 min error** ✓
- Salman's marathon: **8.8 min error** ✓

**Improvement:** 5x reduction in Salman's prediction error (45.4 → 8.8 minutes)

### Final Model Accuracy

**Cross-Validation (5-fold):**
- Model: Random Forest
- MAE: ±21.6 minutes
- Dataset: 21 marathons (filtered 2 bonked races)

**Holdout Validation:**
- Average MAE: 9.2 minutes
- Model generalizes well (holdout MAE < CV MAE)

## Lessons Learned

### 1. Data Quality Issues in Real-World ML

This investigation revealed a critical data pipeline bug that was silently corrupting 90% of training data for specific time periods. Key takeaways:

- **Silent failures are dangerous:** The cache builder skipped corrupted files without logging, making the issue invisible
- **Validate against ground truth:** CSV validation caught the 8x underestimate
- **Test with multiple data sources:** Having both FIT files and CSV provided a fallback

### 2. FIT File Parsing Brittleness

The `fitparse` library is strict about binary format compliance:
- Rejects files with field size mismatches
- No lenient parsing mode for partial data extraction
- Common with certain Garmin device firmware versions

**Mitigation strategies:**
1. CSV fallback (what we implemented)
2. Alternative FIT parsers (e.g., Garmin FIT SDK)
3. Request re-export from Strava with corrected files

### 3. Feature Engineering Pipeline Robustness

The pipeline should handle:
- Missing track points (CSV doesn't have granular data)
- Partial data (some features computable, others not)
- Fallback values for intensity metrics when HR zones unavailable

**Current limitations:**
- CSV cache has 0% for all zone distribution features
- No cardiac drift or terrain analysis (requires track points)
- Model still works with 31/41 features populated

## Files Modified/Created

### Investigation Scripts
- `debug_feature_extraction.py` - Traced cache query issue
- `verify_salman_training.py` - Validated against CSV ground truth
- `diagnose_missing_activities.py` - Identified FIT parsing failures

### Solution Implementation
- `build_cache_from_csv.py` - Alternative cache builder from CSV
- `reextract_salman_jack_jill.py` - Fixed single race features
- `reextract_all_salman_races.py` - Rebuilt all 17 Salman marathons
- `manual_update_features.py` - Verified dataset updates

### Bug Fixes
- `activity_analyzer.py` - Added `check_crc=False` flag (didn't help)
- `terrain_handling.py` - Fixed None pace value handling

## Remaining Work

### 1. Rebuild User's Cache from CSV
Your cache likely has similar issues for recent races. Should run:
```bash
python build_cache_from_csv.py  # (adapted for your data)
python reextract_all_user_races.py
```

### 2. Add Logging to Cache Builder
Modify `activity_cache.py` line 326 to log failed files:
```python
except Exception as e:
    errors += 1
    logging.warning(f"Failed to parse {file_path.name}: {e}")
    continue
```

### 3. Enhance Feature Engineering for CSV Data
When track points unavailable:
- Use average HR instead of zone distribution
- Estimate intensity from pace distribution
- Flag races with incomplete features

### 4. Investigate FIT File Corruption
Root cause could be:
- Specific Garmin device model
- Firmware version during recording
- Strava export process bug
- File transfer corruption

## Conclusion

Successfully resolved critical data quality bug that was causing 8x underestimate of training volume for 5 recent marathons. The CSV-based fallback restored correct feature extraction, improving holdout prediction error from 45.4 minutes to 8.8 minutes (5x improvement).

**Final model accuracy: ±21.6 minutes on 21 marathons**

This investigation demonstrates the importance of:
1. Comprehensive data validation
2. Multiple data source redundancy
3. Robust error handling in data pipelines
4. Ground truth verification for ML features
