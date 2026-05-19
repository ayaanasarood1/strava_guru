# Strava Activity Analyzer - Usage Guide

## Quick Start

```bash
cd ~/PycharmProjects/strava_guru
source .venv/bin/activate
python activity_analyzer.py ~/Downloads/export_40402578/activities/<activity_file>.fit.gz
```

## What It Calculates

The analyzer computes the same statistics you see on Strava's website:

### Overview Stats
- **Distance** - Total miles
- **Moving Time** - Time spent moving (excludes stops)
- **Elapsed Time** - Total time from start to finish
- **Pace** - Minutes per mile (moving time)
- **GAP** - Grade Adjusted Pace (adjusts for elevation)
- **Elevation Gain/Loss** - Total feet climbed and descended
- **Heart Rate** - Average and maximum BPM
- **Calories** - Estimated calories burned

### Mile Splits
For each mile, it shows:
- Split time
- Pace for that mile
- GAP for that mile
- Elevation gain
- Average heart rate

## Examples

### Analyze a single run
```bash
python activity_analyzer.py ~/Downloads/export_40402578/activities/9997205444.fit.gz
```

### Analyze multiple runs
```bash
for file in ~/Downloads/export_40402578/activities/*.fit.gz; do
    python activity_analyzer.py "$file"
    echo "---"
done
```

### Test with sample data
```bash
python test_analyzer.py
```

## Supported File Formats

- `.fit` - FIT files (Garmin, Apple Watch, etc.)
- `.fit.gz` - Gzipped FIT files (Strava export format)
- `.gpx` - GPX files
- `.gpx.gz` - Gzipped GPX files

## How It Works

1. **Parses FIT/GPX files** - Extracts GPS track points with timestamps, location, elevation, heart rate
2. **Calculates distances** - Uses Haversine formula between GPS coordinates
3. **Computes pace** - Divides moving time by distance
4. **Adjusts for grade** - GAP factors in elevation changes to show equivalent flat-ground pace
5. **Generates splits** - Breaks activity into mile segments with detailed stats

## Data Sources

The analyzer reads the following from your activity files:
- GPS coordinates (latitude/longitude)
- Timestamps
- Elevation data
- Heart rate (if available from device)
- Cadence (if available)
- Speed (if available)

## Limitations

- Elevation data may not be available in all FIT files (shows 0 if missing)
- Calorie estimates are rough (~100 cal/mile)
- GAP formula is simplified (Strava uses more sophisticated algorithms)
- Only shows mile splits (not km splits)
- No segment matching (Strava's segment feature requires server-side data)

## Next Steps

You could extend this to:
- Export data to CSV/JSON
- Generate visualizations (pace charts, elevation profiles)
- Compare multiple activities
- Track progress over time
- Build a web interface
- Match against known segments (would need segment database)
