# Strava Guru

A Python tool to analyze Strava activity files (.fit and .gpx) and compute detailed statistics similar to what's shown on Strava's website.

## Features

- **Parse FIT and GPX files** (including gzipped versions)
- **Calculate comprehensive stats**:
  - Distance, pace, moving time, elapsed time
  - Grade Adjusted Pace (GAP)
  - Elevation gain/loss
  - Heart rate metrics (avg, max)
  - Calorie estimates
- **Mile splits** with per-split stats:
  - Pace and GAP for each mile
  - Elevation changes
  - Heart rate
- **Detailed lap analysis**

## Installation

1. Create and activate virtual environment:
```bash
cd ~/PycharmProjects/strava_guru
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Analyze Activities (Text Output)

```bash
python activity_analyzer.py <path_to_activity_file>
```

Examples:
```bash
# Analyze a single activity
python activity_analyzer.py ~/Downloads/export_40402578/activities/12345.fit.gz

# Analyze a GPX file
python activity_analyzer.py ~/Downloads/export_40402578/activities/12345.gpx
```

### Visualize Activities (Generate Charts)

```bash
python visualizer.py <path_to_activity_file> [output_directory]
```

Examples:
```bash
# Generate visualizations for a single activity
python visualizer.py ~/Downloads/export_40402578/activities/12345.fit.gz ./charts/

# Batch process multiple activities (limit to 10)
python visualize_all.py ~/Downloads/export_40402578/activities/ ./charts/ --limit 10

# Process all activities
python visualize_all.py ~/Downloads/export_40402578/activities/ ./charts/
```

### Generated Visualizations

The visualizer creates three types of charts:

1. **Full Report** (`*_full_report.png`) - Comprehensive dashboard with:
   - Pace analysis bar chart (like Strava)
   - Elevation profile over distance
   - Heart rate over distance
   - Summary statistics table
   - Mile splits table

2. **Pace Chart** (`*_pace.png`) - Just the pace analysis bars

3. **Route Map** (`*_route.png`) - GPS route with mile markers

## Output

The analyzer displays:

```
================================================================================
ACTIVITY SUMMARY
================================================================================
Activity Type: Run
Date: Thursday, January 29, 2026 at 07:45 AM

Distance:       18.12 mi
Moving Time:    2:21:55
Elapsed Time:   2:24:09
Pace:           7:50 /mi
GAP:            7:47 /mi

Elevation Gain: 436 ft
Elevation Loss: 436 ft
Elev Range:     203 - 274 ft

Avg Heart Rate: 156 bpm
Max Heart Rate: 171 bpm

Calories:       1812

================================================================================
MILE SPLITS
================================================================================
Mile   Distance     Time       Pace       GAP        Elev ↑     HR
--------------------------------------------------------------------------------
1      1.00 mi     8:50       8:50       8:24       -31 ft     134
2      1.00 mi     8:20       8:20       8:23       -21 ft     167
3      1.00 mi     8:14       8:14       8:10       1 ft       158
...
================================================================================
```

## Data Source

Export your Strava data from: https://www.strava.com/athlete/delete_your_account

This provides a zip file with:
- `activities/` - Individual .fit.gz files for each activity
- `activities.csv` - Summary of all activities
- Routes, segments, and other data

## How It Works

1. **FIT File Parsing**: Uses `fitparse` library to decode binary FIT files
2. **GPS Track Processing**: Extracts lat/lon/elevation/heart rate from each track point
3. **Distance Calculation**: Uses Haversine formula for GPS coordinates
4. **Pace Analysis**: Computes moving time pace and grade-adjusted pace
5. **Split Generation**: Automatically generates mile splits with statistics

## Notes

- GAP (Grade Adjusted Pace) adjusts your pace based on elevation grade
- Moving time excludes periods where speed < 0.5 m/s (stopped)
- Calorie estimates use ~100 calories per mile
- Heart rate data is only available if recorded by device
