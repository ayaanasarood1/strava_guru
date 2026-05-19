# Strava Activity Visualizations

## What's Been Created

Your Strava activity data can now be visualized with beautiful charts similar to what you see on Strava's website!

## Generated Charts Location

All visualizations are saved in: `~/PycharmProjects/strava_guru/charts/`

## Chart Types

### 1. Full Report (`*_full_report.png`)
A comprehensive 4-panel dashboard showing:

**Top Left - Pace Analysis Bar Chart**
- Bars showing pace for each mile (like Strava's pace analysis)
- Elevation gain overlay (gray area behind bars)
- Helps identify fast/slow miles at a glance

**Top Right - Elevation Profile**
- Shows elevation changes over the entire route
- Gray filled area shows the terrain
- Displays total elevation gain/loss

**Bottom Left - Heart Rate Over Distance**
- Red line showing heart rate throughout the run
- Shows average heart rate as a dashed line
- Helps track effort and recovery during the run

**Bottom Right - Split Tables**
- **Summary Stats**: Distance, time, pace, GAP, elevation, HR, calories
- **Mile Splits Table**: Each mile with time, pace, GAP, and heart rate

### 2. Pace Chart Only (`*_pace.png`)
- Just the pace analysis bar chart
- Useful for quick pace comparison

### 3. Route Map (`*_route.png`)
- GPS route plotted on a coordinate grid
- Green dot = Start, Red dot = Finish
- Mile markers shown as numbered circles
- Great for visualizing the route shape

## How to Use

### Visualize a Single Activity
```bash
cd ~/PycharmProjects/strava_guru
source .venv/bin/activate

python visualizer.py ~/Downloads/export_40402578/activities/12345.fit.gz ./charts/
```

This creates:
- `12345_full_report.png` - Complete dashboard
- `12345_pace.png` - Just pace chart
- `12345_route.png` - Route map

### Batch Process Multiple Activities
```bash
# Process first 10 activities
python visualize_all.py ~/Downloads/export_40402578/activities/ ./charts/ --limit 10

# Process ALL activities (may take a while!)
python visualize_all.py ~/Downloads/export_40402578/activities/ ./charts/
```

This creates `*_report.png` files for each activity.

## Example Output

Your charts are ready in: `~/PycharmProjects/strava_guru/charts/`

Currently generated:
- `10003640184_report.png` - Full report for activity 10003640184
- `10010706773_report.png` - Full report for activity 10010706773
- `10016480375_report.png` - Full report for activity 10016480375
- `9997205444_full_report.png` - Full report with all 3 chart types
- `9997205444_pace.png` - Pace chart only
- `9997205444_route.png` - Route map only

## What Each Metric Means

**Pace**: Minutes per mile (moving time ÷ distance)
**GAP** (Grade Adjusted Pace): Pace adjusted for elevation - shows what your pace would be on flat ground
**Elevation Gain**: Total feet climbed during the run
**Heart Rate**: Average BPM throughout the activity
**Moving Time**: Time spent actually moving (excludes stops)

## Tips

1. **Compare Runs**: Generate charts for multiple runs on the same route to see progress
2. **Identify Patterns**: Use pace bars to see which miles were hardest
3. **Heart Rate Analysis**: See if you're maintaining steady effort or going out too fast
4. **Elevation Impact**: Compare pace vs GAP to understand how hills affected your run

## Next Steps

You could:
- Generate charts for all your runs to create a visual history
- Compare multiple runs side-by-side
- Track progress on specific routes over time
- Export data to CSV for deeper analysis
