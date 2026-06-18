# ETRI Lifelog Acceleration Modality Audit

Date: 2026-06-19

## Verdict

**B) Raw accelerometer unavailable, derived activity features only**

The original `ch2025_data_items` parquet files do not contain raw tri-axial accelerometer columns such as `x`, `y`, `z`, `acc_x`, `acc_y`, `acc_z`, or waveform/list-like acceleration samples. The acceleration-adjacent modalities available in the original lifelog are derived activity/pedometer features.

## Original Data Item Inventory

```text
ch2025_mACStatus.parquet      cols=[subject_id, timestamp, m_charging]
ch2025_mActivity.parquet      cols=[subject_id, timestamp, m_activity]
ch2025_mAmbience.parquet      cols=[subject_id, timestamp, m_ambience]
ch2025_mBle.parquet           cols=[subject_id, timestamp, m_ble]
ch2025_mGps.parquet           cols=[subject_id, timestamp, m_gps]
ch2025_mLight.parquet         cols=[subject_id, timestamp, m_light]
ch2025_mScreenStatus.parquet  cols=[subject_id, timestamp, m_screen_use]
ch2025_mUsageStats.parquet    cols=[subject_id, timestamp, m_usage_stats]
ch2025_mWifi.parquet          cols=[subject_id, timestamp, m_wifi]
ch2025_wHr.parquet            cols=[subject_id, timestamp, heart_rate]
ch2025_wLight.parquet         cols=[subject_id, timestamp, w_light]
ch2025_wPedo.parquet          cols=[subject_id, timestamp, step, step_frequency, running_step, walking_step, distance, speed, burned_calories]
```

No file exposes raw tri-axial acceleration.

## Candidate Acceleration-Adjacent Files

### `data/ch2025_data_items/ch2025_mActivity.parquet`

- File size: `4,642,475` bytes
- Rows: `961,062`
- Storage format: parquet, scalar columns
- Schema:

```text
subject_id: string
timestamp: timestamp[ns]
m_activity: int64
```

- Observed timestamp cadence: mostly 60 seconds
- Effective cadence: 1 row/minute aggregate, not raw accelerometer Hz
- Explicit null ratio:

```text
subject_id    0.0
timestamp     0.0
m_activity    0.0
```

- Minute-grid coverage gap ratio between each subject's min/max timestamp: `0.213965`
- Observed `m_activity` values: `[0, 1, 3, 4, 7, 8]`

Sample records:

```text
subject_id  timestamp            m_activity
id01        2024-06-26 12:03:00  4
id01        2024-06-26 12:04:00  0
id01        2024-06-26 12:05:00  0
id01        2024-06-26 12:06:00  0
id01        2024-06-26 12:07:00  0
id01        2024-06-26 12:08:00  0
id01        2024-06-26 12:09:00  0
id01        2024-06-26 12:10:00  0
id01        2024-06-26 12:11:00  3
id01        2024-06-26 12:12:00  3
id01        2024-06-26 12:13:00  3
id01        2024-06-26 12:14:00  3
```

Interpretation: `m_activity` is a derived categorical activity state, likely from phone activity recognition. It is not raw acceleration.

### `data/ch2025_data_items/ch2025_wPedo.parquet`

- File size: `4,769,326` bytes
- Rows: `748,100`
- Storage format: parquet, scalar columns
- Schema:

```text
subject_id: string
timestamp: timestamp[ns]
step: int64
step_frequency: double
running_step: int64
walking_step: int64
distance: double
speed: double
burned_calories: double
```

- Observed timestamp cadence: mostly 60 seconds
- Effective cadence: 1 row/minute pedometer aggregate, not raw accelerometer Hz
- Explicit null ratio:

```text
subject_id         0.0
timestamp          0.0
step               0.0
step_frequency     0.0
running_step       0.0
walking_step       0.0
distance           0.0
speed              0.0
burned_calories    0.0
```

- Minute-grid coverage gap ratio between each subject's min/max timestamp: `0.387760`
- Numeric summary:

```text
step:            min=0, median=0, max=317, mean=3.4516
step_frequency:  min=0, median=0, max=5.2833, mean=0.0575
distance:        min=0, median=0, max=302.2402, mean=2.5790
speed:           min=0, median=0, max=5.0373, mean=0.0430
burned_calories: min=0, median=0, max=310.6953, mean=0.1524
running_step:    only 0 observed
walking_step:    only 0 observed
```

Sample records:

```text
subject_id  timestamp            step  step_frequency  running_step  walking_step  distance  speed     burned_calories
id01        2024-06-26 12:09:00  10    0.166667        0             0             8.33      0.138833  0.0
id01        2024-06-26 12:10:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:11:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:12:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:13:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:14:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:15:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:16:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:17:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:18:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:19:00  0     0.000000        0             0             0.00      0.000000  0.0
id01        2024-06-26 12:20:00  0     0.000000        0             0             0.00      0.000000  0.0
```

Interpretation: `wPedo` is a derived pedometer/activity aggregate: step count, step frequency, distance, speed, and calorie estimates. It is not raw acceleration.

## Required Checks

1. Raw tri-axial acceleration `(x, y, z)` exists?

No. No parquet file contains tri-axial acceleration columns or list-like raw acceleration samples.

2. If raw tri-axial exists, sampling frequency, columns, storage, missing ratio?

Not applicable. Raw tri-axial acceleration is unavailable. The observed derived activity files are minute-level aggregates, with dominant timestamp gap of 60 seconds. This is not an accelerometer sampling frequency.

3. If raw tri-axial is absent, what derived indicators exist?

- `mActivity`: derived phone activity category/state (`m_activity`)
- `wPedo`: derived pedometer aggregates (`step`, `step_frequency`, `distance`, `speed`, `burned_calories`, with `running_step` and `walking_step` present but observed as all zero)

4. Actual parquet files opened?

Yes. The audit directly opened:

- `data/ch2025_data_items/ch2025_mActivity.parquet`
- `data/ch2025_data_items/ch2025_wPedo.parquet`

## Final Conclusion

**B) Raw accelerometer unavailable, derived activity features only**

