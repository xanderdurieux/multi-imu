# multi-imu

A modular Python toolkit for loading, synchronizing, aligning, visualizing, and analyzing data from multiple IMU sensors such as your Arduino (≈50 Hz, ±4g) and custom (≈120 Hz, ±16g) units.

## Features
- CSV loading with column normalization.
- Resampling, gravity removal, and axis normalization utilities.
- Time-offset estimation via cross-correlation and synchronized stream trimming.
- Axis alignment using best-fit rotation matrices.
- Event analytics for falls, harsh braking, and aggressive turns.
- Matplotlib visualizations for sensor comparisons and event overlays.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart
The `examples/pipeline_example.py` script generates synthetic data and runs the full pipeline:
```bash
python examples/pipeline_example.py
```
Replace the synthetic generators with your own CSV files using `multi_imu.load_imu_csv`.

### Converting raw logs
Use `examples/convert_raw.py` to translate raw Arduino or Sporsa captures into the
library's standardized CSV schema using device-recorded timestamps:

```bash
python examples/convert_raw.py --arduino-log path/to/arduino.log --output-dir output/drive1
python examples/convert_raw.py --sporsa-log path/to/sporsa.log --output-dir output/drive1
```

### Typical workflow
1. **Load streams**
   ```python
   from multi_imu import load_imu_csv

   arduino = load_imu_csv("arduino.csv", name="arduino", sample_rate_hz=50.0)
   custom = load_imu_csv("custom.csv", name="custom", sample_rate_hz=120.0)
   ```
2. **Resample target to match reference rate**
   ```python
   from multi_imu import resample_signal
   custom_resampled = resample_signal(custom, target_rate_hz=arduino.sample_rate_hz)
   ```
3. **Synchronize in time**
   ```python
   from multi_imu import synchronize_streams
   synced = synchronize_streams(reference=arduino, target=custom_resampled)
   print("Offset (s):", synced.offset_seconds)
   ```
4. **Align axes**
   ```python
   from multi_imu import compute_alignment_matrix, align_axes
   alignment_matrix = compute_alignment_matrix(synced.reference, synced.target)
   aligned_target = align_axes(synced.target, alignment_matrix)
   ```
5. **Remove gravity and analyze events**
   ```python
   from multi_imu import remove_gravity, detect_falls, detect_braking_events, detect_turns
   arduino_hp = remove_gravity(synced.reference)
   events = detect_falls(arduino_hp) + detect_braking_events(arduino_hp) + detect_turns(arduino_hp)
   ```
6. **Visualize**
   ```python
   from multi_imu import plot_comparison, plot_event_annotations
   plot_comparison(arduino_hp, aligned_target, columns=["ax", "ay", "az"])
   plot_event_annotations(arduino_hp, events)
   ```

Adjust thresholds and column selections to suit your specific motion patterns and calibration motions.
