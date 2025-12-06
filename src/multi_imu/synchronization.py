"""
LIDA (Linear Interpolation Data Alignment) synchronization for IMUSensorData.

LIDA uses:
1. Cross-correlation to find optimal time offset
2. Affine regression to map peripheral timestamps to central timeline
3. Linear interpolation for fine-grained data alignment
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from multi_imu.data_models import IMUSensorData


def _get_accel_axes(data: pd.DataFrame) -> tuple[str, str, str]:
    """Get acceleration axis column names from dataframe."""
    # Check for standard column names
    if all(col in data.columns for col in ["ax", "ay", "az"]):
        return ("ax", "ay", "az")
    elif all(col in data.columns for col in ["x", "y", "z"]):
        return ("x", "y", "z")
    else:
        raise ValueError(
            f"Expected acceleration columns (ax/ay/az or x/y/z) in data, got: {data.columns.tolist()}"
        )


def _compute_magnitude(data: pd.DataFrame) -> np.ndarray:
    """Compute acceleration magnitude from x, y, z components."""
    axes = _get_accel_axes(data)
    return np.sqrt(sum(data[axis] ** 2 for axis in axes)).values


def _estimate_affine_transformation(
    t_peripheral: np.ndarray, t_central: np.ndarray
) -> tuple[float, float]:
    """
    Estimate affine transformation parameters: t_central = a * t_peripheral + b
    
    Uses linear regression to find slope (a) and intercept (b).
    """
    # Reshape for sklearn
    t_peripheral_reshaped = t_peripheral.reshape(-1, 1)
    
    # Fit linear regression
    reg = LinearRegression()
    reg.fit(t_peripheral_reshaped, t_central)
    
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    return slope, intercept


def _find_time_offset_crosscorrelation(
    signal_ref: np.ndarray,
    t_ref: np.ndarray,
    signal_peripheral: np.ndarray,
    t_peripheral: np.ndarray,
    max_offset_s: float = 20.0,
) -> tuple[float, float, float]:
    """
    Find time offset using cross-correlation between reference and peripheral signals.
    
    Args:
        signal_ref: Reference signal magnitude
        t_ref: Reference timestamps (in seconds)
        signal_peripheral: Peripheral signal magnitude
        t_peripheral: Peripheral timestamps (in seconds)
        max_offset_s: Maximum offset to search (in seconds)
    
    Returns:
        Tuple of (offset_s, peak_correlation, peak_lag_s) where:
        - offset_s: Estimated time offset in seconds (to add to peripheral time)
        - peak_correlation: Peak correlation value
        - peak_lag_s: Lag time at peak correlation (in seconds)
    """
    # Resample both signals to a common time grid for cross-correlation
    t_min = max(t_ref.min(), t_peripheral.min())
    t_max = min(t_ref.max(), t_peripheral.max())
    
    if t_max <= t_min:
        return 0.0, 0.0, 0.0
    
    # Use a sampling rate that captures the signal dynamics
    # Use the minimum sampling interval from both signals
    dt_ref = np.median(np.diff(t_ref))
    dt_peripheral = np.median(np.diff(t_peripheral))
    dt = min(dt_ref, dt_peripheral)
    
    # Create common time grid
    t_common = np.arange(t_min, t_max, dt)
    
    # Interpolate both signals to common grid
    f_ref = interp1d(t_ref, signal_ref, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_peripheral = interp1d(
        t_peripheral, signal_peripheral, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    
    ref_interp = f_ref(t_common)
    peripheral_interp = f_peripheral(t_common)
    
    # Normalize signals
    ref_interp = (ref_interp - ref_interp.mean()) / (ref_interp.std() + 1e-10)
    peripheral_interp = (peripheral_interp - peripheral_interp.mean()) / (
        peripheral_interp.std() + 1e-10
    )
    
    # Compute cross-correlation
    correlation = signal.correlate(ref_interp, peripheral_interp, mode="full")
    lags = signal.correlation_lags(len(ref_interp), len(peripheral_interp), mode="full")
    lag_times = lags * dt
    
    # Find peak correlation within reasonable offset range
    valid_mask = np.abs(lag_times) <= max_offset_s
    if not valid_mask.any():
        return 0.0, 0.0, 0.0
    
    valid_correlation = correlation[valid_mask]
    valid_lag_times = lag_times[valid_mask]
    
    peak_idx = np.argmax(valid_correlation)
    peak_lag_time = valid_lag_times[peak_idx]
    peak_correlation = valid_correlation[peak_idx]
    
    # Lag interpretation for scipy.signal.correlate:
    # correlation[k] = sum(ref[n] * peripheral[n + k])
    # Positive lag k means peripheral is shifted forward (delayed) relative to ref
    # This means peripheral is currently BEHIND ref in time
    # So we need to ADD time to peripheral: offset = lag_time
    offset_s = peak_lag_time
    
    return offset_s, peak_correlation, peak_lag_time


def lida_synchronize(
    ref_sensor: IMUSensorData,
    peripheral_sensor: IMUSensorData,
    use_affine: bool = True,
    use_crosscorr: bool = True,
    try_both_directions: bool = False,
    max_offset_s: float = 20.0,
) -> tuple[float, float, dict]:
    """
    Apply LIDA algorithm to synchronize peripheral sensor to reference sensor.
    
    The LIDA algorithm:
    1. Uses cross-correlation to find initial time offset
    2. Uses affine regression to estimate clock drift (t_ref = a * t_peripheral + b)
    3. Uses linear interpolation for fine-grained alignment
    
    Args:
        ref_sensor: Reference IMUSensorData (central node)
        peripheral_sensor: Peripheral IMUSensorData to synchronize
        use_affine: Whether to use affine transformation for clock drift correction
        use_crosscorr: Whether to use cross-correlation for initial offset
        try_both_directions: Whether to try both correlation directions and pick best
        max_offset_s: Maximum time offset to search (in seconds)
    
    Returns:
        Tuple of (offset_s, slope, info) where:
        - offset_s: Estimated time offset in seconds (to add to peripheral time)
        - slope: Estimated clock drift (slope of affine transformation)
        - info: Dictionary with additional information (crosscorr_offset_s, crosscorr_peak, 
                affine_slope, affine_intercept, etc.)
    """
    # Extract data and timestamps
    df_ref = ref_sensor.data
    df_peripheral = peripheral_sensor.data
    
    # Ensure timestamp column exists
    if "timestamp" not in df_ref.columns or "timestamp" not in df_peripheral.columns:
        raise ValueError("Both sensors must have 'timestamp' column in their data")
    
    # Compute magnitude signals for alignment (more robust than individual axes)
    mag_ref = _compute_magnitude(df_ref)
    mag_peripheral = _compute_magnitude(df_peripheral)
    
    t_ref = df_ref["timestamp"].values
    t_peripheral = df_peripheral["timestamp"].values
    
    offset_s = 0.0
    slope = 1.0
    
    info = {}
    
    # Step 1: Find initial time offset using cross-correlation
    if use_crosscorr:
        offset_s, peak_corr, peak_lag = _find_time_offset_crosscorrelation(
            mag_ref, t_ref, mag_peripheral, t_peripheral, max_offset_s=max_offset_s
        )
        
        # Optionally try both directions to see which gives better alignment
        if try_both_directions:
            offset_s_opposite, peak_corr_opposite, _ = _find_time_offset_crosscorrelation(
                mag_peripheral, t_peripheral, mag_ref, t_ref, max_offset_s=max_offset_s
            )
            # If we correlate in opposite direction, the offset sign flips
            offset_s_opposite = -offset_s_opposite
            
            # Use the direction with higher correlation
            if peak_corr_opposite > peak_corr:
                offset_s = offset_s_opposite
                peak_corr = peak_corr_opposite
                info["used_opposite_direction"] = True
        
        info["crosscorr_offset_s"] = offset_s
        info["crosscorr_peak"] = peak_corr
        info["crosscorr_lag_s"] = peak_lag
    
    # Step 2: Apply initial offset and estimate affine transformation
    if use_affine:
        # Apply initial offset
        t_peripheral_offset = t_peripheral + offset_s
        
        # Find overlapping time range for affine regression
        t_overlap_min = max(t_ref.min(), t_peripheral_offset.min())
        t_overlap_max = min(t_ref.max(), t_peripheral_offset.max())
        
        if t_overlap_max > t_overlap_min + 1.0:  # Need at least 1 second overlap
            # Create paired timestamps by finding corresponding points
            # Use magnitude signal to find corresponding samples
            # Interpolate both signals to a common time grid
            dt = min(np.median(np.diff(t_ref)), np.median(np.diff(t_peripheral)))
            t_common = np.arange(t_overlap_min, t_overlap_max, dt)
            
            if len(t_common) > 10:  # Need sufficient samples
                f_ref = interp1d(
                    t_ref, mag_ref, kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                f_peripheral = interp1d(
                    t_peripheral_offset,
                    mag_peripheral,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                
                # Create paired timestamps for affine regression
                # Sample evenly across overlap region and pair corresponding times
                # After initial offset, we assume times are roughly aligned
                # and use closest matching or interpolation to create pairs
                n_samples = min(100, len(t_common) // 2)
                t_ref_samples = np.linspace(t_overlap_min, t_overlap_max, n_samples)
                
                # For each reference sample time, find corresponding peripheral time
                # by finding the peripheral time that gives best signal match
                t_periph_samples = []
                
                for t_r in t_ref_samples:
                    # Find peripheral times near this reference time
                    # Use a small search window
                    search_window_s = 0.2  # 200ms search window
                    candidates = t_peripheral_offset[
                        np.abs(t_peripheral_offset - t_r) <= search_window_s
                    ]
                    
                    if len(candidates) == 0:
                        # Fall back to closest time
                        idx = np.argmin(np.abs(t_peripheral_offset - t_r))
                        t_periph_samples.append(t_peripheral_offset[idx])
                    else:
                        # Use the candidate closest to reference time
                        # (after initial offset, they should be close)
                        best_idx = np.argmin(np.abs(candidates - t_r))
                        t_periph_samples.append(candidates[best_idx])
                
                # Estimate affine transformation: t_ref = slope * t_periph + intercept
                # This maps peripheral time to reference time
                slope, intercept = _estimate_affine_transformation(
                    np.array(t_periph_samples), t_ref_samples
                )
                
                info["affine_slope"] = slope
                info["affine_intercept"] = intercept
                info["n_paired_samples"] = len(t_periph_samples)
                
                # If slope deviates significantly from 1, there's clock drift
                if abs(slope - 1.0) > 0.001:
                    info["drift_detected"] = True
                    # The intercept represents additional offset after accounting for drift
                    # Adjust the total offset
                    offset_s = offset_s + intercept
    
    return offset_s, slope, info


def apply_synchronization(
    sensor: IMUSensorData,
    offset_s: float,
    slope: float = 1.0,
) -> IMUSensorData:
    """
    Apply synchronization offset and drift correction to an IMUSensorData object.
    
    Args:
        sensor: IMUSensorData to synchronize
        offset_s: Time offset to add (in seconds)
        slope: Clock drift correction factor (default 1.0 for no drift)
    
    Returns:
        New IMUSensorData with synchronized timestamps
    """
    if "timestamp" not in sensor.data.columns:
        raise ValueError("Sensor data must have 'timestamp' column")
    
    # Create a copy of the data
    synchronized_data = sensor.data.copy()
    
    # Apply offset and drift correction: t_sync = slope * t_original + offset
    synchronized_data["timestamp"] = slope * synchronized_data["timestamp"] + offset_s
    
    # Sort by new timestamp
    synchronized_data = synchronized_data.sort_values("timestamp").reset_index(drop=True)
    
    # Create new IMUSensorData with same name and sample rate
    return IMUSensorData(
        name=sensor.name,
        data=synchronized_data,
        sample_rate_hz=sensor.sample_rate_hz,
    )

