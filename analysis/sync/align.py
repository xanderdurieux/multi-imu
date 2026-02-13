import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


def _acc_magnitude(data: pd.DataFrame) -> np.ndarray:
	"""Return |a| from IMU dataframe with columns 'ax', 'ay', 'az'."""
	ax = data["ax"].to_numpy(dtype=np.float64)
	ay = data["ay"].to_numpy(dtype=np.float64)
	az = data["az"].to_numpy(dtype=np.float64)
	return np.sqrt(ax**2 + ay**2 + az**2)


def _to_relative_time(
	data: pd.DataFrame,
	time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, np.ndarray]:
	"""Return a copy with `time_col` shifted to start at zero and keep originals."""
	if time_col not in data.columns:
		raise ValueError(f"Expected time column '{time_col}' in data")

	orig_ts = data[time_col].to_numpy(dtype=np.float64)
	if orig_ts.size == 0:
		raise ValueError("Empty time series provided")

	t0 = orig_ts[0]
	rel_ts = orig_ts - t0

	out = data.copy()
	out[f"{time_col}_orig"] = orig_ts
	out[time_col] = rel_ts
	return out, orig_ts


def _estimate_sampling_rate(rel_ts: np.ndarray) -> float:
	"""Estimate nominal sampling rate (Hz) from relative timestamps."""
	if rel_ts.size < 2:
		raise ValueError("Need at least two samples to estimate sampling rate")
	dt = np.diff(rel_ts)
	dt_med = np.median(dt)
	if dt_med <= 0:
		raise ValueError("Non-increasing timestamps, cannot estimate sampling rate")
	return 1.0 / dt_med


def _cross_correlation_lag(
	ref_mag: np.ndarray,
	tgt_mag: np.ndarray,
	fs: float,
	max_lag_s: Optional[float] = None,
) -> float:
	"""Estimate SDA-style lag (s) to add to target using normalized cross-correlation."""
	if ref_mag.size == 0 or tgt_mag.size == 0:
		raise ValueError("Empty magnitude arrays for cross-correlation")

	# Normalize to avoid bias from amplitude scaling
	ref_z = (ref_mag - ref_mag.mean()) / (ref_mag.std() + 1e-12)
	tgt_z = (tgt_mag - tgt_mag.mean()) / (tgt_mag.std() + 1e-12)

	corr = np.correlate(tgt_z, ref_z, mode="full")
	lags = np.arange(-ref_z.size + 1, tgt_z.size)

	if max_lag_s is not None:
		max_lag_samples = int(max_lag_s * fs)
		mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
		corr = corr[mask]
		lags = lags[mask]

	best_idx = int(np.argmax(corr))
	best_lag_samples = lags[best_idx]

	# Positive best_lag_samples means target is *ahead* of reference in the
	# correlation sense. To align target to reference, we ADD a negative time
	# offset (shift back). So the lag we return is:
	lag_s = -best_lag_samples / fs
	return lag_s


def _build_common_grid(
	ref_t: np.ndarray,
	tgt_t: np.ndarray,
	fs: float,
) -> np.ndarray:
	"""Build a common, uniformly sampled time grid over the overlap."""
	t_start = max(ref_t[0], tgt_t[0])
	t_end = min(ref_t[-1], tgt_t[-1])
	if t_end <= t_start:
		raise ValueError("No overlapping time range between streams")

	dt = 1.0 / fs
	n = int(np.floor((t_end - t_start) / dt)) + 1
	return t_start + np.arange(n, dtype=np.float64) * dt


def _interp_to_grid(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
	"""Linearly interpolate `y(t)` onto `grid` (LIDA-style alignment helper)."""
	return np.interp(grid, t, y)


def synchronize_imu_streams(
	ref: pd.DataFrame,
	tgt: pd.DataFrame,
	time_col: str = "timestamp",
	acc_cols: Tuple[str, str, str] = ("ax", "ay", "az"),
	resample_rate: Optional[float] = None,
	max_lag_s: Optional[float] = None,
	method: str = "LIDA",
	ref_csv_path: Optional[str] = None,
	tgt_csv_path: Optional[str] = None,
) -> Dict[str, float]:
	"""Synchronize two IMU streams using SDA lag + LIDA-style linear drift."""
	# 1) Relative timestamps with originals preserved
	ref_rel, ref_orig_ts = _to_relative_time(ref, time_col=time_col)
	tgt_rel, tgt_orig_ts = _to_relative_time(tgt, time_col=time_col)

	ref_t = ref_rel[time_col].to_numpy(dtype=np.float64)
	tgt_t = tgt_rel[time_col].to_numpy(dtype=np.float64)

	# Estimate nominal sampling rate from reference
	fs_nominal = _estimate_sampling_rate(ref_t)
	if resample_rate is None:
		fs = fs_nominal
	else:
		fs = float(resample_rate)

	# 2) Cross-correlation on acceleration magnitude (SDA-style lag)
	ref_mag = _acc_magnitude(ref_rel[list(acc_cols)])
	tgt_mag = _acc_magnitude(tgt_rel[list(acc_cols)])

	grid = _build_common_grid(ref_t, tgt_t, fs=fs)
	ref_mag_g = _interp_to_grid(ref_t, ref_mag, grid)
	tgt_mag_g = _interp_to_grid(tgt_t, tgt_mag, grid)

	lag_s = _cross_correlation_lag(
		ref_mag_g,
		tgt_mag_g,
		fs=fs,
		max_lag_s=max_lag_s,
	)

	# 3) Affine transformation for drift (LIDA-style)
	# Here we use a simple endpoint-based affine scaling between the two clocks.
	ref_duration = ref_t[-1] - ref_t[0]
	tgt_duration = tgt_t[-1] - tgt_t[0]
	if tgt_duration <= 0:
		raise ValueError("Target duration must be positive")

	scale = ref_duration / tgt_duration  # drift scaling between clocks

	# Affine mapping: t_tgt_aligned = scale * t_tgt_rel + lag_s
	# (in the reference relative time base)
	tgt_t_aligned_rel = scale * tgt_t + lag_s

	# 4) Translate target to reference time base and optionally save to CSV
	ref_out = ref_rel.copy()
	ref_out[f"{time_col}_relative"] = ref_t

	tgt_out = tgt_rel.copy()
	tgt_out[f"{time_col}_relative"] = tgt_t
	tgt_out[f"{time_col}_aligned"] = tgt_t_aligned_rel

	# 5) Optional resampling to a common rate (reference time base)
	if resample_rate is not None:
		dt = 1.0 / resample_rate
		t_start = max(ref_t[0], tgt_t_aligned_rel[0])
		t_end = min(ref_t[-1], tgt_t_aligned_rel[-1])
		if t_end <= t_start:
			raise ValueError("No overlapping region after alignment for resampling")

		n = int(np.floor((t_end - t_start) / dt)) + 1
		grid_res = t_start + np.arange(n, dtype=np.float64) * dt

		def _resample_df(df: pd.DataFrame, t_src: np.ndarray, cols: Tuple[str, ...]) -> pd.DataFrame:
			out = {time_col: grid_res}
			for c in cols:
				if c in df.columns and np.issubdtype(df[c].dtype, np.number):
					out[c] = np.interp(grid_res, t_src, df[c].to_numpy(dtype=np.float64))
			return pd.DataFrame(out)

		num_cols = tuple(c for c in ref_out.columns if c not in (time_col, f"{time_col}_orig"))

		ref_out = _resample_df(ref_out, ref_t, num_cols)
		tgt_out = _resample_df(tgt_out, tgt_t_aligned_rel, num_cols)

	# 6) Print and return lag/drift components
	print(f"SDA lag estimate (seconds, added to target): {lag_s:.6f}")
	print(f"LIDA affine drift scale (target->reference time): {scale:.9f}")
	print(f"Nominal reference sampling rate (Hz): {fs_nominal:.3f}")
	if resample_rate is not None:
		print(f"Resampled common rate (Hz): {resample_rate:.3f}")

	# Save aligned streams to CSV if paths provided
	if ref_csv_path is not None:
		ref_out.to_csv(ref_csv_path, index=False)
	if tgt_csv_path is not None:
		tgt_out.to_csv(tgt_csv_path, index=False)

	return {
		"lag_sda_seconds": float(lag_s),
		"scale_lida": float(scale),
		"fs_reference_nominal_hz": float(fs_nominal),
		"fs_resampled_hz": float(resample_rate) if resample_rate is not None else float(fs),
	}


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Synchronize two IMU CSV streams using SDA/LIDA-style alignment.",
	)
	parser.add_argument(
		"ref_csv",
		help="Reference IMU CSV (typically under data/processed/...).",
	)
	parser.add_argument(
		"tgt_csv",
		help="Target IMU CSV to be aligned to the reference.",
	)
	parser.add_argument(
		"--time-col",
		default="timestamp",
		help="Name of timestamp column (default: timestamp).",
	)
	parser.add_argument(
		"--acc-cols",
		nargs=3,
		default=("ax", "ay", "az"),
		metavar=("AX", "AY", "AZ"),
		help="Acceleration column names (default: ax ay az).",
	)
	parser.add_argument(
		"--resample-rate",
		type=float,
		default=None,
		help="Optional common resampling rate in Hz.",
	)
	parser.add_argument(
		"--max-lag-s",
		type=float,
		default=None,
		help="Optional maximum absolute lag (seconds) to search.",
	)
	parser.add_argument(
		"--suffix",
		default="_sync",
		help="Suffix appended to output filenames (default: _sync).",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	ref_path = Path(args.ref_csv)
	tgt_path = Path(args.tgt_csv)

	ref_df = pd.read_csv(ref_path)
	tgt_df = pd.read_csv(tgt_path)

	suffix = args.suffix
	ref_out_path = ref_path.with_name(ref_path.stem + f"{suffix}_ref.csv")
	tgt_out_path = tgt_path.with_name(tgt_path.stem + f"{suffix}_tgt.csv")

	stats = synchronize_imu_streams(
		ref_df,
		tgt_df,
		time_col=args.time_col,
		acc_cols=tuple(args.acc_cols),
		resample_rate=args.resample_rate,
		max_lag_s=args.max_lag_s,
		ref_csv_path=str(ref_out_path),
		tgt_csv_path=str(tgt_out_path),
	)

	print("Synchronization stats:", stats)
	print(f"Wrote reference stream to: {ref_out_path}")
	print(f"Wrote target stream to:   {tgt_out_path}")


if __name__ == "__main__":
	main()


