# Dual-IMU Cycling Feature Catalog

This catalog documents compact feature families grouped by physical interpretation and event hypothesis.

## Design principles
- Prefer interpretable, mechanism-linked features over broad feature explosion.
- Keep legacy generic statistics for backwards compatibility, but prioritize grouped features below for event reasoning and ablations.
- Feature groups are implemented in `analysis/features/families.py` and can be cleanly ablated by family.

## Group 1 — Bumps / road disturbances
**Physical intuition:** Road roughness appears first at bike mount as vertical shocks; rider signal should be attenuated and lagged.

Features:
- `bump_vertical_peak_ms2`: peak absolute bike vertical acceleration.
- `bump_shock_attenuation_ratio`: bike/rider vertical shock ratio.
- `bump_response_lag_s`: lag between bike and rider vertical responses.

## Group 2 — Braking / deceleration
**Physical intuition:** Braking yields negative longitudinal acceleration and pitch-forward coupling.

Features:
- `brake_longitudinal_decel_peak_ms2`: strongest (most negative) bike longitudinal acceleration.
- `brake_pitch_change_deg`: bike pitch excursion across window.
- `brake_pitch_coupling_corr`: bike-rider pitch synchronization.

## Group 3 — Cornering / swerving
**Physical intuition:** Lateral acceleration and roll dynamics increase during turning and swerving.

Features:
- `corner_lateral_energy_ms2_sq`: bike lateral acceleration energy.
- `corner_roll_rate_rms_deg_s`: bike roll-rate RMS.
- `corner_roll_coupling_corr`: bike-rider roll correlation.

## Group 4 — Sprinting / rider exertion
**Physical intuition:** Sprinting introduces periodic cadence-like patterns and high angular motion energy.

Features:
- `sprint_cadence_band_fraction`: acc-norm energy fraction in 1.2–3.5 Hz.
- `sprint_dom_freq_hz`: dominant frequency of bike acceleration norm.
- `sprint_gyro_energy_sum`: combined bike+rider gyro energy.

## Group 5 — Rider-bicycle disagreement / destabilization
**Physical intuition:** Instability can manifest as mismatch in timing, directional energy, and coherence between sensors.

Features:
- `disagree_vec_diff_mean_ms2`: vector acceleration disagreement magnitude.
- `disagree_vertical_coherence`: vertical coherence in 0.5–10 Hz.
- `disagree_energy_axis_ratio_var`: variance of axis-wise energy ratios.

## Legacy features retained
Legacy per-sensor and cross-sensor summary statistics remain in extraction output for continuity with previous analyses and exported tables.

## Degenerate-window sanity checks
For each sensor window:
- `*_window_sanity = ok` for healthy windows.
- Otherwise semicolon-delimited flags among: `short_window`, `acc_low_variance`, `gyro_low_variance`, `all_nan`.

## Scenario summaries (if labels are present)
When `scenario_label` is available, extraction writes:
- `features/scenario_feature_summary.csv` (mean/std/count by scenario)
- `features/plots/scenario_feature_summary.png`

## Thesis-worthy candidate figures
1. **Shock attenuation vs disturbance label** (`bump_shock_attenuation_ratio`) with confidence intervals.
2. **Braking event panel** combining `brake_longitudinal_decel_peak_ms2` and `brake_pitch_change_deg`.
3. **Cornering coupling map** (`corner_roll_coupling_corr` vs `corner_lateral_energy_ms2_sq`).
4. **Sprinting periodicity figure** (`sprint_cadence_band_fraction`, `sprint_dom_freq_hz`) across effort levels.
5. **Destabilization disagreement chart** (`disagree_vec_diff_mean_ms2` + `disagree_vertical_coherence`).
