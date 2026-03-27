# Dynamic orientation comparison — thesis-ready interpretation notes

This note summarizes how to report dynamic orientation filter validation for cycling.

## Methods (suggested wording)

We compared all orientation filters already produced by the pipeline (`complementary`, `Madgwick` variants, `EKF`, and any additional generated variant files) using section-level aligned timestamps. Comparison focused on dynamic cycling relevance rather than static-only calibration checks. Metrics included: (1) inter-filter agreement over time via pairwise quaternion angular distance, (2) smoothness-vs-responsiveness tradeoff via orientation angular-acceleration noise against high-percentile roll/pitch rate response, (3) long-section drift from early versus late static windows, (4) consistency of derived roll/pitch trajectories across filters, and (5) downstream usefulness proxy via event-related feature separability between dynamic and static windows. Magnetometer reliability was flagged from magnetic-field norm distortion and abrupt heading discontinuities.

## Results (how to read the outputs)

- `summary_pairwise_agreement.csv`: lower mean/p95 angle distance indicates stronger consensus across filters.
- `summary_per_filter.csv`: higher `smoothness_responsiveness_ratio` and `event_separability_index`, with lower `drift_static_endpoint_deg`, indicate better practical behavior for dynamic analysis.
- `overlay_static.png` and `overlay_dynamic.png`: visual sanity checks for stability at rest and responsiveness during motion.
- `mag_unreliable_fraction`: high values suggest limiting reliance on magnetometer-corrected yaw for that section/sensor.

## Recommendation logic

The script ranks filters by weighted score (smoothness/responsiveness, drift, event separability, heading reliability). Use the top-ranked filter as the default for dynamic cycling analysis. For sections with frequent magnetometer reliability flags, prefer gyro+accelerometer filters for relative orientation features and treat absolute heading cautiously.
