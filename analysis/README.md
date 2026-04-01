### TODO

- Add comparison metrics across recordings in exports stage, maybe move plotting to visualization (sync params, calibration params)
- Orientation usage: flatten the files. Rename config param to "auto"

#### Necessary pipeline features
- Magnetometer-aware heading and better yaw calculation
- Add EKF Filter to orientation stage
- Add reporting module
- Config reformatting -> all stages are defined in config, not hardcoded in workflow

#### Verify existing correctness
- Review orientation correctness
- Review static detection and how it is used
- Review events and usage
- Review LIDA
- Review thresholds for everything

#### Extra features
- Interpolate/resample Arduino data to have full csvs and rerun pipeline (add optional resample step?)
- GPS integration into yaw calculation