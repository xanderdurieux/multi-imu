### TODO

- Fix faulty calibration sequence extraction in r5

#### Necessary pipeline features
- Magnetometer-aware heading and better yaw calculation (Madgwick MARG + EKF MARG exist; tune and validate)
- Optional thesis report bundling stage (not wired in the workflow yet)

#### Verify existing correctness
- Review orientation correctness
- Review static detection and how it is used
- Review events and usage
- Review LIDA
- Review thresholds for everything

#### Extra features
- Interpolate/resample Arduino data to have full csvs and rerun pipeline (add optional resample step?)
- GPS integration into yaw calculation