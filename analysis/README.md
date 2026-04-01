### TODO

- Add comparison metrics across recordings in exports stage, maybe move plotting of exports to visualization (sync params, calibration params)
- Fix faulty calibration sequence extraction in r5

#### Necessary pipeline features
- Magnetometer-aware heading and better yaw calculation
- Add EKF Filter to orientation stage
- Add reporting module

#### Verify existing correctness
- Review orientation correctness
- Review static detection and how it is used
- Review events and usage
- Review LIDA
- Review thresholds for everything

#### Extra features
- Interpolate/resample Arduino data to have full csvs and rerun pipeline (add optional resample step?)
- GPS integration into yaw calculation