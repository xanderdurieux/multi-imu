"""IMU sensor calibration from recorded calibration sequences.

Each sensor is calibrated independently to the world frame (ENU: East-North-Up)
using the static windows flanking the acceleration-peak bursts that are recorded
at the start and end of each session.

Calibration steps
-----------------
1. :mod:`calibration.static_windows` — detect calibration sequences and slice
   the flanking static sub-DataFrames for each sensor.
2. :mod:`calibration.per_sensor` — estimate gyroscope zero-rate bias, the
   gravity vector in sensor frame, and magnetometer hard-iron offset from the
   aggregated static window samples.
3. :mod:`calibration.orientation` — compute the sensor-to-world rotation matrix
   via the TRIAD method (gravity + magnetometer) when mag data is available,
   falling back to gravity-only (pitch + roll, yaw = 0) otherwise.
4. :mod:`calibration.session` — orchestrate the full pipeline for a recording
   and write ``calibrated/calibration.json`` plus bias-corrected,
   world-rotated sensor CSVs.

Typical usage::

    from calibration.session import calibrate_recording
    calibrate_recording("2026-02-26_5")

CLI::

    python -m calibration.session 2026-02-26_5
    python -m calibration.session 2026-02-26_5 --stage parsed --no-apply
"""

from .static_windows import StaticWindows, extract_static_windows
from .per_sensor import SensorCalibration, calibrate_sensor
from .orientation import OrientationCalibration, compute_orientation_from_vectors
from .session import calibrate_recording

__all__ = [
    "StaticWindows",
    "extract_static_windows",
    "SensorCalibration",
    "calibrate_sensor",
    "OrientationCalibration",
    "compute_orientation_from_vectors",
    "calibrate_recording",
]
