# `arduino/` — Arduino Nano 33 BLE Sensor Firmware

This folder contains the Arduino sketches used to stream IMU data from an Arduino Nano 33 BLE over Bluetooth Low Energy.

## Hardware

The target board is the `Arduino Nano 33 BLE`, using its onboard BMI270 + BMM150 IMU stack through `Arduino_BMI270_BMM150`.

## Shared BLE design

All sketches advertise the same BLE service family:

- Service UUID: `0x696d7530`
- Magnetometer UUID: `0x696d7533`

They differ in how accelerometer and gyroscope samples are packed and transmitted.

## Sketches

### `ble_imu_advertising/ble_imu_advertising.ino`

Baseline sketch with one BLE characteristic per sensor:

- accelerometer characteristic: `0x696d7531`
- gyroscope characteristic: `0x696d7532`
- magnetometer characteristic: `0x696d7533`

Each notification sends a `SensorData` struct with:

- sensor type flag
- `x`, `y`, `z` as floats
- shared `millis()` timestamp

This is the layout expected by `analysis/parser/arduino.py`.

### `imu_mag_advertising/imu_mag_advertising.ino`

Bandwidth-reduced variant:

- packs accelerometer and gyroscope into one `ImuSamplePacked` notification on UUID `0x696d7534`
- keeps magnetometer as a separate low-rate float packet on UUID `0x696d7533`
- targets high-rate accel/gyro streaming with low-rate magnetometer updates

### `imu_mag_batched/imu_mag_batched.ino`

Higher-throughput batched variant:

- uses the same packed accel/gyro format as `imu_mag_advertising`
- batches multiple packed IMU samples into one BLE notification
- keeps the magnetometer on the separate low-rate characteristic
- requires a larger negotiated MTU for the full batch payload

## Tooling requirements

- Arduino IDE 1.8.x or 2.x
- Board package: `Arduino Mbed OS Nano Boards`
- Libraries:
  - `ArduinoBLE`
  - `Arduino_BMI270_BMM150`

## Uploading

1. Open the desired `.ino` file in the Arduino IDE.
2. Select `Arduino Nano 33 BLE` in the board menu.
3. Select the correct serial port.
4. Ensure the required libraries are installed.
5. Upload the sketch.

## Notes

- All sketches timestamp samples with device `millis()`.
- The thesis analysis pipeline currently parses the baseline float-per-sensor `SensorData` format directly.
- The packed and batched variants are experimental throughput-oriented firmware variants and may require matching host-side parsers before they can be used in the main pipeline.
