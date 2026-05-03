# `arduino/` - Arduino Nano 33 BLE Firmware

This folder contains the Arduino sketches used to stream helmet IMU data over Bluetooth Low Energy.

## Hardware

- Board: `Arduino Nano 33 BLE`
- IMU library: `Arduino_BMI270_BMM150`
- BLE library: `ArduinoBLE`

The analysis pipeline currently parses the baseline float-per-sensor packet format from `ble_imu_advertising/`.

## BLE UUIDs

All sketches use the same service family:

```text
Service:       0x696d7530
Accelerometer: 0x696d7531
Gyroscope:     0x696d7532
Magnetometer:  0x696d7533
Packed IMU:    0x696d7534
```

## Sketches

### `ble_imu_advertising/`

Baseline firmware for the main pipeline.

- Sends accelerometer, gyroscope, and magnetometer samples as separate BLE characteristics.
- Uses a `SensorData` packet with sensor type, `x`, `y`, `z`, and `millis()` timestamp.
- Matches the parser in `analysis/parser/arduino.py`.

### `imu_mag_advertising/`

Experimental packed firmware.

- Packs accelerometer and gyroscope into one high-rate notification.
- Sends magnetometer samples separately at lower rate.
- Needs matching host-side parsing before it can replace the baseline format.

### `imu_mag_batched/`

Experimental batched firmware.

- Batches multiple packed accelerometer/gyroscope samples in one BLE notification.
- Keeps magnetometer samples on the low-rate characteristic.
- Requires a larger negotiated MTU for full batches.

## Upload

1. Open the selected `.ino` file in the Arduino IDE.
2. Select `Arduino Nano 33 BLE`.
3. Select the serial port.
4. Install `ArduinoBLE` and `Arduino_BMI270_BMM150`.
5. Upload the sketch.

## Notes

All sketches timestamp samples with device `millis()`. The Python pipeline aligns that device clock to the Sporsa clock during the `sync` stage.
