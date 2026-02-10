# master-thesis/src/arduino

This folder contains the Arduino sketches that are uploaded to the Arduino board(s) used in this thesis.

## Hardware

The board used is the **Arduino Nano 33 BLE**, which provides:

- **Bluetooth Low Energy (BLE)** connectivity
- An onboard **9‑axis IMU** (BMI270 + BMM150) for accelerometer, gyroscope and magnetometer data

## Sketches

### `ble_imu_advertisting/ble_imu_advertising.ino`

This sketch turns the Nano 33 BLE into a **BLE IMU peripheral**. It:

- Initializes the onboard IMU via `Arduino_BMI270_BMM150`
- Initializes BLE via `ArduinoBLE`
- Exposes a single IMU service and three characteristics (one per sensor type)
- Continuously reads:
  - Accelerometer
  - Gyroscope
  - Magnetometer
- Sends readings as BLE notifications to a connected central device

Each packet is encoded in a `SensorData` struct:

- `sensorType` (`uint8_t`): identifies whether the packet is accel, gyro or mag
- `x`, `y`, `z` (`float`): sensor values
- `timestamp` (`uint32_t`): `millis()` at the time of sampling

This structure is 17 bytes and is designed to fit in a single BLE notification (20‑byte MTU including ATT header).

## BLE Service & Characteristics

The sketch uses the following UUIDs:

- **Service UUID**: `0x696d7530`
- **Accelerometer characteristic UUID**: `0x696d7531`
- **Gyroscope characteristic UUID**: `0x696d7532`
- **Magnetometer characteristic UUID**: `0x696d7533`

All three characteristics:

- Support **`BLERead`** and **`BLENotify`**
- Have a fixed size of `sizeof(SensorData)` (17 bytes)

Any BLE central (e.g. a Python script, desktop tool, or mobile app) that knows these UUIDs can subscribe to notifications and reconstruct the IMU data stream.

## Software Requirements

On the host machine running the Arduino IDE:

- **Arduino IDE** (1.8.x or 2.x)
- **Boards package**: *Arduino Mbed OS Nano Boards* (for Arduino Nano 33 BLE)
- **Libraries** (via Library Manager):
  - `ArduinoBLE`
  - `Arduino_BMI270_BMM150`

## Uploading the Sketch

1. Open `ble_imu_advertisting/ble_imu_advertising.ino` in the Arduino IDE.
2. In **Tools → Board**, select **Arduino Nano 33 BLE**.
3. In **Tools → Port**, select the serial port corresponding to the Nano 33 BLE.
4. Ensure the required libraries are installed (see above).
5. Click **Upload**.

After a successful upload, the board will:

- Start advertising the IMU service with local name **"Nano 33 BLE"**
- Accept connections from a BLE central
- Stream accelerometer, gyroscope and magnetometer data via notifications

## Debugging

Serial output in the sketch is currently commented out. To debug:

1. Uncomment the `Serial.begin` and related `Serial.print` lines in `ble_imu_advertising.ino`.
2. Open the **Serial Monitor** in the Arduino IDE.
3. Re‑upload the sketch and observe the initialization messages and optional IMU value prints.