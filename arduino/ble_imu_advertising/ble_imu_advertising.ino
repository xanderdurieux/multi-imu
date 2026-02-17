#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

// Set the same service UUID as in the Peripheral Device
const char* deviceServiceUuid = "0x696d7530";
// Characteristic UUIDs for each sensor type
const char* AccCharUuid = "0x696d7531";
const char* GyroCharUuid = "0x696d7532";
const char* MagCharUuid = "0x696d7533";

// Sensor type flags
#define SENSOR_TYPE_ACCEL  0x01
#define SENSOR_TYPE_GYRO   0x02
#define SENSOR_TYPE_MAG    0x04

// Structure for each sensor type: 3 floats (x,y,z) + timestamp/sequence ID + sensor type
// Size: 3 floats * 4 bytes + 1 uint32_t * 4 bytes + 1 uint8_t = 17 bytes
// NOTE: Fits within default 20-byte notification limit (17 + 3 ATT header = 20 bytes)
// Using float instead of double: ~7 decimal digits precision (sufficient for IMU data)
struct SensorData {
  uint8_t sensorType; // Sensor type flag (SENSOR_TYPE_ACCEL, SENSOR_TYPE_GYRO, SENSOR_TYPE_MAG)
  float x;
  float y;
  float z;
  uint32_t timestamp;  // Sequence ID or timestamp to match readings across characteristics
};

SensorData accData = {SENSOR_TYPE_ACCEL, 0, 0, 0, 0};
SensorData gyroData = {SENSOR_TYPE_GYRO, 0, 0, 0, 0};
SensorData magData = {SENSOR_TYPE_MAG, 0, 0, 0, 0};

float x, y, z, gx, gy, gz, mx, my, mz;
// Use the device clock (millis) to timestamp each sensor batch
// Ensures absolute timing instead of an incremental sequence ID
uint32_t currentTimestamp = 0;

BLEService IMUService(deviceServiceUuid);
// Three characteristics, one for each sensor type (17 bytes each - fits in 20-byte notifications!)
BLECharacteristic AccChar(AccCharUuid, BLERead | BLENotify, sizeof(SensorData), true);
BLECharacteristic GyroChar(GyroCharUuid, BLERead | BLENotify, sizeof(SensorData), true);
BLECharacteristic MagChar(MagCharUuid, BLERead | BLENotify, sizeof(SensorData), true);

void setup() {
  // Serial.begin(9600);
  // while (!Serial);

  if (!BLE.begin()) {
    // Serial.println("Starting Bluetooth failed!");
    while (1);
  }
  // Serial.println("BLE initialized!");
  
  delay(500);

  if (!IMU.begin()) {
    // Serial.println("Failed to initialize IMU!");
    while (1);
  }
  // Serial.println("IMU initialized!");

  // Set connection interval to 7.5ms minimum (for high data rate)
  // Parameters are in units of 1.25ms: 0x0006 = 7.5ms, 0x0C80 = 4000ms
  BLE.setConnectionInterval(0x0006, 0x0C80);
   
  BLE.setLocalName("Nano 33 BLE");
  BLE.setAdvertisedService(IMUService);
  IMUService.addCharacteristic(AccChar);
  IMUService.addCharacteristic(GyroChar);
  IMUService.addCharacteristic(MagChar);
  BLE.addService(IMUService);
  BLE.advertise();
  
  // Serial.print("Sensor data structure size: ");
  // Serial.print(sizeof(SensorData));
  // Serial.println(" bytes per characteristic");
  // Serial.println("Note: Fits in default 20-byte notification limit (17 + 3 overhead = 20 bytes)");

  // Serial.println("IMU Peripheral (Sending Data)");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    if (central.connected()) {
      // Serial.println("Connected to central device");
      // Serial.print("Device MAC address: ");
      // Serial.println(central.address());
    } else {
      // Serial.println("Disconnected from central device");
    }
  }

  // Capture current timestamp once per loop iteration (shared by all sensor packets)
  currentTimestamp = millis();
  
  // Read and send accelerometer data
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);
    accData.x = x;
    accData.y = y;
    accData.z = z;
    accData.timestamp = currentTimestamp;
    AccChar.writeValue((uint8_t*)&accData, sizeof(SensorData));
  }

  // Read and send gyroscope data
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
    gyroData.x = gx;
    gyroData.y = gy;
    gyroData.z = gz;
    gyroData.timestamp = currentTimestamp;
    GyroChar.writeValue((uint8_t*)&gyroData, sizeof(SensorData));
  }

  // Read and send magnetometer data
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
    magData.x = mx;
    magData.y = my;
    magData.z = mz;
    magData.timestamp = currentTimestamp;
    MagChar.writeValue((uint8_t*)&magData, sizeof(SensorData));
  }
  
  // Optional: Print values for debugging
  // Serial.print("accX:");
  // Serial.print(accData.x);
  // Serial.print(" accY:");
  // Serial.print(accData.y);
  // Serial.print(" accZ:");
  // Serial.print(accData.z);
  // Serial.print(" gyroX:");
  // Serial.print(gyroData.x);
  // Serial.print(" gyroY:");
  // Serial.print(gyroData.y);
  // Serial.print(" gyroZ:");
  // Serial.print(gyroData.z);
  // Serial.print(" magX:");
  // Serial.print(magData.x);
  // Serial.print(" magY:");
  // Serial.print(magData.y);
  // Serial.print(" magZ:");
  // Serial.println(magData.z);
}