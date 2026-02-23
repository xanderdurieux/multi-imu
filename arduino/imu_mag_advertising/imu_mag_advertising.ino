#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

// Service and characteristic UUIDs
// Reuse same service UUID family as your other sketch
const char* deviceServiceUuid = "0x696d7530";
// Packed IMU (acc + gyro) characteristic
const char* ImuPackedCharUuid = "0x696d7534";
// Optional low-rate magnetometer characteristic
const char* MagCharUuid       = "0x696d7533";

// Packed IMU sample:
//  - ax, ay, az: accelerometer in g, scaled to int16
//  - gx, gy, gz: gyroscope in dps, scaled to int16
//  - timestamp:  millis() when the sample was created
// Total: 6 * int16 (12 bytes) + 1 * uint32 (4 bytes) = 16 bytes
struct ImuSamplePacked {
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t gx;
  int16_t gy;
  int16_t gz;
  uint32_t timestamp;
};

// Simple float-based structure for low-rate magnetometer output
struct MagSample {
  float x;
  float y;
  float z;
  uint32_t timestamp;
};

BLEService IMUService(deviceServiceUuid);
BLECharacteristic ImuPackedChar(ImuPackedCharUuid, BLERead | BLENotify, sizeof(ImuSamplePacked), true);
BLECharacteristic MagChar(MagCharUuid, BLERead | BLENotify, sizeof(MagSample), true);

// Working buffers
ImuSamplePacked imuPkt;
MagSample magPkt;

float ax, ay, az;
float gx, gy, gz;
float mx, my, mz;

// Timing and diagnostics
uint32_t currentTimestamp = 0;
uint32_t lastDebugMillis  = 0;
uint32_t lastMagMillis    = 0;

uint32_t loopCount   = 0;
uint32_t imuSamples  = 0;
uint32_t magSamples  = 0;

void setup() {
  // Fast serial for occasional debug; non-blocking wait
  Serial.begin(115200);
  unsigned long serialStart = millis();
  while (!Serial && (millis() - serialStart < 2000)) {
    // Give USB a moment to connect, but don't block forever
  }

  if (!BLE.begin()) {
    while (1) { }
  }
  Serial.println("BLE initialized!");

  delay(500);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1) { }
  }
  Serial.println("IMU initialized!");

  // Set connection interval to 7.5ms minimum for high data throughput.
  // Parameters are in units of 1.25ms: 0x0006 = 7.5ms, 0x0C80 = 4000ms
  BLE.setConnectionInterval(0x0006, 0x0C80);

  BLE.setLocalName("Nano 33 BLE Packed");
  BLE.setAdvertisedService(IMUService);
  IMUService.addCharacteristic(ImuPackedChar);
  IMUService.addCharacteristic(MagChar);
  BLE.addService(IMUService);
  BLE.advertise();

  Serial.print("ImuSamplePacked size: ");
  Serial.print(sizeof(ImuSamplePacked));
  Serial.println(" bytes");

  Serial.print("MagSample size: ");
  Serial.print(sizeof(MagSample));
  Serial.println(" bytes");

  Serial.println("IMU Packed Peripheral (Sending Data)");
}

void loop() {
  loopCount++;

  // Service BLE stack
  BLE.poll();

  currentTimestamp = millis();

  // High-rate path: accelerometer + gyroscope combined into one packet
  // Both sensors are configured in the library to 400 Hz ODR.
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    // Scale floats to int16 based on library's default ranges:
    //  - Accelerometer: ±4g  -> 8192 LSB/g  (INT16_to_G in BMI270.cpp)
    //  - Gyroscope:    ±2000 dps -> 16.384 LSB/dps (INT16_to_DPS)
    const float accScale = 8192.0f;
    const float gyrScale = 16.384f;

    imuPkt.ax = (int16_t)(ax * accScale);
    imuPkt.ay = (int16_t)(ay * accScale);
    imuPkt.az = (int16_t)(az * accScale);
    imuPkt.gx = (int16_t)(gx * gyrScale);
    imuPkt.gy = (int16_t)(gy * gyrScale);
    imuPkt.gz = (int16_t)(gz * gyrScale);
    imuPkt.timestamp = currentTimestamp;

    ImuPackedChar.writeValue((uint8_t*)&imuPkt, sizeof(ImuSamplePacked));
    imuSamples++;
  }

  // Low-rate magnetometer: target ~10 Hz to avoid wasting BLE bandwidth
  if (currentTimestamp - lastMagMillis >= 100) { // 100 ms ~ 10 Hz
    if (IMU.magneticFieldAvailable()) {
      IMU.readMagneticField(mx, my, mz);
      magPkt.x = mx;
      magPkt.y = my;
      magPkt.z = mz;
      magPkt.timestamp = currentTimestamp;

      MagChar.writeValue((uint8_t*)&magPkt, sizeof(MagSample));
      magSamples++;
    }
    lastMagMillis = currentTimestamp;
  }

  // // Lightweight debug once per second: report effective rates
  // if (currentTimestamp - lastDebugMillis >= 1000) {
  //   uint32_t dt = currentTimestamp - lastDebugMillis;
  //   if (dt == 0) {
  //     dt = 1;
  //   }

  //   float loopHz = (loopCount  * 1000.0f) / dt;
  //   float imuHz  = (imuSamples * 1000.0f) / dt;
  //   float magHz  = (magSamples * 1000.0f) / dt;

  //   Serial.print("loopHz:");
  //   Serial.print(loopHz, 1);
  //   Serial.print(" imuHz:");
  //   Serial.print(imuHz, 1);
  //   Serial.print(" magHz:");
  //   Serial.print(magHz, 1);
  //   Serial.print(" lastImu[");
  //   Serial.print((float)imuPkt.ax / 8192.0f, 3);
  //   Serial.print(", ");
  //   Serial.print((float)imuPkt.ay / 8192.0f, 3);
  //   Serial.print(", ");
  //   Serial.print((float)imuPkt.az / 8192.0f, 3);
  //   Serial.println("]");

  //   loopCount  = 0;
  //   imuSamples = 0;
  //   magSamples = 0;
  //   lastDebugMillis = currentTimestamp;
  // }
}

