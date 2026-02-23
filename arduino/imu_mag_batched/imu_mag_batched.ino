#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

// Service and characteristic UUIDs
const char* deviceServiceUuid   = "0x696d7530";
const char* ImuPackedCharUuid   = "0x696d7534";  // batched IMU (acc + gyro)
const char* MagCharUuid         = "0x696d7533";  // low-rate magnetometer

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

// Number of IMU samples per BLE notification.
// 4 * 16 = 64-byte payload (requires MTU >= 67 bytes including ATT header).
// nRF Connect can typically negotiate a large MTU; if not, reduce this to 2.
const uint8_t IMU_BATCH_SIZE = 4;

BLEService IMUService(deviceServiceUuid);
BLECharacteristic ImuPackedChar(
  ImuPackedCharUuid,
  BLERead | BLENotify,
  sizeof(ImuSamplePacked) * IMU_BATCH_SIZE,
  true
);
BLECharacteristic MagChar(MagCharUuid, BLERead | BLENotify, sizeof(MagSample), true);

// Working buffers
ImuSamplePacked imuBatch[IMU_BATCH_SIZE];
uint8_t imuBatchIndex = 0;

MagSample magPkt;

float ax, ay, az;
float gx, gy, gz;
float mx, my, mz;

// Timing and diagnostics
uint32_t currentTimestamp = 0;
uint32_t lastDebugMillis  = 0;
uint32_t lastMagMillis    = 0;

uint32_t loopCount    = 0;
uint32_t imuSamples   = 0;  // individual IMU samples
uint32_t imuBatches   = 0;  // BLE notifications for IMU
uint32_t magSamples   = 0;

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

  // Request high-throughput connection parameters.
  BLE.setConnectionInterval(0x0006, 0x0C80);

  BLE.setLocalName("Nano 33 BLE Batched");
  BLE.setAdvertisedService(IMUService);
  IMUService.addCharacteristic(ImuPackedChar);
  IMUService.addCharacteristic(MagChar);
  BLE.addService(IMUService);
  BLE.advertise();

  Serial.print("ImuSamplePacked size: ");
  Serial.print(sizeof(ImuSamplePacked));
  Serial.println(" bytes");

  Serial.print("IMU batch payload size: ");
  Serial.print(sizeof(ImuSamplePacked) * IMU_BATCH_SIZE);
  Serial.println(" bytes");

  Serial.print("MagSample size: ");
  Serial.print(sizeof(MagSample));
  Serial.println(" bytes");

  Serial.println("IMU Batched Peripheral (Sending Data)");
}

static void flushImuBatch() {
  if (imuBatchIndex == 0) {
    return;
  }

  // Send only the filled part of the batch.
  const uint16_t payloadBytes =
    imuBatchIndex * sizeof(ImuSamplePacked);

  ImuPackedChar.writeValue(
    reinterpret_cast<uint8_t*>(imuBatch),
    payloadBytes
  );

  imuBatches++;
  imuBatchIndex = 0;
}

void loop() {
  loopCount++;

  // Service BLE stack
  BLE.poll();

  currentTimestamp = millis();

  // High-rate path: accelerometer + gyroscope combined into batched packets.
  // IMU is configured in the library to 400 Hz ODR.
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    const float accScale = 8192.0f;   // LSB per g
    const float gyrScale = 16.384f;   // LSB per deg/s

    ImuSamplePacked& s = imuBatch[imuBatchIndex];
    s.ax = static_cast<int16_t>(ax * accScale);
    s.ay = static_cast<int16_t>(ay * accScale);
    s.az = static_cast<int16_t>(az * accScale);
    s.gx = static_cast<int16_t>(gx * gyrScale);
    s.gy = static_cast<int16_t>(gy * gyrScale);
    s.gz = static_cast<int16_t>(gz * gyrScale);
    s.timestamp = currentTimestamp;

    imuBatchIndex++;
    imuSamples++;

    if (imuBatchIndex >= IMU_BATCH_SIZE) {
      flushImuBatch();
    }
  }

  // Low-rate magnetometer: target ~10 Hz
  if (currentTimestamp - lastMagMillis >= 100) {  // 100 ms ~ 10 Hz
    if (IMU.magneticFieldAvailable()) {
      IMU.readMagneticField(mx, my, mz);
      magPkt.x = mx;
      magPkt.y = my;
      magPkt.z = mz;
      magPkt.timestamp = currentTimestamp;

      MagChar.writeValue(
        reinterpret_cast<uint8_t*>(&magPkt),
        sizeof(MagSample)
      );
      magSamples++;
    }
    lastMagMillis = currentTimestamp;
  }

  // Lightweight debug once per second: report effective rates.
  if (currentTimestamp - lastDebugMillis >= 1000) {
    uint32_t dt = currentTimestamp - lastDebugMillis;
    if (dt == 0) {
      dt = 1;
    }

    float loopHz   = (loopCount  * 1000.0f) / dt;
    float imuHz    = (imuSamples * 1000.0f) / dt;
    float batchHz  = (imuBatches * 1000.0f) / dt;
    float magHz    = (magSamples * 1000.0f) / dt;

    Serial.print("loopHz:");
    Serial.print(loopHz, 1);
    Serial.print(" imuHz:");
    Serial.print(imuHz, 1);
    Serial.print(" batchHz:");
    Serial.print(batchHz, 1);
    Serial.print(" magHz:");
    Serial.print(magHz, 1);
    Serial.println();

    loopCount   = 0;
    imuSamples  = 0;
    imuBatches  = 0;
    magSamples  = 0;
    lastDebugMillis = currentTimestamp;
  }
}

