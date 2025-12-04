# multi-imu

I have collected Accelerometer and Gyroscope data in an experiment and I want to use this data for analysis. One sensor is the Arduino that was attached on my head and has recorded acc, gyro and mag data at +- 50 Hz with 4g range. The other sensor is a custom IMU that records at 120Hz with 16g range.

I am looking for a python project that helps me processing this data.

It should
- Have some tools that helps me visualise the data, compare the sensors, see the acceleration and rotation etc.
- Be able to synchronize the data in time, now the recordings are not started at the same time and thus the movements are not lined up properly. So I need to be able to run an algorithm that finds the best time displacement that matches up these movements. I have done some synchronizing motions in the beginning of every recording so that can maybe be used.
- It should be able to align the axes, or do some kind of calibration or remove the gravity vector in or something else so the data can be used for fusion and be used for a motion model. 
- It should do some kind of danger analysis, like detect falls or heavy brake motions or maybe heavy turning or some other useful thing

Create all this in a modular python setup and use clear code that can easily be reviewed and adapted where needed.