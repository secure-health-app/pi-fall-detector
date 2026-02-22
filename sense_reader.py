"""
sense_reader.py

Reads raw accelerometer and gyroscope data from the Sense HAT's IMU.
This file is only responsible for getting clean sensor data.

The IMU gives me accelerometer and gyroscope data which I use to detect falls.

- Accelerometer: measures forces acting on the device (in g)
- Gyroscope: measures how fast the device is rotating (degrees/sec)

I'm turning off the compass here because I don't need it and it can
interfere with the other sensors if left on.
"""

import math
import time
from sense_hat import SenseHat


# Initialise the Sense HAT once when this module loads.
# This avoids reinitialising the IMU every time we take a reading.
_sense = SenseHat()

# Disable compass (not needed), enable gyro + accelerometer
_sense.set_imu_config(compass_enabled=False, gyro_enabled=True, accel_enabled=True)


def get_reading() -> dict:
    """
    Takes a single sensor sample from the IMU.

    Returns:
        A dictionary containing:
        - Raw acceleration values (x, y, z)
        - Raw gyroscope values (x, y, z)
        - Calculated acceleration magnitude
        - Timestamp (so we know exactly when this happened)
    """

    # Get raw acceleration and rotation data
    accel = _sense.get_accelerometer_raw()
    gyro = _sense.get_gyroscope_raw()

    # Extract axis values
    x = accel['x']
    y = accel['y']
    z = accel['z']

    # Calculate total acceleration magnitude using vector formula:
    # sqrt(x² + y² + z²)
    # This is important because falls are detected using overall force.
    magnitude = math.sqrt(x**2 + y**2 + z**2)

    # Return everything rounded slightly to reduce noise in logs
    return {
        'accel_x': round(x, 4),
        'accel_y': round(y, 4),
        'accel_z': round(z, 4),
        'gyro_x':  round(gyro['x'], 4),
        'gyro_y':  round(gyro['y'], 4),
        'gyro_z':  round(gyro['z'], 4),
        'accel_magnitude': round(magnitude, 4),
        'timestamp': time.time()  # Unix timestamp
    }


def get_smoothed_reading(samples: int = 3) -> dict:
    """
    Takes multiple readings and averages them.
    This helps reduce small sensor spikes and noise.

    Default = 3 samples.
    """

    # Take multiple readings quickly
    readings = [get_reading() for _ in range(samples)]

    avg = {}

    # Average each value
    for key in readings[0]:
        if key == 'timestamp':
            # Just use the most recent timestamp
            avg[key] = readings[-1][key]
        else:
            avg[key] = round(sum(r[key] for r in readings) / samples, 4)

    return avg