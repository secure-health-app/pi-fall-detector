"""
fall_logic.py

This file handles the fall detection logic.
I'm using a 3-phase approach to detect falls:

1. Freefall  - the acceleration drops suddenly (person is falling)
2. Impact    - big spike in acceleration (person hits the ground)
3. Stillness - very little movement after impact (person is on the ground)

If all 3 happen in the right order and within a certain time window,
I count it as a confirmed fall and send an alert.

I'm also logging every reading to a CSV file so I can use the data
later to train a machine learning model.
"""

import csv
import os
from datetime import datetime

# these thresholds are in g (gravitational units)
FREE_FALL_THRESHOLD = 0.4    # if magnitude drops below this, could be freefall
IMPACT_THRESHOLD    = 2.0    # if magnitude spikes above this, likely an impact
STILLNESS_MIN       = 0.7    # stillness range - close to 1g means lying still
STILLNESS_MAX       = 1.3
STILLNESS_DURATION  = 1.0    # how many seconds of stillness before I confirm a fall

# where the CSV log gets saved
LOG_DIR  = os.path.join(os.path.dirname(__file__), 'data')
LOG_FILE = os.path.join(LOG_DIR, 'sensor_log.csv')

# column headers for the CSV - these will become features for the ML model later
CSV_HEADERS = [
    'timestamp', 'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z', 'accel_magnitude', 'label'
]


def _ensure_log_file():
    # creates the data folder and CSV file if they don't already exist
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def log_reading(reading: dict, label: str = 'normal'):
    """
    Saves a sensor reading to the CSV file with a label.

    The label tells me what was happening at that moment:
    - 'normal'         = regular movement, nothing unusual
    - 'freefall'       = low acceleration detected
    - 'impact'         = high acceleration spike
    - 'fall_confirmed' = all 3 phases matched, this was a real fall

    Having labelled data is important because when I train the ML model
    later, it needs to know what each reading actually represents.
    """
    _ensure_log_file()
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.fromtimestamp(reading['timestamp']).isoformat(),
            reading['accel_x'],
            reading['accel_y'],
            reading['accel_z'],
            reading['gyro_x'],
            reading['gyro_y'],
            reading['gyro_z'],
            reading['accel_magnitude'],
            label
        ])


class FallDetector:
    """
    This class keeps track of what phase we're in and decides if a fall happened.

    I'm using a state machine because a fall isn't just one moment -
    it's a sequence of events that have to happen in order.
    The states are:
        MONITORING -> FREEFALL_DETECTED -> IMPACT_DETECTED -> fall confirmed

    If something breaks the sequence (e.g. impact never comes after freefall),
    it resets back to MONITORING to avoid false positives.
    """

    def __init__(self):
        self._state = 'MONITORING'
        self._peak_acceleration = 0.0
        self._phase_start_time = None
        self._stillness_start_time = None

        # if freefall doesn't lead to impact within this time, reset
        # stops things like slowly bending down from triggering it
        self._phase_timeout = 3.0

    def update(self, reading: dict) -> dict | None:
        """
        Called with every new sensor reading from the Sense HAT.
        Returns a dict with fall details if confirmed, otherwise returns None.
        """
        mag = reading['accel_magnitude']
        now = reading['timestamp']

        # normal state - just watching for anything unusual
        if self._state == 'MONITORING':
            log_reading(reading, 'normal')

            if mag < FREE_FALL_THRESHOLD:
                self._state = 'FREEFALL_DETECTED'
                self._phase_start_time = now
                self._peak_acceleration = mag

        elif self._state == 'FREEFALL_DETECTED':
            log_reading(reading, 'freefall')

            # if it's been too long and no impact came, probably wasn't a fall
            if now - self._phase_start_time > self._phase_timeout:
                print("[FallDetector] Freefall timed out, resetting.")
                self._reset()
                return None

            if mag > IMPACT_THRESHOLD:
                self._state = 'IMPACT_DETECTED'
                self._peak_acceleration = mag
                self._phase_start_time = now
                print(f"[FallDetector] Impact detected (magnitude: {mag:.2f}g)")

        elif self._state == 'IMPACT_DETECTED':
            log_reading(reading, 'impact')
            print(f"[FallDetector] Impact phase mag={mag:.2f}g")

            # keep updating peak in case it gets even higher during impact
            if mag > self._peak_acceleration:
                self._peak_acceleration = mag

            # check if the person is now lying still (magnitude close to 1g = gravity only)
            if STILLNESS_MIN <= mag <= STILLNESS_MAX:
                if self._stillness_start_time is None:
                    self._stillness_start_time = now
                elif now - self._stillness_start_time >= STILLNESS_DURATION:
                    log_reading(reading, 'fall_confirmed')
                    print(f"[FallDetector] FALL CONFIRMED - peak acceleration: {self._peak_acceleration:.2f}g")
                    result = {
                        'detected': True,
                        'peak_acceleration': round(self._peak_acceleration, 3),
                        'detection_phase': 'IMPACT',
                        'timestamp': now
                    }
                    self._reset()
                    return result
            else:
                # they moved again, might have just stumbled and recovered
                self._stillness_start_time = None

            # safety timeout - if stuck in impact phase too long something went wrong
            if now - self._phase_start_time > 10.0:
                print("[FallDetector] Impact phase timed out, resetting.")
                self._reset()

        return None

    def _reset(self):
        # go back to the beginning and clear all the tracked values
        self._state = 'MONITORING'
        self._peak_acceleration = 0.0
        self._phase_start_time = None
        self._stillness_start_time = None