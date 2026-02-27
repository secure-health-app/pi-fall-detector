"""
main.py

This is the entry point for the fall detection service on the Raspberry Pi.
It ties everything together - reading from the sensor, checking for falls,
and sending an alert to the backend if one is detected.

The loop runs continuously at whatever interval is set in POLL_INTERVAL.
I'm using 0.1 seconds (10 readings per second) as the default, which should
be fast enough to catch the freefall and impact phases without overloading the Pi.

Environment variables are loaded from a .env file (see .env.example).
This keeps sensitive values like the API key out of the code and off GitHub.
"""

import os
import time
import requests
from dotenv import load_dotenv
from sense_reader import get_reading
from fall_logic import FallDetector

# load environment variables from the .env file
load_dotenv()

BACKEND_URL    = os.getenv('BACKEND_URL', 'http://localhost:8080')
DEVICE_API_KEY = os.getenv('DEVICE_API_KEY', '')
DEVICE_ID      = os.getenv('DEVICE_ID', 'raspberry-pi-sense-hat')
POLL_INTERVAL  = float(os.getenv('POLL_INTERVAL', '0.1'))

ALERT_ENDPOINT = f"{BACKEND_URL}/api/alerts/fall"


def send_alert(fall_result: dict):
    """
    Sends a POST request to the Spring Boot backend with the fall details.

    I'm using a pre-shared API key in the request header (X-Device-Key)
    instead of a user JWT because the Pi isn't a user - it's a device.
    The backend checks this key before saving anything to the database.

    I'm wrapping this in a try/except so that if the backend is unreachable
    (e.g. laptop not on the same network), the Pi doesn't just crash -
    it prints an error and keeps monitoring.
    """
    payload = {
        'deviceId':         DEVICE_ID,
        'peakAcceleration': fall_result['peak_acceleration'],
        'detectionPhase':   fall_result['detection_phase']
    }
    headers = {
        'Content-Type': 'application/json',
        'X-Device-Key': DEVICE_API_KEY
    }

    try:
        response = requests.post(ALERT_ENDPOINT, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            print(f"[main] Alert sent successfully: {response.json()}")
        else:
            print(f"[main] Backend returned an error: {response.status_code} {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"[main] Couldn't reach the backend at {ALERT_ENDPOINT} - is Spring Boot running?")
    except requests.exceptions.Timeout:
        print("[main] Request timed out - backend took too long to respond.")
    except Exception as e:
        print(f"[main] Something went wrong sending the alert: {e}")


def main():
    print("=" * 50)
    print(" SmartGuardian Fall Detection Service")
    print(f" Device:   {DEVICE_ID}")
    print(f" Backend:  {ALERT_ENDPOINT}")
    print(f" Interval: {POLL_INTERVAL}s")
    print("=" * 50)

    detector = FallDetector()

    try:
        while True:
            reading = get_reading()
            result = detector.update(reading)

            # if update() returns something, a fall was confirmed
            if result and result.get('detected'):
                print("\n  FALL DETECTED - sending alert...")
                send_alert(result)
                print("Resuming monitoring...\n")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        # Ctrl+C to stop - useful when testing
        print("\n[main] Stopped.")


if __name__ == '__main__':
    main()