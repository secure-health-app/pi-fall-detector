"""
main.py

Entry point for the fall detection + ML feature pipeline.
"""

import os
import time
import requests
from dotenv import load_dotenv

from sense_reader import get_reading
from fall_logic import FallDetector
from ml_feature_extractor import WindowFeatureExtractor
from dataset_writer import DatasetWriter


# Load environment variables
load_dotenv()

BACKEND_URL    = os.getenv("BACKEND_URL", "http://localhost:8080")
DEVICE_API_KEY = os.getenv("DEVICE_API_KEY", "")
DEVICE_ID      = os.getenv("DEVICE_ID", "raspberry-pi")
POLL_INTERVAL  = float(os.getenv("POLL_INTERVAL", "0.1"))

# ML window settings
SESSION_LABEL  = os.getenv("SESSION_LABEL", "normal")
WINDOW_SECONDS = float(os.getenv("WINDOW_SECONDS", "1.0"))
WINDOW_STEP    = float(os.getenv("WINDOW_STEP", "0.5"))

ALERT_ENDPOINT = f"{BACKEND_URL}/api/alerts/fall"


def send_alert(fall_result: dict):
    payload = {
        "deviceId": DEVICE_ID,
        "peakAcceleration": fall_result["peak_acceleration"],
        "detectionPhase": fall_result["detection_phase"],
    }

    headers = {
        "Content-Type": "application/json",
        "X-Device-Key": DEVICE_API_KEY,
    }

    try:
        response = requests.post(ALERT_ENDPOINT, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            print(f"[main] Alert sent successfully: {response.json()}")
        else:
            print(f"[main] Backend error: {response.status_code} {response.text}")
    except Exception as e:
        print(f"[main] Failed to send alert: {e}")


def main():
    print("=" * 50)
    print(" SmartGuardian Fall Detection Service")
    print(f" Device:   {DEVICE_ID}")
    print(f" Backend:  {ALERT_ENDPOINT}")
    print(f" Interval: {POLL_INTERVAL}s")
    print("=" * 50)

    detector = FallDetector()

    # ML feature extractor setup
    sample_rate_hz = 1.0 / POLL_INTERVAL
    extractor = WindowFeatureExtractor(
        window_seconds=WINDOW_SECONDS,
        sample_rate_hz=sample_rate_hz
    )

    writer = DatasetWriter(out_dir="data")

    print(f"[main] ML dataset file: {writer.path}")
    print(f"[main] Session label: {SESSION_LABEL}")

    last_window_write = 0.0

    try:
        while True:
            reading = get_reading()

            # Feed reading into ML window buffer
            extractor.add(reading)

            # Normal session window writing
            now = time.time()
            if (
                SESSION_LABEL == "normal"
                and extractor.ready()
                and (now - last_window_write) >= WINDOW_STEP
            ):
                row = extractor.extract(label="normal")
                if row:
                    writer.write(row)
                    last_window_write = now

            # Rule-based fall detection
            result = detector.update(reading)

            if result and result.get("detected"):
                print("\n  FALL DETECTED - sending alert...")

                for _ in range(3):
                    # Write fall window
                    if extractor.ready():
                        fall_row = extractor.extract(label="fall")
                        if fall_row:
                            writer.write(fall_row)
                    time.sleep(WINDOW_STEP)

                send_alert(result)
                print("Resuming monitoring...\n")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[main] Stopped.")


if __name__ == "__main__":
    main()