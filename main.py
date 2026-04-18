"""
main.py

Entry point for the fall detection + ML feature pipeline.
Rule-based detector acts as a first filter; RF model confirms before alerting.

SESSION_LABEL modes:
  normal  - writes normal windows continuously; RF confirms before alerting
  fall    - writes impact-time windows as "fall" when rule-based confirms;
            RF is skipped so bad model doesn't block data collection
"""

import os
import time
import requests
import joblib
import numpy as np
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

ALERT_ENDPOINT     = f"{BACKEND_URL}/api/alerts/fall"
DETECTION_COOLDOWN = float(os.getenv("DETECTION_COOLDOWN", "8.0"))
ALERT_COOLDOWN     = float(os.getenv("ALERT_COOLDOWN", "30.0"))

# Acceleration threshold for snapshotting the window at impact time
IMPACT_SNAPSHOT_THRESHOLD = float(os.getenv("IMPACT_SNAPSHOT_THRESHOLD", "2.0"))

# Load RF model
_bundle      = joblib.load("fall_model_best.joblib")
RF_MODEL     = _bundle["model"]
RF_SCALER    = _bundle["scaler"]        # None for Random Forest
FEATURE_COLS = _bundle["features"]


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

    max_retries = 3

    for attempt in range(max_retries):
        try:
            print(f"[main] Sending alert (attempt {attempt + 1})...")

            response = requests.post(
                ALERT_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=(5, 20)
            )

            if response.status_code == 200:
                print(f"[main] Alert sent successfully: {response.json()}")
                return
            else:
                print(f"[main] Backend error: {response.status_code} {response.text}")

        except requests.exceptions.Timeout:
            print("[main] Timeout - backend may be waking up (Render cold start)")

        except Exception as e:
            print(f"[main] Failed to send alert: {e}")

        # wait before retrying
        time.sleep(3)

    print("[main] All retry attempts failed. Alert not delivered.")


def main():
    print("=" * 50)
    print(" SmartGuardian Fall Detection Service")
    print(f" Device:   {DEVICE_ID}")
    print(f" Backend:  {ALERT_ENDPOINT}")
    print(f" Interval: {POLL_INTERVAL}s")
    print(f" Model:    fall_model_best.joblib")
    print(f" Mode:     {SESSION_LABEL}")
    print("=" * 50)

    if SESSION_LABEL == "fall":
        print("[main] FALL COLLECTION MODE — RF disabled, writing impact snapshots")
    else:
        print("[main] NORMAL MODE — RF active, alerts enabled")

    detector = FallDetector()

    # ML feature extractor setup
    sample_rate_hz = 1.0 / POLL_INTERVAL
    extractor = WindowFeatureExtractor(
        window_seconds=WINDOW_SECONDS,
        sample_rate_hz=sample_rate_hz
    )

    writer = DatasetWriter(out_dir="data")

    print(f"[main] ML dataset file: {writer.path}")
    print(f"[main] Features:        {FEATURE_COLS}")

    last_window_write   = 0.0
    last_detection_time = 0.0
    last_alert_time     = 0.0
    impact_snapshot     = None  # window captured at impact, before stillness fills it

    try:
        while True:
            reading = get_reading()

            # Feed reading into ML window buffer
            extractor.add(reading)

            # Snapshot the window at the moment of high acceleration (impact).
            # By the time the rule-based detector confirms a fall (after stillness),
            # the impact has already scrolled out of the buffer — so we save it here.
            if reading["accel_magnitude"] > IMPACT_SNAPSHOT_THRESHOLD and extractor.ready():
                impact_snapshot = extractor.extract(label="live")

            # Normal session: write normal windows continuously
            current_time = time.time()
            if (
                SESSION_LABEL == "normal"
                and extractor.ready()
                and (current_time - last_window_write) >= WINDOW_STEP
            ):
                row = extractor.extract(label="normal")
                if row:
                    writer.write(row)
                    last_window_write = current_time

            # Rule-based fall detection (first filter)
            result = detector.update(reading)

            if result and result.get("detected"):
                current_time = time.time()

                # Block duplicate detections — do NOT reset timer here
                if current_time - last_detection_time < DETECTION_COOLDOWN:
                    remaining = int(DETECTION_COOLDOWN - (current_time - last_detection_time))
                    print(f"[main] Detection ignored (duplicate movement, {remaining}s remaining)")
                    continue

                # Only update on a new valid detection
                last_detection_time = current_time

                # --- FALL COLLECTION MODE ---
                # RF is skipped — write the impact snapshot directly as "fall"
                # This ensures training data contains the actual impact window
                if SESSION_LABEL == "fall":
                    row = impact_snapshot if impact_snapshot else extractor.extract(label="fall")
                    impact_snapshot = None
                    if row:
                        row.label = "fall"
                        writer.write(row)
                        print(f"[main] Fall window written (max_acc={row.max_acc:.3f}g)")
                    else:
                        print("[main] No snapshot available — fall window not written")
                    continue

                # --- NORMAL MODE ---
                # Block alert spam
                if current_time - last_alert_time < ALERT_COOLDOWN:
                    remaining = int(ALERT_COOLDOWN - (current_time - last_alert_time))
                    print(f"[main] Fall detected but alert cooldown active ({remaining}s remaining)")
                    continue

                # RF model confirmation (second filter)
                # Use impact snapshot — contains the actual fall spike
                row = impact_snapshot if impact_snapshot else extractor.extract(label="live")
                impact_snapshot = None  # clear after use

                if row:
                    features = np.array([[
                        row.max_acc, row.min_acc, row.mean_acc, row.std_acc,
                        row.max_gyro, row.min_gyro, row.mean_gyro, row.std_gyro,
                    ]])
                    if RF_SCALER:
                        features = RF_SCALER.transform(features)
                    prediction = RF_MODEL.predict(features)[0]
                else:
                    prediction = 0
                    print("[main] RF skipped — window not ready")

                if prediction == 1:
                    # Both rule-based and RF agree — confirmed fall
                    print("\n  FALL DETECTED (RF confirmed) - sending alert...")

                    # Relabel and write to dataset
                    row.label = "fall"
                    writer.write(row)

                    send_alert(result)
                    last_alert_time = current_time
                    print("Resuming monitoring...\n")

                else:
                    # Rule-based triggered but RF says not a fall
                    print("[main] Rule-based trigger rejected by RF model — not sending alert")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[main] Stopped.")


if __name__ == "__main__":
    main()