# SmartGuardian - Raspberry Pi Fall Detection Module

## Overview

This module runs on a Raspberry Pi 5 equipped with a Sense HAT and performs real-time fall detection using accelerometer and gyroscope data.

Falls are detected using a three-stage sequence:

1. Sudden drop in acceleration (freefall)
2. Large acceleration spike (impact)
3. Low movement for several seconds (stillness)

If all three events occur in sequence within a defined time window, a fall is confirmed and an alert is sent to the backend.

Sensor data is also grouped into 1-second sliding windows. Each window is converted into statistical features and saved to a CSV file for supervised machine learning training.

Generated datasets were later used to train and evaluate Random Forest, SVM, and Logistic Regression models, with Random Forest selected for final deployment.

---

## Features

- Real-time fall detection on-device  
- Accelerometer + gyroscope sensing  
- Low-latency edge processing  
- Automatic backend alert submission  
- Sliding-window ML dataset generation  
- Configurable thresholds and timings  
- Environment-based deployment configuration

---

## Hardware

- Raspberry Pi 5
- Sense HAT V2 (accelerometer + gyroscope)
- Power supply

During testing the device was mounted at waist level using a belt-mounted case to better approximate how the sensor moves during a real fall.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/secure-health-app/pi-fall-detector.git
cd pi-fall-detector
```

### 2. Install Sense HAT drivers

```bash
sudo apt update
sudo apt install python3-sense-hat sense-hat
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
nano .env
```

| Variable | Description |
|---|---|
| `BACKEND_URL` | Backend API URL. Local example: `http://192.168.1.x:8080` or deployed prototype: `https://health-app-backend-icgv.onrender.com` |
| `DEVICE_API_KEY` | Pre-shared key - must match the value set in the backend config |
| `DEVICE_ID` | Name for this device, used in alert records in the database |
| `POLL_INTERVAL` | Seconds between sensor reads (default `0.1` = 10 reads per second) |
| `SESSION_LABEL` | Label for this recording session e.g. `normal` or `fall` |
| `WINDOW_SECONDS` | Length of each feature window in seconds (default `1.0`) |
| `WINDOW_STEP` | How often to write a new window in seconds (default `0.5`) |

## Running

```bash
python3 main.py
```

The terminal displays detector state transitions, impact events, and confirmed fall alerts.

Press `Ctrl+C` to stop.

## ML Dataset Generation

Sensor readings are grouped into sliding windows and converted into statistical features (mean, max, standard deviation per axis). Each window is saved as a row in:

```
data/ml_windows_<timestamp>.csv
```

The `SESSION_LABEL` environment variable controls how each row is labelled. Set it to `normal` during regular activity recording, and `fall` when simulating falls. This labelled dataset is used to train a supervised machine learning classifier.

---

## Project Structure

```
pi-fall-detector/
├── main.py                  # entry point, main detection loop
├── sense_reader.py          # reads raw data from the Sense HAT
├── fall_logic.py            # rule-based fall detection state machine
├── ml_feature_extractor.py  # groups readings into windows and extracts features
├── dataset_writer.py        # writes feature rows to CSV
├── requirements.txt
├── .env.example             # template for environment variables
└── data/
    └── ml_windows_<timestamp>.csv  # generated at runtime
```

---

## Author

Louise Deeth  
BSc (Hons) Software Development  
Atlantic Technological University
