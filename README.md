# SmartGuardian – Raspberry Pi Fall Detection Module

## Overview

This module runs on a Raspberry Pi 5 equipped with a Sense HAT and performs real-time fall detection using accelerometer and gyroscope data.

Falls are detected using a three-stage sequence:

1. Sudden drop in acceleration (freefall)
2. Large acceleration spike (impact)
3. Low movement for several seconds (stillness)

If all three occur in the right order within a set time window, a fall is confirmed and an alert is sent to the backend.

Sensor data is also grouped into 1-second sliding windows. Each window is converted into statistical features and saved to a CSV file for supervised machine learning training.

## Hardware

- Raspberry Pi 5
- Sense HAT V2 (accelerometer + gyroscope)
- Power supply

During testing the device was mounted at waist level using a belt-mounted case to better approximate how the sensor moves during a real fall.

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
| `BACKEND_URL` | URL of the Spring Boot backend e.g. `http://192.168.1.x:8080` |
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

The terminal will show the current state of the detector and print a message when a fall is confirmed.

Press `Ctrl+C` to stop.

## ML Dataset Generation

Sensor readings are grouped into sliding windows and converted into statistical features (mean, max, standard deviation per axis). Each window is saved as a row in:

```
data/ml_windows_<timestamp>.csv
```

The `SESSION_LABEL` environment variable controls how each row is labelled. Set it to `normal` during regular activity recording, and `fall` when simulating falls. This labelled dataset is used to train a supervised machine learning classifier.

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
