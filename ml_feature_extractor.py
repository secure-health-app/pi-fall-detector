"""
ml_feature_extractor.py

Turns a rolling window of sensor readings into a single ML feature vector.
Each window becomes ONE row in a training CSV (much smaller than raw logging).
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import math
import statistics
import time
from typing import Deque, Optional


@dataclass
class FeatureRow:
    ts: float
    label: str
    max_acc: float
    min_acc: float
    mean_acc: float
    std_acc: float
    max_gyro: float
    min_gyro: float
    mean_gyro: float
    std_gyro: float


class WindowFeatureExtractor:
    def __init__(self, window_seconds: float = 1.0, sample_rate_hz: float = 10.0):
        self.window_seconds = window_seconds
        self.sample_rate_hz = sample_rate_hz
        self.maxlen = max(3, int(window_seconds * sample_rate_hz))
        self.buf: Deque[dict] = deque(maxlen=self.maxlen)

    def add(self, reading: dict) -> None:
        self.buf.append(reading)

    def ready(self) -> bool:
        return len(self.buf) >= self.maxlen

    def _gyro_magnitude(self, r: dict) -> float:
        gx, gy, gz = r["gyro_x"], r["gyro_y"], r["gyro_z"]
        return math.sqrt(gx * gx + gy * gy + gz * gz)

    def extract(self, label: str) -> Optional[FeatureRow]:
        if not self.ready():
            return None

        acc = [r["accel_magnitude"] for r in self.buf]
        gyro = [self._gyro_magnitude(r) for r in self.buf]

        # statistics.stdev requires at least 2 values
        std_acc = statistics.stdev(acc) if len(acc) > 1 else 0.0
        std_gyro = statistics.stdev(gyro) if len(gyro) > 1 else 0.0

        return FeatureRow(
            ts=time.time(),
            label=label,
            max_acc=max(acc),
            min_acc=min(acc),
            mean_acc=statistics.mean(acc),
            std_acc=std_acc,
            max_gyro=max(gyro),
            min_gyro=min(gyro),
            mean_gyro=statistics.mean(gyro),
            std_gyro=std_gyro,
        )