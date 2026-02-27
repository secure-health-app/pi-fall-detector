"""
dataset_writer.py

Writes ML feature rows to a CSV file.
One row per window (not raw 10Hz logs).
"""

import csv
import os
from datetime import datetime
from typing import Optional
from ml_feature_extractor import FeatureRow


class DatasetWriter:
    HEADERS = [
        "timestamp",
        "label",
        "max_acc",
        "min_acc",
        "mean_acc",
        "std_acc",
        "max_gyro",
        "min_gyro",
        "mean_gyro",
        "std_gyro",
    ]

    def __init__(self, out_dir: str = "data", filename: Optional[str] = None):
        os.makedirs(out_dir, exist_ok=True)

        if filename is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"ml_windows_{stamp}.csv"

        self.path = os.path.join(out_dir, filename)

        # Create file + write headers if new
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)

    def write(self, row: FeatureRow) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.fromtimestamp(row.ts).isoformat(),
                    row.label,
                    f"{row.max_acc:.6f}",
                    f"{row.min_acc:.6f}",
                    f"{row.mean_acc:.6f}",
                    f"{row.std_acc:.6f}",
                    f"{row.max_gyro:.6f}",
                    f"{row.min_gyro:.6f}",
                    f"{row.mean_gyro:.6f}",
                    f"{row.std_gyro:.6f}",
                ]
            )