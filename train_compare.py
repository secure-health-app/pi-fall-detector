"""
train_compare.py

Compares Random Forest, SVM, and Logistic Regression
on the same windowed feature dataset used in train_model.py.

Prints a summary table and saves the best model as fall_model_best.joblib.
"""

import glob
import time
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Load data

files = glob.glob("data/ml_windows_*.csv")
if not files:
    raise FileNotFoundError("No ml_windows_*.csv files found in data/")

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["normal", "fall"])].copy()

print("Label counts:")
print(df["label"].value_counts())
print()

FEATURE_COLS = [
    "max_acc", "min_acc", "mean_acc", "std_acc",
    "max_gyro", "min_gyro", "mean_gyro", "std_gyro",
]

X = df[FEATURE_COLS].values
y = (df["label"] == "fall").astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# SVM and LR need scaled features; RF does not but scaling won't hurt it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# Models

models = {
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        X_train,        # RF doesn't need to be scaled
        X_test,
    ),
    "SVM (RBF)": (
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
        X_train_scaled,
        X_test_scaled,
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        X_train_scaled,
        X_test_scaled,
    ),
}


# Train & evaluate

results = []

for name, (model, X_tr, X_te) in models.items():
    print(f"{'=' * 50}")
    print(f" {name}")
    print(f"{'=' * 50}")

    start = time.perf_counter()
    model.fit(X_tr, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_te)
    infer_time = (time.perf_counter() - start) / len(X_te) * 1000  # ms per sample

    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=["normal", "fall"],
        output_dict=True
    )
    print(classification_report(y_test, y_pred, target_names=["normal", "fall"]))

    results.append({
        "Model":            name,
        "Fall Recall":      round(report["fall"]["recall"],     3),
        "Fall Precision":   round(report["fall"]["precision"],  3),
        "Fall F1":          round(report["fall"]["f1-score"],   3),
        "Overall Accuracy": round(report["accuracy"],           3),
        "Train Time (s)":   round(train_time,                   3),
        "Infer Time (ms)":  round(infer_time,                   4),
    })

print("\n" + "=" * 50)
print(" SUMMARY")
print("=" * 50)
summary = pd.DataFrame(results).set_index("Model")
print(summary.to_string())


# Save best model by fall recall 
# Fall recall is the most important metric - missing a fall is worse than a false alarm

best_name = max(results, key=lambda r: (r["Fall F1"], r["Fall Recall"]))["Model"]
best_model, best_X_tr, _ = models[best_name]

print(f"\nBest model by fall recall: {best_name}")

# Re-fit on full training data before saving
best_model.fit(best_X_tr, y_train)
joblib.dump(
    {"model": best_model, "features": FEATURE_COLS, "scaler": scaler if best_name != "Random Forest" else None},
    "fall_model_best.joblib"
)
print(f"Saved as fall_model_best.joblib")