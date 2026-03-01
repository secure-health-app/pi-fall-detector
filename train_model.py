import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load all window files
files = glob.glob("data/ml_windows_*.csv")
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

# Clean labels
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["normal", "fall"])].copy()

print("Label counts:")
print(df["label"].value_counts())

feature_cols = [
    "max_acc", "min_acc", "mean_acc", "std_acc",
    "max_gyro", "min_gyro", "mean_gyro", "std_gyro"
]

X = df[feature_cols]
y = (df["label"] == "fall").astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["normal", "fall"]))

joblib.dump({"model": model, "features": feature_cols}, "fall_model.joblib")
print("\nModel saved as fall_model.joblib")