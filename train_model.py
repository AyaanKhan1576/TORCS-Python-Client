# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# 1) Load the raw CSV logged by your manual runs
df = pd.read_csv("driving_data.csv")

# 2) Select your input FEATURES (sensor readings)
#    Adjust these as needed to match your Driver.drive() feature set
feature_cols = [
    "speedX",    # longitudinal speed
    "angle",     # car heading relative to track axis
    "trackPos",  # lateral offset from track center
    "rpm"        # engine RPM
]
X = df[feature_cols].astype(float)

# 3) Select the 7 control TARGETS you logged:
#    steer_input, accel_input, brake_input, left_key, right_key, up_key, down_key
y_cols = [
    "steer_input",
    "accel_input",
    "brake_input",
    "left_key",
    "right_key",
    "up_key",
    "down_key"
]
y = df[y_cols].astype(float)

# 4) Split into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape} | Val size: {X_val.shape}")

# 5) Build & train a multi-output Random Forest regressor
base_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model = MultiOutputRegressor(base_rf)
model.fit(X_train, y_train)

# 6) Evaluate on hold-out set
r2 = model.score(X_val, y_val)
print(f"Validation R²: {r2:.4f}")

# 7) Save the trained model for use in driver.py
#    Use the filename expected by your Driver (torcmodel.pkl)
joblib.dump(model, "torcmodel.pkl")
print("Saved torcsmodel.pkl")
