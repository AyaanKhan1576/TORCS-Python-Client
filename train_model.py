# ────────────────────────────────────────────────────────────
#  train_model.py
#  Trains a multi-output Random-Forest regressor
#  to predict [steering, accel, brake] from sensor data.
# ────────────────────────────────────────────────────────────
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

CSV_FILE = "driving_data_test.csv"
MODEL_FN = "torcs_model.pkl"

# 1) --------------------------------------------------------------------------
df = pd.read_csv(CSV_FILE).dropna()

# 2)  INPUT FEATURES  (feel free to add more)
feature_cols = ["speedX", "angle", "trackPos", "rpm"]
X = df[feature_cols].astype(float)

# 3)  TARGET EFFECTORS  (continuous)
target_cols = ["steering", "accel", "brake"]
y = df[target_cols].astype(float)

# 4) --------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape},  Val: {X_val.shape}")

# 5) --------------------------------------------------------------------------
base_rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model = MultiOutputRegressor(base_rf).fit(X_train, y_train)

# 6) --------------------------------------------------------------------------
print(f"Validation R²: {model.score(X_val, y_val):.4f}")

# 7) --------------------------------------------------------------------------
joblib.dump(model, MODEL_FN)
print(f"Saved model → {MODEL_FN}")
