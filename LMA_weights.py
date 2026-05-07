import pandas as pd
import numpy as np
from ast import literal_eval
from scipy.optimize import least_squares, Bounds
import joblib

# Load cleaned dataset
df = pd.read_csv("combined_clean_sorted_dataset.csv")

# Parse actuation pattern strings into actual tuples
df["actuation_pattern"] = df["actuation_pattern"].apply(literal_eval)

# Extract input: actuation patterns (list of 4-length tuples)
patterns = np.array(df["actuation_pattern"].tolist())  # Shape: (N, 4)

# Add a column of 1s for the fifth virtual actuator
patterns_extended = np.hstack([patterns, np.ones((patterns.shape[0], 1))])  # Shape: (N, 5)

# Extract output: delta pose vectors (shape: N x 7)
pose_columns = ['del_x', 'del_y', 'del_z', 'del_qx', 'del_qy', 'del_qz', 'del_qw']
delta_poses = df[pose_columns].values  # Shape: (N, 7)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    patterns_extended, delta_poses, test_size=0.2, random_state=42
)
joblib.dump((X_train, X_test, y_train, y_test), "train_test_split_LMA.pkl")

# Model: weighted scalar per pose
def model(weights, pattern_vec):
    return np.dot(pattern_vec, weights)[:, None]  # (N, 1)


# Residuals: predicted - actual
def residuals(weights, pattern_vec, actual_deltas):
    actuation_scalars = model(weights, pattern_vec)
    predicted_deltas = actuation_scalars * np.ones_like(actual_deltas)
    return (predicted_deltas - actual_deltas).ravel()

# Initial guess: equal weights
w0 = np.array([0.6, 0.15, 0.1, 0.1, 0.05])

# Enforce each weight in [0, 1]
bounds = Bounds([0]*5, [1]*5)

# Run least squares (no constraints, normalize later)
result = least_squares(
    residuals,
    w0,
    bounds=bounds,
    args=(X_train, y_train),
    verbose=2
)

# Normalize to ensure sum of weights = 1
weights = result.x / np.sum(result.x)

# Results
print("\n📌 Optimized Weights [w1, w2, w3, w4, w5]:")
print(weights)
print("\n✅ Sum of weights:", np.sum(weights))
print("📉 Final Cost (MSE-based):", result.cost / len(delta_poses))
