import pandas as pd
import numpy as np
import joblib
from ast import literal_eval
from scipy.optimize import least_squares, Bounds
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("combined_clean_sorted_dataset.csv")
df["actuation_pattern"] = df["actuation_pattern"].apply(literal_eval)

patterns = np.array(df["actuation_pattern"].tolist())
patterns_extended = np.hstack([patterns, np.ones((patterns.shape[0], 1))])
pose_columns = ['del_x', 'del_y', 'del_z', 'del_qx', 'del_qy', 'del_qz', 'del_qw']
delta_poses = df[pose_columns].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    patterns_extended, delta_poses, test_size=0.2, random_state=42
)
joblib.dump((X_train, X_test, y_train, y_test), "train_test_split_LMA.pkl")

# Residuals with descending weights and sum(weights)==1 penalty
def residuals(weights, pattern_vec, actual_deltas):
    scalar = np.dot(pattern_vec, weights)[:, None]
    pred = scalar * np.ones_like(actual_deltas)
    res = (pred - actual_deltas).ravel()

    # Penalty 1: Sum of weights ≠ 1
    penalty_sum = 1000 * (np.sum(weights) - 1) ** 2

    # Penalty 2: Enforce descending order: w1 ≥ w2 ≥ w3 ≥ w4 ≥ w5
    penalty_order = 10000 * sum([
    max(0, weights[i+1] - weights[i]) ** 2 for i in range(len(weights) - 1)
    ])

    total_penalty = penalty_sum + penalty_order
    return np.append(res, total_penalty)

# Initial guess: More weight to w1
w0 = np.array([0.6, 0.15, 0.1, 0.1, 0.05])
bounds = Bounds([0]*5, [1]*5)

# Run optimization
result = least_squares(
    residuals,
    w0,
    bounds=bounds,
    args=(X_train, y_train),
    verbose=2
)

# Results
weights_opt = result.x
print("\n📌 Trained Weights:", weights_opt)
print("✅ Sum of Weights:", np.sum(weights_opt))

# Test prediction
test_scalars = np.dot(X_test, weights_opt)[:, None]
y_pred = test_scalars * np.ones_like(y_test)
test_mse = np.mean((y_pred - y_test) ** 2)

print("\n📉 Test MSE on 20% data:", test_mse)
