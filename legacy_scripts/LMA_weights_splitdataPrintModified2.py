import pandas as pd
import numpy as np
import joblib
from ast import literal_eval
from scipy.optimize import least_squares, Bounds
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("combined_clean_sorted_dataset.csv")
df["actuation_pattern"] = df["actuation_pattern"].apply(literal_eval)

# Extract features and targets
patterns = np.array(df["actuation_pattern"].tolist())  # shape: (N, 4)
delta_poses = df[['del_x', 'del_y', 'del_z', 'del_qx', 'del_qy', 'del_qz', 'del_qw']].values  # shape: (N, 7)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(patterns, delta_poses, test_size=0.2, random_state=42)
joblib.dump((X_train, X_test, y_train, y_test), "train_test_split_LMA_Fixed4x5.pkl")

# Residual function: weights is flattened (20,) = 4 strings * 5 segments
def residuals(weights_flat, u, y_actual):
    weights = weights_flat.reshape(4, 5)  # shape: (4 strings, 5 segments)
    
    # Weighted actuation = (N, 4) @ (4, 5) = (N, 5 segments)
    segment_contrib = u @ weights
    
    # Net contribution per datapoint = sum over segments
    pred_scalar = segment_contrib.sum(axis=1).reshape(-1, 1)  # (N, 1)
    pred = np.repeat(pred_scalar, 7, axis=1)  # (N, 7)

    res = (pred - y_actual).ravel()

    # Penalty 1: encourage decreasing segment weights per string
    penalty_decreasing = 0
    for s in range(4):
        for i in range(4):
            penalty_decreasing += 10000 * max(0, weights[s, i+1] - weights[s, i]) ** 2

    # Penalty 2: each row sum close to 1
    penalty_row_sum = 1000 * np.sum((weights.sum(axis=1) - 1) ** 2)

    return np.append(res, penalty_decreasing + penalty_row_sum)

# Initial guess: each string pulls mostly segment 1, less later
w0 = np.tile(np.array([0.6, 0.2, 0.1, 0.07, 0.03]), 4)  # shape (20,)
bounds = Bounds([0]*20, [1]*20)

# Optimize
result = least_squares(residuals, w0, bounds=bounds, args=(X_train, y_train), verbose=2, max_nfev=1000)
weights_opt = result.x.reshape(4, 5)
print("\n🔧 Optimized Weights Matrix (4 strings × 5 segments):\n", weights_opt)
print("✅ Row-wise sums (should be ~1):", weights_opt.sum(axis=1))

# Prediction
segment_contrib_test = X_test @ weights_opt  # (N, 5)
pred_scalar_test = segment_contrib_test.sum(axis=1).reshape(-1, 1)
pred_test = np.repeat(pred_scalar_test, 7, axis=1)
mse = np.mean((pred_test - y_test)**2)
print("\n📉 Test MSE:", mse)

# Save results
columns = [f'pred_{c}' for c in ['del_x','del_y','del_z','del_qx','del_qy','del_qz','del_qw']] + \
          [f'actual_{c}' for c in ['del_x','del_y','del_z','del_qx','del_qy','del_qz','del_qw']]
df_result = pd.DataFrame(np.hstack((pred_test, y_test)), columns=columns)
df_result.to_excel("LMA_4x5_weights_Predictions.xlsx", index=False)
