import pandas as pd
import numpy as np
from ast import literal_eval
from scipy.optimize import least_squares, Bounds
from sklearn.model_selection import train_test_split
import joblib
import openpyxl

# Load dataset
df = pd.read_csv("combined_clean_sorted_dataset.csv")
df["actuation_pattern"] = df["actuation_pattern"].apply(literal_eval)

# Inputs: actuation pattern + virtual actuator
patterns = np.array(df["actuation_pattern"].tolist())  # (N, 4)
patterns_extended = np.hstack([patterns, np.ones((patterns.shape[0], 1))])  # (N, 5)

# Outputs: delta pose (7D)
pose_columns = ['del_x', 'del_y', 'del_z', 'del_qx', 'del_qy', 'del_qz', 'del_qw']
delta_poses = df[pose_columns].values  # (N, 7)

# Train-test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    patterns_extended, delta_poses, test_size=0.2, random_state=42
)
joblib.dump((X_train, X_test, y_train, y_test), "train_test_split_LMA.pkl")

# Model: weighted sum per segment, 7 outputs
def model(W, pattern_vec):
    W = W.reshape(7, 5)  # (7, 5)
    return pattern_vec @ W.T  # (N, 7)

# Residuals for optimization
def residuals(W, pattern_vec, actual_deltas):
    preds = model(W, pattern_vec)
    return (preds - actual_deltas).ravel()

# Initial guess: small uniform values
W0 = np.ones((7, 5)) * 0.2
bounds = Bounds([0] * 35, [1] * 35)

# Run least squares
result = least_squares(
    residuals,
    W0.ravel(),
    bounds=bounds,
    args=(X_train, y_train),
    verbose=2
)

# Extract weights
W_opt = result.x.reshape(7, 5)
# Print each output dimension's weights
output_labels = ["del_x", "del_y", "del_z", "del_qx", "del_qy", "del_qz", "del_qw"]
for i in range(7):
    print(f"Weights for {output_labels[i]}: {W_opt[i]}")
print("\n📌 Optimized Weight Matrix (7 outputs × 5 segments):\n", W_opt)

# Predict on test set
y_pred = model(W_opt, X_test)
test_mse = np.mean((y_pred - y_test) ** 2)
print("\n📉 Test MSE on 20% data:", test_mse)

# Save predictions to Excel
df_out = pd.DataFrame(np.hstack([y_pred, y_test]), columns=[
    "pred_del_x", "pred_del_y", "pred_del_z", "pred_del_qx", "pred_del_qy", "pred_del_qz", "pred_del_qw",
    "actual_del_x", "actual_del_y", "actual_del_z", "actual_del_qx", "actual_del_qy", "actual_del_qz", "actual_del_qw"
])
df_out.to_excel("LMA_Predictions_Matrix_vs_Actual.xlsx", index=False)
print("✅ Saved predictions to 'LMA_Predictions_Matrix_vs_Actual.xlsx'")
