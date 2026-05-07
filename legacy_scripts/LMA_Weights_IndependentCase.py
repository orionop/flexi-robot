import pandas as pd
import numpy as np
from ast import literal_eval
from scipy.optimize import least_squares, Bounds
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("combined_clean_sorted_dataset.csv")
df["actuation_pattern"] = df["actuation_pattern"].apply(literal_eval)

pose_columns = ['del_x', 'del_y', 'del_z', 'del_qx', 'del_qy', 'del_qz', 'del_qw']
all_weights = []

# Loop over cases 1 to 8
for case_num in range(1, 9):
    print(f"\n================ CASE {case_num} ================\n")

    # Filter data for this case
    case_df = df[df['case'] == case_num].copy()
    
    if len(case_df) < 5:
        print(f"⚠️ Not enough data for case {case_num}, skipping...")
        continue

    # Inputs and outputs
    patterns = np.array(case_df["actuation_pattern"].tolist())
    patterns_extended = np.hstack([patterns, np.ones((patterns.shape[0], 1))])
    delta_poses = case_df[pose_columns].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        patterns_extended, delta_poses, test_size=0.2, random_state=42
    )

    # Residuals with penalty for sum(weights) ≠ 1
    def residuals(weights, pattern_vec, actual_deltas):
        scalar = np.dot(pattern_vec, weights)[:, None]
        pred = scalar * np.ones_like(actual_deltas)
        res = (pred - actual_deltas).ravel()
        penalty = 1000 * (np.sum(weights) - 1) ** 2
        return np.append(res, penalty)

    # Initial guess w more wt on 1st seg
    w0 = np.array([0.6, 0.15, 0.1, 0.1, 0.05])
    bounds = Bounds([0]*5, [1]*5)

    # Least squares optimization
    result = least_squares(
        residuals,
        w0,
        bounds=bounds,
        args=(X_train, y_train),
        verbose=0  # set to 2 if you want detailed logs
    )

    # Extracting weights and test error
    weights_opt = result.x
    all_weights.append(weights_opt)

    test_scalars = np.dot(X_test, weights_opt)[:, None]
    y_pred = test_scalars * np.ones_like(y_test)
    test_mse = np.mean((y_pred - y_test) ** 2)

    # results
    print(f"📌 Trained Weights: {weights_opt}")
    print(f"✅ Sum of Weights: {np.sum(weights_opt)}")
    print(f"📉 Test MSE: {test_mse}")

# Avg wts across all cases...
if all_weights:
    avg_weights = np.mean(all_weights, axis=0)
    print("\n================ FINAL SUMMARY ================\n")
    for i, weights in enumerate(all_weights, 1):
        print(f"Case {i} Weights: {weights}")
    print("\n📌 Average Weights Across All Cases:", avg_weights)
    print("✅ Sum of Average Weights:", np.sum(avg_weights))
