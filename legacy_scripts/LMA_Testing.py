import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import random

# CASE 1 only
df = pd.read_csv("case1_processed.csv").dropna()

# Extract actuation (assumed ST_A_x_interp is the actuation input)
act = df["ST_A_x_interp"].values.reshape(-1, 1)  # shape: (N, 1)

# Extract pose delta: Δx, Δy, Δz, Δqx, Δqy (5D output)
pose_delta = df[["Δx", "Δy", "Δz", "Δqx", "Δqy"]].values  # shape: (N, 5)

# Residual func: diff b/w predicted and actual
def residuals(w, act, pose_delta):
    return (act @ w.reshape(1, -1) - pose_delta).flatten()

# Initial weights (5 elements for the 5D pose delta)
w0 = np.zeros(5)

# Levenberg-Marquardt Alg aka LMA
result = least_squares(residuals, w0, args=(act, pose_delta), method='lm')

# Estimated weights
weights = result.x
print(f"\n Estimated Weights [w1 to w5]:\n{weights}")

# Mse...
mse = np.mean(residuals(weights, act, pose_delta) ** 2)
print(f"\n Mean Squared Error (MSE): {mse:.6f}")

# Testing fr random actuation value
idx = random.randint(0, len(df) - 1)
test_act = act[idx][0]
actual_pose = pose_delta[idx]

# Predicted using model
predicted_pose = test_act * weights

print(f"\n Random Sample Index: {idx}")
print(f"Actuation: {test_act:.6f}")
print(f"\n Actual ΔPose [Δx, Δy, Δz, Δqx, Δqy]:\n{actual_pose}")
print(f" Predicted ΔPose using model:\n{predicted_pose}")

# Error vector
error = actual_pose - predicted_pose
print(f"\n Pose Error Vector:\n{error}")
print(f" Norm of Pose Error: {np.linalg.norm(error):.6f}")
