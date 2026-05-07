import pandas as pd
import numpy as np
from scipy.optimize import least_squares

# Config for Case 1
filename = "case1_processed.csv"
actuator_column = "ST_A_x_interp"  # Only ST_A pulling in case 1
feature_columns = ["Δx", "Δy", "Δz", "Δqx", "Δqy"]  # 5 pose deltas

print(f"\nProcessing {filename} with actuator column '{actuator_column}'")

# Load n clean data
try:
    df = pd.read_csv(filename)
    df = df.dropna(subset=[actuator_column] + feature_columns)

    # Get act. vector and pose deltas
    actuation = df[[actuator_column]].values  # shape: (N, 1)
    pose_deltas = df[feature_columns].values  # shape: (N, 5)

    # Define model: pose_deltas = actuation * [w1 w2 w3 w4 w5]
    def residuals(weights, act, pose):
        pred = act @ weights.reshape(1, -1)  # shape: (N, 5)
        return (pred - pose).flatten()

    # Initial guess
    w0 = np.zeros(5)

    # Solve using Levenberg-Marquardt
    result = least_squares(residuals, w0, args=(actuation, pose_deltas), method='lm')

    weights = result.x
    print(f"Est. Weights [w1, w2, w3, w4, w5] for string 1:\n{weights}")

    # Calculate predns and MSE
    predictions = actuation @ weights.reshape(1, -1)
    mse = np.mean((predictions - pose_deltas) ** 2)
    print(f"Mean Squared Error: {mse:.6f}")

except Exception as e:
    print(f"Error processing {filename}: {e}")
