import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares, Bounds

# -------------------------
# 1. Load and prepare data
# -------------------------
df = pd.read_csv("combined_clean_sorted_dataset.csv")
df["actuation_pattern"] = df["actuation_pattern"].apply(literal_eval)

# Extract inputs and outputs
patterns = np.array(df["actuation_pattern"].tolist())
patterns_extended = np.hstack([patterns, np.ones((patterns.shape[0], 1))])
pose_columns = ['del_x', 'del_y', 'del_z', 'del_qx', 'del_qy', 'del_qz', 'del_qw']
delta_poses = df[pose_columns].values

# 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    patterns_extended, delta_poses, test_size=0.2, random_state=42
)

# -------------------------
# 2. Define your average weights (from earlier)
# -------------------------
avg_weights = np.array([0.34104563, 0.25610208, 0.1765479, 0.22281153, 0.00340257])

# -------------------------
# 3. Estimate a realistic unit direction vector from training data
# -------------------------
unit_direction = np.mean(y_train, axis=0)
unit_direction /= np.linalg.norm(unit_direction)  # Normalize to unit vector

# -------------------------
# 4. Predict using average weights and direction vector
# -------------------------
def predict_batch(X, weights, direction_vector):
    scalars = np.dot(X, weights)[:, None]  # shape (N, 1)
    return scalars * direction_vector  # shape (N, 7)

y_pred = predict_batch(X_test, avg_weights, unit_direction)

# -------------------------
# 5. Evaluate error
# -------------------------
test_mse = np.mean((y_pred - y_test) ** 2)
print("📉 MSE on 20% test data using average weights model:", test_mse)

# Optional: check a few predictions
print("\n🔍 Sample predictions vs actuals:")
for i in range(3):
    print(f"\n🔸 Predicted: {np.round(y_pred[i], 4)}")
    print(f"🔸 Actual   : {np.round(y_test[i], 4)}")
