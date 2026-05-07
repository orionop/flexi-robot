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
joblib.dump((X_train, X_test, y_train, y_test), "train_test_split_LMA_4x5_improved.pkl")

# Residual function with enhanced constraints
def residuals(params, u, y_actual):
    # Parse parameters: first 20 are weights, next 7 are scaling factors
    weights = params[:20].reshape(4, 5)  # (4 strings, 5 segments)
    c = params[20:27]  # scaling factors (7,)
    
    # Calculate segment contributions (N, 5)
    segment_contrib = u @ weights
    
    # Compute scalar per sample (sum over segments) (N,)
    scalar = segment_contrib.sum(axis=1)
    
    # Apply component-specific scaling (N, 7)
    pred = scalar[:, None] * c
    
    # Calculate residuals
    res = (pred - y_actual).ravel()
    
    # Penalty 1: Strict decreasing weights per string
    penalty_decreasing = 0
    for s in range(4):
        # Enforce w1 > w2 > w3 > w4 > w5 with strong penalty
        for i in range(4):
            diff = weights[s, i+1] - weights[s, i]
            if diff > 0:
                penalty_decreasing += 100000 * (diff + 0.01)**2  # Add 0.01 to penalize even small violations
    
    # Penalty 2: Sum of weights per string = 1
    penalty_row_sum = 10000 * np.sum((weights.sum(axis=1) - 1)**2)
    
    # Penalty 3: Force w1 > 0.5 for all strings
    penalty_w1_min = 10000 * np.sum(np.maximum(0.5 - weights[:, 0], 0)**2)
    
    return np.append(res, penalty_decreasing + penalty_row_sum + penalty_w1_min)

# Initial guess - enforce strong first segment weights
w0_weights = np.tile(np.array([0.55, 0.25, 0.12, 0.06, 0.02]), 4)  # Sums to 1.0
c0 = np.median(y_train, axis=0) / 0.5  # Scale by approximate median displacement
params0 = np.concatenate([w0_weights, c0])

# Bounds: weights [0,1], scaling factors [-10,10]
lb = [0.01]*20 + [-10.0]*7
ub = [0.99]*20 + [10.0]*7
bounds = Bounds(lb, ub)

# Optimize with stricter settings
result = least_squares(
    residuals,
    params0,
    bounds=bounds,
    args=(X_train, y_train),
    verbose=2,
    ftol=1e-8,
    xtol=1e-8,
    max_nfev=5000
)

# Extract optimized parameters
params_opt = result.x
weights_opt = params_opt[:20].reshape(4, 5)
c_opt = params_opt[20:27]

print("\n🔧 Optimized Weights Matrix (4 strings × 5 segments):")
print(pd.DataFrame(weights_opt, 
                  columns=[f'Seg{i+1}' for i in range(5)],
                  index=[f'String{j+1}' for j in range(4)]))
print("\n✅ Row-wise sums:", weights_opt.sum(axis=1))
print("\n🔧 Component Scaling Factors:")
print(pd.Series(c_opt, index=['del_x','del_y','del_z','del_qx','del_qy','del_qz','del_qw']))

# Prediction function
def predict(u, weights, c):
    segment_contrib = u @ weights
    scalar = segment_contrib.sum(axis=1)
    return scalar[:, None] * c

# Evaluate
train_pred = predict(X_train, weights_opt, c_opt)
test_pred = predict(X_test, weights_opt, c_opt)

train_mse = np.mean((train_pred - y_train)**2)
test_mse = np.mean((test_pred - y_test)**2)
print(f"\n📊 Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")

# Save detailed results
def create_result_df(u, y_actual, pred, prefix=''):
    df_res = pd.DataFrame(u, columns=[f'String{i+1}' for i in range(4)])
    
    for i, col in enumerate(['del_x','del_y','del_z','del_qx','del_qy','del_qz','del_qw']):
        df_res[f'{prefix}actual_{col}'] = y_actual[:, i]
        df_res[f'{prefix}pred_{col}'] = pred[:, i]
        df_res[f'{prefix}err_{col}'] = pred[:, i] - y_actual[:, i]
    
    segment_contrib = u @ weights_opt
    for seg in range(5):
        df_res[f'Seg{seg+1}_contrib'] = segment_contrib[:, seg]
    df_res['Total_Scalar'] = segment_contrib.sum(axis=1)
    
    return df_res

# Save detailed results
train_df = create_result_df(X_train, y_train, train_pred, 'train_')
test_df = create_result_df(X_test, y_test, test_pred, 'test_')

with pd.ExcelWriter("LMA_Optimized_Results.xlsx") as writer:
    train_df.to_excel(writer, sheet_name="Train_Results", index=False)
    test_df.to_excel(writer, sheet_name="Test_Results", index=False)
    pd.DataFrame(weights_opt).to_excel(writer, sheet_name="Weights")
    pd.DataFrame(c_opt).to_excel(writer, sheet_name="Scaling_Factors")

print("💾 Results saved to LMA_Optimized_Results.xlsx")