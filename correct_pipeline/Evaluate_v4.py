import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Import both builders to bypass Keras loading bugs
from Train_ForwardNN_v4 import build_v4_blackbox
from Train_InverseNN_v4 import build_v4_inverse

def evaluate_v4():
    print("Loading data and saved scalers...")
    try:
        df = pd.read_csv("robot_data_fixed.csv")
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_Y = joblib.load('scaler_Y.pkl')
    except FileNotFoundError as e:
        print(f"Error: Missing file! {e}")
        return

    # 1. Extract Ground Truth Data (Multiplying by 1000 here sets the baseline in mm)
    raw_X = df[['act1', 'act2']].values
    act_true_mm = raw_X * 1000.0  
    
    raw_Y = df[['x', 'y', 'z']].values
    pos_true_mm = raw_Y * 1000.0  
    
    quat_true = df[['qx', 'qy', 'qz', 'qw']].values

    # 2. Scale the inputs for the models
    X_scaled = scaler_X.transform(raw_X)
    Y_scaled = scaler_Y.transform(raw_Y)

    print("Rebuilding V4 Architectures and loading weights...")
    
    # Load Forward Model
    fk_model = build_v4_blackbox()
    fk_model.load_weights("forward_model_v4.keras")
    
    # Load Inverse Model
    ik_model = build_v4_inverse()
    ik_model.load_weights("inverse_model_v4.keras")

    print("\nRunning predictions...")

    # --- FORWARD EVALUATION ---
    predictions = fk_model.predict(X_scaled, verbose=0)
    pos_pred_scaled = predictions[0] 
    
    # Inverse Transform and calculate error
    pos_pred_mm = scaler_Y.inverse_transform(pos_pred_scaled) * 1000.0
    pos_error_mm = np.linalg.norm(pos_pred_mm - pos_true_mm, axis=1)

    # --- INVERSE EVALUATION ---
    act_pred_scaled = ik_model.predict({'target_position': Y_scaled, 'target_quaternion': quat_true}, verbose=0)
    
    # Inverse Transform and calculate error
    act_pred_mm = scaler_X.inverse_transform(act_pred_scaled) * 1000.0
    act_error_mm = np.linalg.norm(act_pred_mm - act_true_mm, axis=1)

    # --- RESULTS ---
    print(f"\n" + "="*45)
    print(f"       V4 DEEP BLACK BOX RESULTS")
    print(f"="*45)
    print(f"FORWARD KINEMATICS (Actuation -> Position)")
    print(f"  RMSE:     {np.sqrt(np.mean(pos_error_mm**2)):.2f} mm")
    print(f"  MAE:      {np.mean(pos_error_mm):.2f} mm")
    print(f"  Max Error:{np.max(pos_error_mm):.2f} mm")
    print(f"-"*45)
    print(f"INVERSE KINEMATICS (Position -> Actuation)")
    print(f"  RMSE:     {np.sqrt(np.mean(act_error_mm**2)):.2f} mm")
    print(f"  MAE:      {np.mean(act_error_mm):.2f} mm")
    print(f"  Max Error:{np.max(act_error_mm):.2f} mm")
    print("="*45 + "\n")

if __name__ == "__main__":
    evaluate_v4()