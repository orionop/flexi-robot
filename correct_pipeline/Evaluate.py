import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation
import joblib
import json

def quaternion_loss(y_true, y_pred):
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    return 1.0 - tf.square(dot)

def evaluate_models():
    # Load data
    df = pd.read_csv("robot_data_fixed.csv")
    actuations = df[['act1_norm', 'act2_norm']].values
    pos_true = df[['x', 'y', 'z']].values
    quat_true = df[['qx', 'qy', 'qz', 'qw']].values
    
    # Load scalers
    scaler_pos_fwd = joblib.load("scaler_pos_forward.pkl")
    scaler_pos_inv = joblib.load("scaler_pos_inverse.pkl")
    actuation_scaler = joblib.load("actuation_scaler.pkl")
    max_act = actuation_scaler["max_act"]
    
    from Train_ForwardNN_v2 import build_forward_model
    from Train_InverseNN_v2 import build_inverse_model
    
    # Load models by building the architecture and loading weights
    # This bypasses Keras serialization bugs across versions
    fk_model = build_forward_model()
    fk_model.load_weights("forward_model_v2.keras")
    
    ik_model = build_inverse_model()
    ik_model.load_weights("inverse_model_v2.keras")
    
    print("--- Forward Kinematics Evaluation ---")
    pos_pred_scaled, quat_pred = fk_model.predict(actuations, verbose=0)
    pos_pred = scaler_pos_fwd.inverse_transform(pos_pred_scaled)
    
    # Position Error
    pos_error_m = np.linalg.norm(pos_pred - pos_true, axis=1)
    pos_error_mm = pos_error_m * 1000
    
    rmse_pos_mm = np.sqrt(np.mean(pos_error_mm**2))
    mae_pos_mm = np.mean(pos_error_mm)
    
    print(f"Position RMSE: {rmse_pos_mm:.2f} mm")
    print(f"Position MAE:  {mae_pos_mm:.2f} mm")
    
    # Orientation Error
    # Make sure all quaternions are normalized before passing to Rotation
    quat_pred_norm = quat_pred / np.linalg.norm(quat_pred, axis=1)[:, np.newaxis]
    
    rot_pred = Rotation.from_quat(quat_pred_norm)
    rot_true = Rotation.from_quat(quat_true)
    
    rot_err = rot_pred.inv() * rot_true
    angle_err_rad = rot_err.magnitude()
    angle_err_deg = np.degrees(angle_err_rad)
    
    rmse_angle_deg = np.sqrt(np.mean(angle_err_deg**2))
    mae_angle_deg = np.mean(angle_err_deg)
    
    print(f"Orientation RMSE: {rmse_angle_deg:.2f}°")
    print(f"Orientation MAE:  {mae_angle_deg:.2f}°")
    
    print("\n--- Inverse Kinematics Evaluation ---")
    pos_true_scaled_inv = scaler_pos_inv.transform(pos_true)
    act_pred = ik_model.predict(
        {'position': pos_true_scaled_inv, 'quaternion': quat_true}, 
        verbose=0
    )
    
    # Re-scale back to real displacement (meters)
    act_true_m = actuations * max_act
    act_pred_m = act_pred * max_act
    
    act_error_m = np.linalg.norm(act_pred_m - act_true_m, axis=1)
    act_error_mm = act_error_m * 1000
    
    rmse_act_mm = np.sqrt(np.mean(act_error_mm**2))
    mae_act_mm = np.mean(act_error_mm)
    
    print(f"Actuation RMSE: {rmse_act_mm:.2f} mm")
    print(f"Actuation MAE:  {mae_act_mm:.2f} mm")
    
    print("\n--- FK -> IK Roundtrip Consistency ---")
    # Feed predicted poses back into IK
    pos_pred_scaled_inv = scaler_pos_inv.transform(pos_pred)
    act_roundtrip = ik_model.predict(
        {'position': pos_pred_scaled_inv, 'quaternion': quat_pred_norm}, 
        verbose=0
    )
    
    act_rt_m = act_roundtrip * max_act
    rt_error_m = np.linalg.norm(act_rt_m - act_true_m, axis=1)
    rt_error_mm = rt_error_m * 1000
    
    rmse_rt_mm = np.sqrt(np.mean(rt_error_mm**2))
    print(f"Roundtrip Actuation RMSE: {rmse_rt_mm:.2f} mm")

    metrics = {
        "fk_pos_rmse_mm": float(rmse_pos_mm),
        "fk_pos_mae_mm": float(mae_pos_mm),
        "fk_angle_rmse_deg": float(rmse_angle_deg),
        "fk_angle_mae_deg": float(mae_angle_deg),
        "ik_act_rmse_mm": float(rmse_act_mm),
        "ik_act_mae_mm": float(mae_act_mm),
        "roundtrip_act_rmse_mm": float(rmse_rt_mm)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
if __name__ == "__main__":
    evaluate_models()
