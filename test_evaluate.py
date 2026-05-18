import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation
import joblib
import json

def r6d_to_matrix(r6d):
    x = r6d[:, 0:3]
    y = r6d[:, 3:6]
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_n = x / np.maximum(x_norm, 1e-8)
    dot = np.sum(x_n * y, axis=1, keepdims=True)
    y_ortho = y - dot * x_n
    y_norm = np.linalg.norm(y_ortho, axis=1, keepdims=True)
    y_n = y_ortho / np.maximum(y_norm, 1e-8)
    z_n = np.cross(x_n, y_n)
    return np.stack((x_n, y_n, z_n), axis=-1)

def evaluate_models():
    df = pd.read_csv("test_data.csv")
    X_cols = ['act_A_norm', 'act_B_norm', 'act_C_norm', 'act_D_norm', 
              'act_A_norm_t1', 'act_B_norm_t1', 'act_C_norm_t1', 'act_D_norm_t1']
    actuations_fk = df[X_cols].values
    pos_true = df[['del_x', 'del_y', 'del_z']].values
    quat_true = df[['qx', 'qy', 'qz', 'qw']].values
    
    scaler_pos_fwd = joblib.load("scaler_pos_forward.pkl")
    scaler_pos_inv = joblib.load("scaler_pos_inverse.pkl")
    max_act = joblib.load("actuation_scaler.pkl")["max_act"]
    
    from Train_ForwardNN_v2 import build_forward_model
    from Train_InverseNN_v2 import build_inverse_model
    
    fk_model = build_forward_model()
    fk_model.load_weights("forward_model_v2.keras")
    ik_model = build_inverse_model()
    ik_model.load_weights("inverse_model_v2.keras")
    
    print("--- Forward Kinematics Evaluation ---")
    pos_pred_scaled, rot_pred_6d = fk_model.predict(actuations_fk, verbose=0)
    pos_pred = scaler_pos_fwd.inverse_transform(pos_pred_scaled)
    
    pos_error_m = np.linalg.norm(pos_pred - pos_true, axis=1)
    rmse_pos_mm = np.sqrt(np.mean((pos_error_m * 1000)**2))
    mae_pos_mm = np.mean(pos_error_m * 1000)
    print(f"Position RMSE: {rmse_pos_mm:.2f} mm")
    print(f"Position MAE:  {mae_pos_mm:.2f} mm")
    
    rot_pred_matrices = r6d_to_matrix(rot_pred_6d)
    rot_pred = Rotation.from_matrix(rot_pred_matrices)
    rot_true = Rotation.from_quat(quat_true)
    rot_err = rot_pred.inv() * rot_true
    angle_err_deg = np.degrees(rot_err.magnitude())
    rmse_angle_deg = np.sqrt(np.mean(angle_err_deg**2))
    mae_angle_deg = np.mean(angle_err_deg)
    print(f"Orientation RMSE: {rmse_angle_deg:.2f}°")
    print(f"Orientation MAE:  {mae_angle_deg:.2f}°")
    
    print("\n--- Inverse Kinematics Evaluation ---")
    pos_cols = ['del_x', 'del_y', 'del_z', 'del_x_t1', 'del_y_t1', 'del_z_t1']
    rot_cols = ['r6d_1', 'r6d_2', 'r6d_3', 'r6d_4', 'r6d_5', 'r6d_6', 
                'r6d_1_t1', 'r6d_2_t1', 'r6d_3_t1', 'r6d_4_t1', 'r6d_5_t1', 'r6d_6_t1']
    act_t1_cols = ['act_A_norm_t1', 'act_B_norm_t1', 'act_C_norm_t1', 'act_D_norm_t1']
    
    pos_true_scaled_inv = scaler_pos_inv.transform(df[pos_cols].values)
    act_pred = ik_model.predict({
        'position': pos_true_scaled_inv, 
        'rotation': df[rot_cols].values,
        'actuation_t1': df[act_t1_cols].values
    }, verbose=0)
    
    act_true_m = df[['act_A_norm', 'act_B_norm', 'act_C_norm', 'act_D_norm']].values * max_act
    act_pred_m = act_pred * max_act
    
    act_error_m = np.linalg.norm(act_pred_m - act_true_m, axis=1)
    rmse_act_mm = np.sqrt(np.mean((act_error_m * 1000)**2))
    mae_act_mm = np.mean(act_error_m * 1000)
    print(f"Actuation RMSE: {rmse_act_mm:.2f} mm")
    print(f"Actuation MAE:  {mae_act_mm:.2f} mm")
    
    print("\n--- FK -> IK Roundtrip Consistency ---")
    pos_pred_6d_array = np.zeros((len(pos_pred), 6))
    pos_pred_6d_array[:, 0:3] = pos_pred
    pos_pred_6d_array[:, 3:6] = df[['del_x_t1', 'del_y_t1', 'del_z_t1']].values
    pos_pred_scaled_inv = scaler_pos_inv.transform(pos_pred_6d_array)
    
    rot_pred_12d = np.zeros((len(rot_pred_6d), 12))
    rot_pred_12d[:, 0:6] = rot_pred_6d
    rot_pred_12d[:, 6:12] = df[['r6d_1_t1', 'r6d_2_t1', 'r6d_3_t1', 'r6d_4_t1', 'r6d_5_t1', 'r6d_6_t1']].values
    
    act_roundtrip = ik_model.predict({
        'position': pos_pred_scaled_inv, 
        'rotation': rot_pred_12d,
        'actuation_t1': df[act_t1_cols].values
    }, verbose=0)
    
    act_rt_m = act_roundtrip * max_act
    rt_error_m = np.linalg.norm(act_rt_m - act_true_m, axis=1)
    rmse_rt_mm = np.sqrt(np.mean((rt_error_m * 1000)**2))
    print(f"Roundtrip Actuation RMSE: {rmse_rt_mm:.2f} mm")

if __name__ == "__main__":
    evaluate_models()
