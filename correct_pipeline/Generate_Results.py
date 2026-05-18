import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def generate_figures():
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    
    df = pd.read_csv("test_data.csv")
    
    X_cols = ['act_A_norm', 'act_B_norm', 'act_C_norm', 'act_D_norm', 
              'act_A_norm_t1', 'act_B_norm_t1', 'act_C_norm_t1', 'act_D_norm_t1']
    actuations_fk = df[X_cols].values
    pos_true = df[['del_x', 'del_y', 'del_z']].values
    
    pos_cols = ['del_x', 'del_y', 'del_z', 'del_x_t1', 'del_y_t1', 'del_z_t1']
    rot_cols = ['r6d_1', 'r6d_2', 'r6d_3', 'r6d_4', 'r6d_5', 'r6d_6', 
                'r6d_1_t1', 'r6d_2_t1', 'r6d_3_t1', 'r6d_4_t1', 'r6d_5_t1', 'r6d_6_t1']
    act_t1_cols = ['act_A_norm_t1', 'act_B_norm_t1', 'act_C_norm_t1', 'act_D_norm_t1']
    act_true_m = df[['act_A_norm', 'act_B_norm', 'act_C_norm', 'act_D_norm']].values
    
    from Train_ForwardNN_v2 import build_forward_model
    from Train_InverseNN_v2 import build_inverse_model
    
    fk_model = build_forward_model()
    fk_model.load_weights("forward_model_v2.keras")
    
    ik_model = build_inverse_model()
    ik_model.load_weights("inverse_model_v2.keras")
    
    scaler_pos_fwd = joblib.load("scaler_pos_forward.pkl")
    scaler_pos_inv = joblib.load("scaler_pos_inverse.pkl")
    max_act = joblib.load("actuation_scaler.pkl")["max_act"]
    
    pos_pred_scaled, rot_pred_6d = fk_model.predict(actuations_fk, verbose=0)
    pos_pred = scaler_pos_fwd.inverse_transform(pos_pred_scaled)
    
    pos_true_scaled_inv = scaler_pos_inv.transform(df[pos_cols].values)
    act_pred = ik_model.predict({
        'position': pos_true_scaled_inv, 
        'rotation': df[rot_cols].values,
        'actuation_t1': df[act_t1_cols].values
    }, verbose=0)
    
    # 1. 3D Trajectory Comparison
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    idx = np.random.choice(len(pos_true), min(500, len(pos_true)), replace=False)
    ax.scatter(pos_true[idx, 0], pos_true[idx, 1], pos_true[idx, 2], c='blue', label='Ground Truth (Qualisys)', alpha=0.6, s=20)
    ax.scatter(pos_pred[idx, 0], pos_pred[idx, 1], pos_pred[idx, 2], c='red', label='NN Prediction', alpha=0.6, s=20, marker='^')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Forward Kinematics: 3D Workspace Prediction')
    ax.legend()
    plt.savefig('fig_3d_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-Component Position Error Distribution
    errors_m = pos_pred - pos_true
    errors_mm = errors_m * 1000
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[errors_mm[:, 0], errors_mm[:, 1], errors_mm[:, 2]], palette="Set3")
    plt.xticks([0, 1, 2], ['X Error', 'Y Error', 'Z Error'])
    plt.ylabel('Error (mm)')
    plt.title('Forward Kinematics: Position Error Distribution')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('fig_position_error_box.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Inverse Kinematics Error
    act_true_m = act_true_m * max_act
    act_pred_m = act_pred * max_act
    act_error_mm = (act_pred_m - act_true_m) * 1000
    
    plt.figure(figsize=(8, 6))
    sns.histplot(np.linalg.norm(act_error_mm, axis=1), bins=30, kde=True, color='purple')
    plt.xlabel('Actuation Error Magnitude (mm)')
    plt.ylabel('Frequency')
    plt.title('Inverse Kinematics: Actuation Prediction Error')
    plt.savefig('fig_ik_error_hist.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Figures generated successfully (fig_3d_trajectory.png, fig_position_error_box.png, fig_ik_error_hist.png)")

if __name__ == "__main__":
    generate_figures()
