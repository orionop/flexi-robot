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

def quaternion_loss(y_true, y_pred):
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    return 1.0 - tf.square(dot)

def generate_figures():
    # Set style for publication
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    
    # Load Data (using test set)
    df = pd.read_csv("test_data.csv")
    actuations = df[['act1_norm', 'act2_norm']].values
    pos_true = df[['del_x', 'del_y', 'del_z']].values
    quat_true = df[['qx', 'qy', 'qz', 'qw']].values
    
    # Load Models and Scalers
    from Train_ForwardNN_v2 import build_forward_model
    from Train_InverseNN_v2 import build_inverse_model
    
    fk_model = build_forward_model()
    fk_model.load_weights("forward_model_v2.keras")
    
    ik_model = build_inverse_model()
    ik_model.load_weights("inverse_model_v2.keras")
    scaler_pos_fwd = joblib.load("scaler_pos_forward.pkl")
    scaler_pos_inv = joblib.load("scaler_pos_inverse.pkl")
    max_act = joblib.load("actuation_scaler.pkl")["max_act"]
    
    # Generate Predictions
    pos_pred_scaled, quat_pred = fk_model.predict(actuations, verbose=0)
    pos_pred = scaler_pos_fwd.inverse_transform(pos_pred_scaled)
    quat_pred_norm = quat_pred / np.linalg.norm(quat_pred, axis=1)[:, np.newaxis]
    
    pos_true_scaled_inv = scaler_pos_inv.transform(pos_true)
    act_pred = ik_model.predict({'position': pos_true_scaled_inv, 'quaternion': quat_true}, verbose=0)
    
    # 1. 3D Trajectory Comparison
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for clearer plotting if necessary
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
    act_true_m = actuations * max_act
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
