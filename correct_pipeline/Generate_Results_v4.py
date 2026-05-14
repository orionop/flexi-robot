import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Import builders to bypass Keras loading bugs
from Train_ForwardNN_v4 import build_v4_blackbox
from Train_InverseNN_v4 import build_v4_inverse

def generate_v4_figures():
    # Use a clean, professional style for the paper
    plt.style.use('seaborn-v0_8-paper')
    
    print("Loading Data and Models...")
    try:
        df = pd.read_csv("robot_data_fixed.csv")
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_Y = joblib.load('scaler_Y.pkl')
    except FileNotFoundError:
        print("Error: Missing CSV or Scaler files.")
        return

    # 1. Extract and format data
    raw_act = df[['act1', 'act2']].values
    raw_pos = df[['x', 'y', 'z']].values
    quat_true = df[['qx', 'qy', 'qz', 'qw']].values
    
    pos_true_mm = raw_pos * 1000.0
    act_true_mm = raw_act * 1000.0
    
    # Scale inputs
    X_scaled = scaler_X.transform(raw_act)
    Y_scaled = scaler_Y.transform(raw_pos)
    
    # 2. Load Models
    fk_model = build_v4_blackbox()
    fk_model.load_weights("forward_model_v4.keras")
    
    ik_model = build_v4_inverse()
    ik_model.load_weights("inverse_model_v4.keras")
    
    # 3. Generate Predictions
    print("Generating predictions for plots...")
    pos_pred_scaled = fk_model.predict(X_scaled, verbose=0)[0]
    pos_pred_mm = scaler_Y.inverse_transform(pos_pred_scaled) * 1000.0
    pos_errors = np.linalg.norm(pos_pred_mm - pos_true_mm, axis=1)

    act_pred_scaled = ik_model.predict({'target_position': Y_scaled, 'target_quaternion': quat_true}, verbose=0)
    act_pred_mm = scaler_X.inverse_transform(act_pred_scaled) * 1000.0
    
    # --- FIGURE 1: Error Distribution Histogram ---
    print("Plotting Figure 1: Error Histogram...")
    plt.figure(figsize=(8, 5))
    plt.hist(pos_errors, bins=30, color='teal', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(pos_errors), color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean Error ({np.mean(pos_errors):.1f} mm)')
    plt.title("Forward Kinematics: Tip Position Error Distribution")
    plt.xlabel("Absolute Position Error (mm)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("fig_v4_error_hist.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- FIGURE 2: 3D Workspace Deviation ---
    print("Plotting Figure 2: 3D Workspace Deviation...")
    # Sample 50 random points so the plot isn't a solid blur of 2000 lines
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    np.random.seed(42) # For reproducible plots
    indices = np.random.choice(len(pos_true_mm), size=50, replace=False)
    
    ax.scatter(pos_true_mm[indices, 0], pos_true_mm[indices, 1], pos_true_mm[indices, 2], 
               color='green', label='Ground Truth Tip', s=40)
    ax.scatter(pos_pred_mm[indices, 0], pos_pred_mm[indices, 1], pos_pred_mm[indices, 2], 
               color='red', marker='x', label='Predicted Tip', s=40)
    
    # Draw error lines connecting true to predicted
    for i in indices:
        ax.plot([pos_true_mm[i, 0], pos_pred_mm[i, 0]], 
                [pos_true_mm[i, 1], pos_pred_mm[i, 1]], 
                [pos_true_mm[i, 2], pos_pred_mm[i, 2]], color='gray', alpha=0.5)
        
    ax.set_title("3D Tip Position: Ground Truth vs Predicted (50 Random Samples)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.invert_zaxis() # Robot hangs downward
    ax.legend()
    plt.savefig("fig_v4_workspace_3d.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- FIGURE 3: Inverse Kinematics Motor Map ---
    print("Plotting Figure 3: IK Motor Map...")
    plt.figure(figsize=(8, 6))
    plt.scatter(act_true_mm[:, 0], act_true_mm[:, 1], color='blue', alpha=0.3, label='Ground Truth Actuation')
    plt.scatter(act_pred_mm[:, 0], act_pred_mm[:, 1], color='orange', alpha=0.3, marker='x', label='Predicted Actuation')
    plt.title("Inverse Kinematics: Motor Action Space")
    plt.xlabel("Actuator 1 (mm)")
    plt.ylabel("Actuator 2 (mm)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("fig_v4_ik_motors.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Success! Generated: \n - fig_v4_error_hist.png \n - fig_v4_workspace_3d.png \n - fig_v4_ik_motors.png")

if __name__ == "__main__":
    generate_v4_figures()