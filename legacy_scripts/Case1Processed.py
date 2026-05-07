import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

# Load data
df = pd.read_csv("case1.csv")

# Keep only useful columns and drop full NaN pose rows
pose_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
df_clean = df.dropna(subset=pose_cols, how='any')

# Reference pose: first valid snake_tip
snake_tip_df = df_clean[df_clean["RigidBody"] == "snake_tip"]
ref_pose = snake_tip_df.iloc[0]

# Compute Δx, Δy, Δz w.r.t. reference pose for all rows
df_clean["Δx"] = df_clean["x"] - ref_pose["x"]
df_clean["Δy"] = df_clean["y"] - ref_pose["y"]
df_clean["Δz"] = df_clean["z"] - ref_pose["z"]

# Quaternion delta: only for rows with valid quaternions
ref_quat = np.array([ref_pose["qx"], ref_pose["qy"], ref_pose["qz"], ref_pose["qw"]])
ref_rot = Rotation.from_quat(ref_quat)

# Create a Rotation object for all rows
rot_all = Rotation.from_quat(df_clean[["qx", "qy", "qz", "qw"]].values)
rel_rot = rot_all * ref_rot.inv()
df_clean[["Δqx", "Δqy", "Δqz", "Δqw"]] = rel_rot.as_quat()

# Optional: Interpolate ST_A positions over time if needed
st_a_df = df_clean[df_clean["RigidBody"] == "ST_A"]
if not st_a_df.empty:
    for axis in ['x', 'y', 'z']:
        valid_idx = st_a_df[axis].notna()
        interp_fn = interp1d(st_a_df.index[valid_idx], st_a_df[axis][valid_idx], kind='linear', fill_value="extrapolate")
        df_clean.loc[st_a_df.index, f"ST_A_{axis}_interp"] = interp_fn(st_a_df.index)

# Save to file
df_clean.to_csv("case1_processed.csv", index=False)
