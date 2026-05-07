import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

# Load Case 2 data
df = pd.read_csv("case2.csv")

# Define which columns must be present
pose_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# Drop rows where pose data is missing
df_clean = df.dropna(subset=pose_cols, how='any').copy()

# Get reference pose: first valid snake_tip
snake_tip_df = df_clean[df_clean["RigidBody"] == "snake_tip"]
ref_pose = snake_tip_df.iloc[0]

# Δx, Δy, Δz relative to reference
df_clean["Δx"] = df_clean["x"] - ref_pose["x"]
df_clean["Δy"] = df_clean["y"] - ref_pose["y"]
df_clean["Δz"] = df_clean["z"] - ref_pose["z"]

# Quaternion delta: compute using scipy Rotation
ref_quat = np.array([ref_pose["qx"], ref_pose["qy"], ref_pose["qz"], ref_pose["qw"]])
ref_rot = Rotation.from_quat(ref_quat)

# Apply to all cleaned data
rot_all = Rotation.from_quat(df_clean[["qx", "qy", "qz", "qw"]].values)
rel_rot = rot_all * ref_rot.inv()
df_clean[["Δqx", "Δqy", "Δqz", "Δqw"]] = rel_rot.as_quat()

# Optional interpolation for ST_A
st_a_df = df_clean[df_clean["RigidBody"] == "ST_A"]
if not st_a_df.empty:
    for axis in ['x', 'y', 'z']:
        valid_idx = st_a_df[axis].notna()
        if valid_idx.any():  # Ensure there are valid entries to interpolate
            interp_fn = interp1d(
                st_a_df.index[valid_idx],
                st_a_df[axis][valid_idx],
                kind='linear',
                fill_value="extrapolate"
            )
            df_clean.loc[st_a_df.index, f"ST_A_{axis}_interp"] = interp_fn(st_a_df.index)

# Save the processed data
df_clean.to_csv("case2_processed.csv", index=False)

print("✅ Case 2 processed and saved to case2_processed.csv")
