import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

# List of cases to process
case_files = ["case1.csv","case2.csv","case3.csv", "case4.csv","case5.csv","case6.csv","case7.csv","case8.csv"]

for file in case_files:
    print(f"🔄 Processing {file}...")

    # Load file
    df = pd.read_csv(file)

    # Define pose-related columns
    pose_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

    # Drop rows with any missing pose data
    df_clean = df.dropna(subset=pose_cols, how='any').copy()

    # Get reference pose from first valid snake_tip entry
    snake_tip_df = df_clean[df_clean["RigidBody"] == "snake_tip"]
    if snake_tip_df.empty:
        print(f"⚠️ No valid 'snake_tip' in {file}. Skipping.")
        continue
    ref_pose = snake_tip_df.iloc[0]

    # Position deltas
    df_clean["Δx"] = df_clean["x"] - ref_pose["x"]
    df_clean["Δy"] = df_clean["y"] - ref_pose["y"]
    df_clean["Δz"] = df_clean["z"] - ref_pose["z"]

    # Orientation deltas (quaternions)
    ref_quat = np.array([ref_pose["qx"], ref_pose["qy"], ref_pose["qz"], ref_pose["qw"]])
    ref_rot = Rotation.from_quat(ref_quat)

    rot_all = Rotation.from_quat(df_clean[["qx", "qy", "qz", "qw"]].values)
    rel_rot = rot_all * ref_rot.inv()
    df_clean[["Δqx", "Δqy", "Δqz", "Δqw"]] = rel_rot.as_quat()

    # Interpolation of ST_A
    st_a_df = df_clean[df_clean["RigidBody"] == "ST_A"]
    if not st_a_df.empty:
        for axis in ['x', 'y', 'z']:
            valid_idx = st_a_df[axis].notna()
            if valid_idx.any():
                interp_fn = interp1d(
                    st_a_df.index[valid_idx],
                    st_a_df[axis][valid_idx],
                    kind='linear',
                    fill_value="extrapolate"
                )
                df_clean.loc[st_a_df.index, f"ST_A_{axis}_interp"] = interp_fn(st_a_df.index)

    # Save file
    output_file = file.replace(".csv", "_processed.csv")
    df_clean.to_csv(output_file, index=False)
    print(f"✅ Saved processed file as: {output_file}")
