import pandas as pd
import numpy as np
import glob

# Path to your processed case files
case_files = sorted(glob.glob("case*_processed.csv"))

all_rows = []

for file in case_files:
    df = pd.read_csv(file)

    # Ensure required RigidBodies are present
    needed_bodies = ['ST_A', 'ST_B', 'ST_C', 'ST_D', 'snake_tip']
    if not all(rb in df["RigidBody"].unique() for rb in needed_bodies):
        continue  # Skip file if any key rigid body is missing

    # Get reference (first) string positions
    ref_positions = {}
    for rb in needed_bodies:
        ref_row = df[df["RigidBody"] == rb].iloc[0]
        ref_positions[rb] = np.array([ref_row["x"], ref_row["y"], ref_row["z"]])

    # Group by timestamp
    grouped = df.groupby("Time")

    for time, group in grouped:
        try:
            tip = group[group["RigidBody"] == "snake_tip"].iloc[0]
            A = group[group["RigidBody"] == "ST_A"].iloc[0]
            B = group[group["RigidBody"] == "ST_B"].iloc[0]
            C = group[group["RigidBody"] == "ST_C"].iloc[0]
            D = group[group["RigidBody"] == "ST_D"].iloc[0]

            tip_pos = np.array([tip["x"], tip["y"], tip["z"]])
            tip_quat = np.array([tip["qx"], tip["qy"], tip["qz"], tip["qw"]])

            # Compute string lengths (distance from tip to ST_*)
            len_A = np.linalg.norm(tip_pos - A[["x", "y", "z"]].values)
            len_B = np.linalg.norm(tip_pos - B[["x", "y", "z"]].values)
            len_C = np.linalg.norm(tip_pos - C[["x", "y", "z"]].values)
            len_D = np.linalg.norm(tip_pos - D[["x", "y", "z"]].values)

            # Compute reference string lengths
            ref_len_A = np.linalg.norm(ref_positions["snake_tip"] - ref_positions["ST_A"])
            ref_len_B = np.linalg.norm(ref_positions["snake_tip"] - ref_positions["ST_B"])
            ref_len_C = np.linalg.norm(ref_positions["snake_tip"] - ref_positions["ST_C"])
            ref_len_D = np.linalg.norm(ref_positions["snake_tip"] - ref_positions["ST_D"])

            # ΔL = current - reference
            dA = len_A - ref_len_A
            dB = len_B - ref_len_B
            dC = len_C - ref_len_C
            dD = len_D - ref_len_D

            # act1: A-C system, act2: B-D system
            act1 = dA - dC
            act2 = dB - dD

            all_rows.append({
                "x": tip_pos[0], "y": tip_pos[1], "z": tip_pos[2],
                "qx": tip_quat[0], "qy": tip_quat[1], "qz": tip_quat[2], "qw": tip_quat[3],
                "act1": act1,
                "act2": act2
            })
        except Exception as e:
            continue  # Skip timestamp if any value is missing

# Final DataFrame
robot_df = pd.DataFrame(all_rows)
robot_df.to_csv("robot_data.csv", index=False)
print("✅ robot_data.csv generated with", len(robot_df), "samples.")
