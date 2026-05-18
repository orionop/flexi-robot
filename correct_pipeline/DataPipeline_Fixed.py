import pandas as pd
import numpy as np
import glob
from scipy.spatial.transform import Slerp, Rotation

def process_data():
    case_files = sorted(glob.glob("../raw_mocap_data/Case*.csv"))
    all_raw_samples = []

    for file in case_files:
        df = pd.read_csv(file)
        pose_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        
        # Interpolate missing values per RigidBody over Time to recover data
        for col in pose_cols:
            df[col] = df.groupby('RigidBody')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
            
        df = df.dropna(subset=pose_cols, how='any')
        
        needed_bodies = ['ST_A', 'ST_B', 'ST_C', 'ST_D', 'snake_tip']
        if not all(rb in df["RigidBody"].unique() for rb in needed_bodies):
            print(f"Skipping {file}: missing rigid bodies")
            continue

        # Group by Time to handle duplicate timestamps
        grouped = df.groupby("Time")
        
        # Get reference positions (first timestamp)
        ref_positions = {}
        first_time = list(grouped.groups.keys())[0]
        first_group = grouped.get_group(first_time)
        
        for rb in needed_bodies:
            rb_rows = first_group[first_group["RigidBody"] == rb]
            if len(rb_rows) > 0:
                ref_positions[rb] = rb_rows[["x", "y", "z"]].mean().values
        
        if len(ref_positions) != 5:
            print(f"Skipping {file}: missing rigid bodies in first timestamp")
            continue
            
        case_samples = []
        
        for time, group in grouped:
            try:
                # Average readings for this timestamp
                bodies = {}
                for rb in needed_bodies:
                    rb_rows = group[group["RigidBody"] == rb]
                    if len(rb_rows) == 0:
                        raise ValueError(f"Missing {rb}")
                    bodies[rb] = rb_rows[["x", "y", "z"]].mean().values
                
                tip_rows = group[group["RigidBody"] == "snake_tip"]
                tip_pos = bodies["snake_tip"]
                
                # For quaternion, average and normalize
                quats = tip_rows[["qx", "qy", "qz", "qw"]].values
                mean_quat = np.mean(quats, axis=0)
                mean_quat /= np.linalg.norm(mean_quat)
                
                # Compute string displacements (magnitude with sign of Z change)
                # Tip is at z=1.2, ST is at z=1.7. Pulled = moves UP = +Z.
                displacements = {}
                for rb in ['ST_A', 'ST_B', 'ST_C', 'ST_D']:
                    delta_pos = bodies[rb] - ref_positions[rb]
                    sign = np.sign(delta_pos[2]) if abs(delta_pos[2]) > 0.001 else 1.0
                    disp = sign * np.linalg.norm(delta_pos)
                    displacements[rb] = disp
                
                act1 = displacements['ST_A'] - displacements['ST_C']
                act2 = displacements['ST_B'] - displacements['ST_D']
                
                # Calculate tip deltas from initial tip pose
                tip_delta_pos = tip_pos - ref_positions['snake_tip']
                
                case_samples.append({
                    "case": file,
                    "time": time,
                    "act1": act1,
                    "act2": act2,
                    "x": tip_pos[0], "y": tip_pos[1], "z": tip_pos[2],
                    "del_x": tip_delta_pos[0], "del_y": tip_delta_pos[1], "del_z": tip_delta_pos[2],
                    "qx": mean_quat[0], "qy": mean_quat[1], "qz": mean_quat[2], "qw": mean_quat[3]
                })
            except Exception as e:
                pass # Missing data at this timestamp
                
        all_raw_samples.extend(case_samples)
        
    df_raw = pd.DataFrame(all_raw_samples)
    
    # Global actuation normalization
    max_act = max(df_raw["act1"].abs().max(), df_raw["act2"].abs().max())
    df_raw["act1_norm"] = df_raw["act1"] / max_act
    df_raw["act2_norm"] = df_raw["act2"] / max_act
    
    # Data Augmentation (SLERP interpolation)
    print(f"Raw samples: {len(df_raw)}")
    
    augmented_samples = []
    
    # Group by case to interpolate within cases
    for case_name, group in df_raw.groupby("case"):
        group = group.sort_values("time").reset_index(drop=True)
        
        for i in range(len(group) - 1):
            row1 = group.iloc[i]
            row2 = group.iloc[i+1]
            
            # Skip if time gap is too large
            if row2["time"] - row1["time"] > 5000:
                continue
                
            augmented_samples.append(row1.to_dict())
            
            # Create 9 intermediate points
            t_vals = np.linspace(0, 1, 11)[1:-1]
            
            q1 = row1[["qx", "qy", "qz", "qw"]].values
            q2 = row2[["qx", "qy", "qz", "qw"]].values
            
            # Make sure quaternions are close (dot product > 0)
            if np.dot(q1, q2) < 0:
                q2 = -q2
                
            try:
                rotations = Rotation.from_quat([q1, q2])
                slerp = Slerp([0, 1], rotations)
                interp_quats = slerp(t_vals).as_quat()
            except:
                continue # Skip if slerp fails
                
            for j, t in enumerate(t_vals):
                new_row = {
                    "case": case_name,
                    "time": row1["time"] + t * (row2["time"] - row1["time"]),
                    "act1": (1-t)*row1["act1"] + t*row2["act1"],
                    "act2": (1-t)*row1["act2"] + t*row2["act2"],
                    "act1_norm": (1-t)*row1["act1_norm"] + t*row2["act1_norm"],
                    "act2_norm": (1-t)*row1["act2_norm"] + t*row2["act2_norm"],
                    "x": (1-t)*row1["x"] + t*row2["x"],
                    "y": (1-t)*row1["y"] + t*row2["y"],
                    "z": (1-t)*row1["z"] + t*row2["z"],
                    "del_x": (1-t)*row1["del_x"] + t*row2["del_x"],
                    "del_y": (1-t)*row1["del_y"] + t*row2["del_y"],
                    "del_z": (1-t)*row1["del_z"] + t*row2["del_z"],
                    "qx": interp_quats[j][0],
                    "qy": interp_quats[j][1],
                    "qz": interp_quats[j][2],
                    "qw": interp_quats[j][3]
                }
                augmented_samples.append(new_row)
                
        augmented_samples.append(group.iloc[-1].to_dict())
        
    df_aug = pd.DataFrame(augmented_samples)
    print(f"Augmented samples: {len(df_aug)}")
    
    # Save the data
    df_aug.to_csv("robot_data_fixed.csv", index=False)
    print("Saved to robot_data_fixed.csv")
    
    import joblib
    joblib.dump({"max_act": max_act}, "actuation_scaler.pkl")
    print(f"Saved actuation scaler (max_act = {max_act:.5f})")

if __name__ == "__main__":
    process_data()
