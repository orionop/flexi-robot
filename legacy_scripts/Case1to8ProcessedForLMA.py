import pandas as pd
import os

# Config
required_bodies = ["snake_tip", "ST_D", "ST_C", "ST_A", "ST_B"]
pose_cols = ["x", "y", "z", "qx", "qy", "qz", "qw"]

# Pulling patterns (as described)
pulling_patterns = {
    1: (1, 0, 0, 0),
    2: (0, 1, 0, 0),
    3: (0, 0, 1, 0),
    4: (0, 0, 0, 1),
    5: (1, 0, 1, 0),
    6: (1, 0, 0, 1),
    7: (0, 1, 1, 0),
    8: (0, 1, 0, 1)
}

combined = []

for case_num in range(1, 9):
    file = f"case{case_num}_processed.csv"
    print(f"🔍 Processing {file}...")

    if not os.path.exists(file):
        print(f"❌ File {file} not found. Skipping.")
        continue

    df = pd.read_csv(file)

    # Only keep relevant columns
    df = df[["Time", "RigidBody"] + pose_cols]

    # Drop rows with NaNs in pose
    df = df.dropna(subset=pose_cols)

    # Group by time
    grouped = df.groupby("Time")

    ref_pose = None

    for _, group in grouped:
        if set(group["RigidBody"]) >= set(required_bodies):
            # Sort group by required body order
            sorted_group = pd.concat([group[group["RigidBody"] == rb].head(1) for rb in required_bodies], ignore_index=True)

            # Extract only pose data for snake_tip (row 0)
            current_pose = sorted_group.iloc[0][pose_cols].values.astype(float)

            if ref_pose is None:
                ref_pose = current_pose
                continue  # skip reference row

            delta_pose = current_pose - ref_pose

            entry = {
                "case": case_num,
                "actuation_pattern": pulling_patterns[case_num],
                **{f"Δ{col}": delta_pose[i] for i, col in enumerate(pose_cols)}
            }
            combined.append(entry)

    if not combined:
        print(f"❌ No valid clean pose groups found in {file}.")

# Save combined dataset
if combined:
    combined_df = pd.DataFrame(combined)
    combined_df.to_csv("combined_clean_sorted_dataset.csv", index=False)
    print("✅ Combined dataset saved to combined_clean_sorted_dataset.csv")
else:
    print("❌ No valid data found in any case.")
