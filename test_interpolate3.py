import pandas as pd
df = pd.read_csv('raw_mocap_data/Case1.csv')
needed_bodies = ['ST_A', 'ST_B', 'ST_C', 'ST_D', 'snake_tip']
for time, group in df.groupby("Time"):
    missing = []
    for rb in needed_bodies:
        if len(group[group["RigidBody"] == rb]) == 0:
            missing.append(rb)
    if missing:
        print(f"Time {time} missing: {missing}")
