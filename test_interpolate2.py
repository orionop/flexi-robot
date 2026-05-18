import pandas as pd
import glob
file = "raw_mocap_data/Case1.csv"
df = pd.read_csv(file)
pose_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

df = df.sort_values(by=['RigidBody', 'Time'])
for col in pose_cols:
    df[col] = df.groupby('RigidBody')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
df = df.dropna(subset=pose_cols, how='any')

needed_bodies = ['ST_A', 'ST_B', 'ST_C', 'ST_D', 'snake_tip']
grouped = df.groupby("Time")
valid = 0
for time, group in grouped:
    if all(rb in group["RigidBody"].values for rb in needed_bodies):
        valid += 1
print("Valid timestamps with all 5 bodies:", valid)
