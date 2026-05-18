import pandas as pd
import glob
file = "raw_mocap_data/Case1.csv"
df = pd.read_csv(file)
pose_cols = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# Original way
df_orig = df.dropna(subset=pose_cols, how='any')
print("Original remaining rows:", len(df_orig))

# New way
df = df.sort_values(by=['RigidBody', 'Time'])
for col in pose_cols:
    df[col] = df.groupby('RigidBody')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    
df_new = df.dropna(subset=pose_cols, how='any')
print("New remaining rows:", len(df_new))
