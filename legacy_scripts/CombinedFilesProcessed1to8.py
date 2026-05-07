import pandas as pd

# List of files to combine
file_paths = [f'case{i}_processed.csv' for i in range(1, 9)]

# Load and combine
df_all = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

# Drop any remaining NaNs (if any)
df_all = df_all.dropna(subset=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'act1', 'act2'])

# Save final dataset
df_all.to_csv('robot_data_combined.csv', index=False)
