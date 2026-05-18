import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("correct_pipeline/robot_data_fixed.csv")
df['case_base'] = df['case'].apply(lambda c: c.replace('\\', '/').split('/')[-1])

# Re-derive the 4 tendon displacements if possible from the CSV?
# Wait, robot_data_fixed.csv doesn't save the raw ST_A, ST_B, ST_C, ST_D!
print(df.columns)
