import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("correct_pipeline/robot_data_fixed.csv")
df['case_base'] = df['case'].apply(lambda c: c.replace('\\', '/').split('/')[-1])
test_cases = ['Case7.csv', 'Case8.csv']
train_df = df[~df['case_base'].isin(test_cases)].copy()
test_df = df[df['case_base'].isin(test_cases)].copy()

X_train = train_df[['act1_norm', 'act2_norm']].values
Y_train = train_df[['del_x', 'del_y', 'del_z']].values
X_test = test_df[['act1_norm', 'act2_norm']].values
Y_test = test_df[['del_x', 'del_y', 'del_z']].values

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
pos_error_m = np.linalg.norm(Y_pred - Y_test, axis=1)
pos_error_mm = pos_error_m * 1000

print(f"RF Position RMSE: {np.sqrt(np.mean(pos_error_mm**2)):.2f} mm")
print(f"RF Position MAE:  {np.mean(pos_error_mm):.2f} mm")
