# Scaling is not actually happening since range is b/w 0 and 1 only. So Actual values are outputted 
# which don't need to be unscaled
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and scalers
model = load_model("inverse_model.h5", compile=False)  # compile=False to avoid loading loss issues
scaler_motor = joblib.load("scaler_motor.pkl")
scaler_pose = joblib.load("scaler_pose.pkl")

# Testing Printing...coz scaler_motor fn doesn't support mean and other such fns
print("Motor scaler min:", scaler_motor.min_)
print("Motor scaler scale:", scaler_motor.scale_)
print("Motor scaler data_min:", scaler_motor.data_min_)
print("Motor scaler data_max:", scaler_motor.data_max_)
print("Motor scaler data_range:", scaler_motor.data_range_)

# Example multiple input poses (x, y, z, qx, qy, qz, qw)
input_poses = np.array([
    [0.02, 0.03, 0.12, 0, 0, 0.7071, 0.7071],
    [0.01, 0.02, 0.10, 0.1, 0.1, 0.7, 0.7],
    [0.03, 0.04, 0.15, 0, 0, 1, 0],
])

# Scale the input poses using the pose scaler (important!)
input_scaled = scaler_pose.transform(input_poses)

# Predict motor angles (scaled output)
motor_angles_scaled = model.predict(input_scaled)

# Unscale motor angles back to real-world values
motor_angles = scaler_motor.inverse_transform(motor_angles_scaled)

# Print results
for i, pose in enumerate(input_poses):
    print(f"Input Pose {i+1} (x, y, z, qx, qy, qz, qw):/n{pose}")
    print(f"Predicted Motor Angles: {motor_angles_scaled[i]} (real units, since scaler range is [0,1])")
    print("-" * 40)

