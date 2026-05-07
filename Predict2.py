import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("inverse_model.h5", compile=False)

# Load the scalers used during training
scaler_pose = joblib.load("scaler_pose.pkl")
scaler_motor = joblib.load("scaler_motor.pkl")

# 🔽 Replace with your batch of input poses (each row: [x, y, z, qx, qy, qz, qw])
pose_input = np.array([
    [0.12, -0.08, 1.19, 0.05, -0.01, -0.26, 0.96],
    [0.10, -0.07, 1.17, 0.06,  0.00, -0.25, 0.96],
    [0.14, -0.09, 1.21, 0.04, -0.02, -0.27, 0.95]
])

# Normalize the pose inputs
pose_scaled = scaler_pose.transform(pose_input)

# Predict motor values (batch prediction)
motor_scaled = model.predict(pose_scaled)

# Inverse transform to get real motor outputs
motor_output = scaler_motor.inverse_transform(motor_scaled)

# Display each result
for i, (pose, motor) in enumerate(zip(pose_input, motor_output), 1):
    print(f"\n Pose {i} (x, y, z, qx, qy, qz, qw): {pose}")
    print(f" Predicted Actuations → act1: {motor[0]:.6f}, act2: {motor[1]:.6f}")
