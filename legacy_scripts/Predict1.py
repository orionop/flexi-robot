import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model without compiling (to avoid deserialization issues)
model = load_model("inverse_model.h5", compile=False)

# Load the scalers
scaler_pose = joblib.load("scaler_pose.pkl")
scaler_motor = joblib.load("scaler_motor.pkl")

# ex input: Pose (x, y, z, qx, qy, qz, qw) 
# We need 2 input the real pose here...
pose_input = np.array([[0.1000,-0.1220,0.1033,0.6059,0.6453,-0.1237,0.4486]])

# Normalize pose
pose_scaled = scaler_pose.transform(pose_input)

# Predict the motor values
motor_scaled = model.predict(pose_scaled)

# Inverse transform to get real motor values...See the scaling code (Predic_Unscale_Multiple.py)
motor_output = scaler_motor.inverse_transform(motor_scaled)

print("/nInput Pose (x, y, z, qx, qy, qz, qw):")
print(pose_input)
print("/nPredicted Motor Angles:")
print(motor_output)
