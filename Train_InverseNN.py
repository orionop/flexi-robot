import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv('robot_data.csv')

# Inputs: Pose → 7 features coz it's a quaternion
X = df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values

# Outputs: Motor values → 2 targets
print("Available columns:", df.columns)
Y = df[['act1', 'act2']].values

# Normalize inputs and outputs
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_Y = MinMaxScaler()
Y_scaled = scaler_Y.fit_transform(Y)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Define exponential learning rate decay
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.9
)

# Build model
model = Sequential([
    Dense(32, input_dim=7, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2)  # Output: motor_1 and motor_2
])

# Compile
optimizer = Adam(learning_rate=lr_schedule, beta_1=0.9)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train
history = model.fit(
    X_train, Y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16
)

# Evaluate
loss, mae = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Save model and scalers
model.save("inverse_model.h5")

import joblib
joblib.dump(scaler_X, "scaler_pose.pkl")
joblib.dump(scaler_Y, "scaler_motor.pkl")
