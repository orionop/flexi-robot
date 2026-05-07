import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import joblib

# Load dataset
df = pd.read_csv('robot_data.csv')

# Forward Model: Inputs = motor values (act1, act2), Outputs = Pose (x, y, z, qx, qy, qz, qw)
X = df[['act1', 'act2']].values
Y = df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values

# Normalize inputs and outputs
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_Y = MinMaxScaler()
Y_scaled = scaler_Y.fit_transform(Y)

# Train-test split
X_train, X_test, y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Define learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100,
    decay_rate=0.9
)

# Build model
model = Sequential([
    Dense(32, input_dim=2, activation='relu'),
    Dense(64, activation='relu'),
    Dense(7)  # x, y, z, qx, qy, qz, qw
])

# Compile
model.compile(optimizer=Adam(learning_rate=lr_schedule, beta_1=0.9),
              loss='mse',
              metrics=['mae'])

# Train and store training history
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16
)

# Evaluate on test set
loss, mae = model.evaluate(X_test, Y_test)
print(f"\nTest Loss: {loss:.6f}, Test MAE: {mae:.6f}")

# Show final validation metrics
final_val_loss = history.history['val_loss'][-1]
final_val_mae = history.history['val_mae'][-1]
print(f"Final Validation Loss: {final_val_loss:.6f}, Final Validation MAE: {final_val_mae:.6f}")


# Evaluate
loss, mae = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss:.6f}, Test MAE: {mae:.6f}")

# Save model and scalers
model.save("forward_model.h5")
joblib.dump(scaler_X, "scaler_motor.pkl")
joblib.dump(scaler_Y, "scaler_pose.pkl")
