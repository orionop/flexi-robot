import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
import joblib # For saving the scalers

def build_v4_blackbox():
    # Input: 2 motor actuations
    act_in = layers.Input(shape=(2,), name='actuation_in')
    
    # Deeper, wider network to learn the complex PLA deformation
    x = layers.Dense(256, activation='swish')(act_in)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x) # Prevents overfitting
    
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(128, activation='swish')(x)
    
    # Branch 1: Predict Position (x, y, z)
    pos_out = layers.Dense(3, name='position_out')(x)
    
    # Branch 2: Predict Orientation (qx, qy, qz, qw)
    quat_out = layers.Dense(4, name='quaternion_out')(x)
    
    model = Model(inputs=act_in, outputs=[pos_out, quat_out])
    
    # Compile with balanced loss weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'position_out': 'mse', 'quaternion_out': 'mse'},
        loss_weights={'position_out': 1.0, 'quaternion_out': 1.0} 
    )
    return model

if __name__ == "__main__":
    print("Loading and Scaling data...")
    try:
        df = pd.read_csv("robot_data_fixed.csv")
    except FileNotFoundError:
        print("Error: robot_data_fixed.csv not found.")
        exit()

    # 1. Extract raw data
    raw_X = df[['act1', 'act2']].values
    raw_Y_pos = df[['x', 'y', 'z']].values
    Y_quat = df[['qx', 'qy', 'qz', 'qw']].values # Quaternions are already -1 to 1!

    # 2. Initialize Scalers
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    # Fit and transform the data
    X_act_scaled = scaler_X.fit_transform(raw_X)
    Y_pos_scaled = scaler_Y.fit_transform(raw_Y_pos)

    # 3. Save the scalers so Evaluate_v4.py can use them later
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_Y, 'scaler_Y.pkl')

    model = build_v4_blackbox()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    ]

    print("Training Scaled V4 Deep Black Box...")
    model.fit(
        X_act_scaled, 
        {'position_out': Y_pos_scaled, 'quaternion_out': Y_quat},
        validation_split=0.2,
        epochs=400,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    model.save("forward_model_v4.keras")
    print("Success! V4 Model and Scalers saved.")