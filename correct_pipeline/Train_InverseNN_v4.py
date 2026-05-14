import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import joblib

def build_v4_inverse():
    # Inputs: Target position and Target orientation
    pos_in = layers.Input(shape=(3,), name='target_position')
    quat_in = layers.Input(shape=(4,), name='target_quaternion')
    
    # Combine the inputs into one vector
    x = layers.Concatenate()([pos_in, quat_in])
    
    # Deep network to map 3D space back to motor pulls
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(128, activation='swish')(x)
    
    # Output: 2 Motor Actuations
    act_out = layers.Dense(2, name='actuation_out')(x)
    
    model = Model(inputs=[pos_in, quat_in], outputs=act_out)
    
    # Actuations are scaled, so standard MSE is perfectly balanced
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

if __name__ == "__main__":
    print("Loading data and pre-fitted scalers...")
    try:
        df = pd.read_csv("robot_data_fixed.csv")
        scaler_X = joblib.load('scaler_X.pkl') # Loads Actuation scaler
        scaler_Y = joblib.load('scaler_Y.pkl') # Loads Position scaler
    except FileNotFoundError:
        print("Error: Files missing! Run Train_ForwardNN_v4.py first to generate scalers.")
        exit()

    # 1. Extract Ground Truth Data
    raw_X_act = df[['act1', 'act2']].values
    raw_Y_pos = df[['x', 'y', 'z']].values
    Y_quat = df[['qx', 'qy', 'qz', 'qw']].values

    # 2. Scale the data using the EXACT same rules as the Forward Model
    X_act_scaled = scaler_X.transform(raw_X_act) # This is our TARGET
    Y_pos_scaled = scaler_Y.transform(raw_Y_pos) # This is our INPUT

    model = build_v4_inverse()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    ]

    print("Training Scaled V4 Deep Inverse Model...")
    model.fit(
        {'target_position': Y_pos_scaled, 'target_quaternion': Y_quat},
        X_act_scaled,
        validation_split=0.2,
        epochs=400,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    model.save("inverse_model_v4.keras")
    print("Success! V4 Inverse Model saved.")