import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure predictable execution
np.random.seed(42)
tf.random.set_seed(42)

def build_inverse_model():
    pos_input = layers.Input(shape=(3,), name='position')
    quat_input = layers.Input(shape=(4,), name='quaternion')
    
    x = layers.Concatenate()([pos_input, quat_input])
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    
    act_output = layers.Dense(2, name='actuation')(x)
    
    model = Model(inputs=[pos_input, quat_input], outputs=act_output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model

def train():
    df = pd.read_csv("robot_data_fixed.csv")
    
    Y_act = df[['act1_norm', 'act2_norm']].values
    X_pos = df[['x', 'y', 'z']].values
    X_quat = df[['qx', 'qy', 'qz', 'qw']].values
    
    # Scale positions (quaternions stay untouched)
    scaler_pos = StandardScaler()
    X_pos_scaled = scaler_pos.fit_transform(X_pos)
    
    joblib.dump(scaler_pos, "scaler_pos_inverse.pkl")
    print("Saved scaler_pos_inverse.pkl")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_val_loss = float('inf')
    best_model = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(Y_act)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_pos_train, X_pos_val = X_pos_scaled[train_idx], X_pos_scaled[val_idx]
        X_quat_train, X_quat_val = X_quat[train_idx], X_quat[val_idx]
        Y_train, Y_val = Y_act[train_idx], Y_act[val_idx]
        
        model = build_inverse_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )
        ]
        
        history = model.fit(
            {'position': X_pos_train, 'quaternion': X_quat_train},
            Y_train,
            validation_data=(
                {'position': X_pos_val, 'quaternion': X_quat_val},
                Y_val
            ),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        val_loss = min(history.history['val_loss'])
        print(f"Fold {fold+1} Best Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            print("  New best model found!")
            
    best_model.save("inverse_model_v2.keras")
    print("\nTraining complete. Saved best model to inverse_model_v2.keras")

if __name__ == "__main__":
    train()
