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

def quaternion_loss(y_true, y_pred):
    # L2 normalize just to be safe
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    dot = tf.reduce_sum(y_true * y_pred, axis=-1)
    return 1.0 - tf.square(dot)

def position_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

def build_forward_model():
    inputs = layers.Input(shape=(2,), name='actuation')
    
    # Shared backbone
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Position head
    pos_branch = layers.Dense(64, activation='relu')(x)
    pos_output = layers.Dense(3, name='position')(pos_branch)
    
    # Quaternion head
    quat_branch = layers.Dense(64, activation='relu')(x)
    quat_raw = layers.Dense(4)(quat_branch)
    quat_output = layers.Lambda(
        lambda q: tf.math.l2_normalize(q, axis=-1), 
        name='quaternion'
    )(quat_raw)
    
    model = Model(inputs=inputs, outputs=[pos_output, quat_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'position': 'mse',
            'quaternion': quaternion_loss
        },
        loss_weights={
            'position': 1.0,
            'quaternion': 5.0
        }
    )
    return model

def train():
    df = pd.read_csv("robot_data_fixed.csv")
    
    X = df[['act1_norm', 'act2_norm']].values
    Y_pos = df[['x', 'y', 'z']].values
    Y_quat = df[['qx', 'qy', 'qz', 'qw']].values
    
    # Scale positions (quaternions don't need scaling, they are already unit norm)
    scaler_pos = StandardScaler()
    Y_pos_scaled = scaler_pos.fit_transform(Y_pos)
    
    # Save the scaler for inference
    joblib.dump(scaler_pos, "scaler_pos_forward.pkl")
    print("Saved scaler_pos_forward.pkl")
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    best_val_loss = float('inf')
    best_model = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, X_val = X[train_idx], X[val_idx]
        Y_pos_train, Y_pos_val = Y_pos_scaled[train_idx], Y_pos_scaled[val_idx]
        Y_quat_train, Y_quat_val = Y_quat[train_idx], Y_quat[val_idx]
        
        model = build_forward_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )
        ]
        
        history = model.fit(
            X_train, 
            {'position': Y_pos_train, 'quaternion': Y_quat_train},
            validation_data=(
                X_val, 
                {'position': Y_pos_val, 'quaternion': Y_quat_val}
            ),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate on validation set
        val_loss = min(history.history['val_loss'])
        print(f"Fold {fold+1} Best Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            print("  New best model found!")
            
    # Save the best model
    best_model.save("forward_model_v2.keras")
    print("\nTraining complete. Saved best model to forward_model_v2.keras")

if __name__ == "__main__":
    train()
