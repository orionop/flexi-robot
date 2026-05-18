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



def position_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

def build_forward_model():
    inputs = layers.Input(shape=(8,), name='actuation')
    
    # Smaller backbone
    x = layers.Dense(16, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    
    # Position head
    pos_branch = layers.Dense(16, activation='relu')(x)
    pos_output = layers.Dense(3, name='position')(pos_branch)
    
    # Rotation head (6D)
    rot_branch = layers.Dense(16, activation='relu')(x)
    rot_output = layers.Dense(6, name='rotation')(rot_branch)
    
    model = Model(inputs=inputs, outputs=[pos_output, rot_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'position': 'mse',
            'rotation': 'mse'
        },
        loss_weights={
            'position': 1.0,
            'rotation': 5.0
        }
    )
    return model

def train():
    df = pd.read_csv("robot_data_fixed.csv")
    
    # Proper train/test split
    df['case_base'] = df['case'].apply(lambda c: c.replace('\\', '/').split('/')[-1])
    test_cases = ['Case7.csv', 'Case8.csv']
    
    train_df = df[~df['case_base'].isin(test_cases)].copy()
    test_df = df[df['case_base'].isin(test_cases)].copy()
    
    test_df.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv with {len(test_df)} test samples")
    
    X_cols = ['act_A_norm', 'act_B_norm', 'act_C_norm', 'act_D_norm', 
              'act_A_norm_t1', 'act_B_norm_t1', 'act_C_norm_t1', 'act_D_norm_t1']
    X = train_df[X_cols].values
    Y_pos = train_df[['del_x', 'del_y', 'del_z']].values
    rot_cols = ['r6d_1', 'r6d_2', 'r6d_3', 'r6d_4', 'r6d_5', 'r6d_6']
    Y_rot = train_df[rot_cols].values
    
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
        Y_rot_train, Y_rot_val = Y_rot[train_idx], Y_rot[val_idx]
        
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
            {'position': Y_pos_train, 'rotation': Y_rot_train},
            validation_data=(
                X_val, 
                {'position': Y_pos_val, 'rotation': Y_rot_val}
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
