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
    pos_input = layers.Input(shape=(6,), name='position')
    rot_input = layers.Input(shape=(12,), name='rotation')
    act_t1_input = layers.Input(shape=(4,), name='actuation_t1')
    
    x = layers.Concatenate()([pos_input, rot_input, act_t1_input])
    
    x = layers.Dense(16, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    
    act_output = layers.Dense(4, name='actuation')(x)
    
    model = Model(inputs=[pos_input, rot_input, act_t1_input], outputs=act_output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model

def train():
    df = pd.read_csv("robot_data_fixed.csv")
    
    df['case_base'] = df['case'].apply(lambda c: c.replace('\\', '/').split('/')[-1])
    test_cases = ['Case7.csv', 'Case8.csv']
    train_df = df[~df['case_base'].isin(test_cases)].copy()
    
    pos_cols = ['del_x', 'del_y', 'del_z', 'del_x_t1', 'del_y_t1', 'del_z_t1']
    rot_cols = ['r6d_1', 'r6d_2', 'r6d_3', 'r6d_4', 'r6d_5', 'r6d_6', 
                'r6d_1_t1', 'r6d_2_t1', 'r6d_3_t1', 'r6d_4_t1', 'r6d_5_t1', 'r6d_6_t1']
    act_t1_cols = ['act_A_norm_t1', 'act_B_norm_t1', 'act_C_norm_t1', 'act_D_norm_t1']
    
    Y_act = train_df[['act_A_norm', 'act_B_norm', 'act_C_norm', 'act_D_norm']].values
    X_pos = train_df[pos_cols].values
    X_rot = train_df[rot_cols].values
    X_act_t1 = train_df[act_t1_cols].values
    
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
        X_rot_train, X_rot_val = X_rot[train_idx], X_rot[val_idx]
        X_act_t1_train, X_act_t1_val = X_act_t1[train_idx], X_act_t1[val_idx]
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
            {'position': X_pos_train, 'rotation': X_rot_train, 'actuation_t1': X_act_t1_train},
            Y_train,
            validation_data=(
                {'position': X_pos_val, 'rotation': X_rot_val, 'actuation_t1': X_act_t1_val},
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
