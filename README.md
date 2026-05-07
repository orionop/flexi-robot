# Flexi-Robot: Neural Network Kinematics for Tendon-Driven Soft Robotic Arm

Forward and Inverse Kinematics modeling of a 5-segment tendon-driven soft robotic arm using neural networks.

## System

- **Robot:** 5-segment continuum arm actuated by 4 tendons in a "+" configuration, driven by 2 motors
- **Data:** OptiTrack motion capture (5 rigid body markers: `snake_tip`, `ST_A`, `ST_B`, `ST_C`, `ST_D`)
- **Models:** FK (`act1, act2` → `x, y, z, qx, qy, qz, qw`) and IK (reverse)

## Structure

- `Case*.csv` — Raw motion capture data (8 experimental cases)
- `Case*Processed.py` — Data cleaning and preprocessing
- `DataSetProcessed1to8.py` — Dataset consolidation
- `Train_ForwardNN.py` — Forward kinematics NN training
- `Train_InverseNN.py` — Inverse kinematics NN training
- `Arm_Kinematics.py` — Analytical constant-curvature bending model
- `Predict*.py` — Inference scripts

## Dependencies

Python 3.x, TensorFlow, NumPy, Pandas, SciPy, scikit-learn, joblib
