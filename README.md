# Flexi-Robot: Neural Network Kinematics for Tendon-Driven Soft Robotic Manipulator

Forward and Inverse Kinematics modeling of a 5-segment tendon-driven soft robotic arm using neural networks.

## System

- **Robot:** 5-segment continuum arm actuated by 4 tendons in a "+" configuration, driven by 2 differential motors.
- **Data:** Qualisys motion capture (5 rigid body markers: `snake_tip`, `ST_A`, `ST_B`, `ST_C`, `ST_D`).
- **Models:** Forward Kinematics (`act1, act2` → `x, y, z, qx, qy, qz, qw`) and Inverse Kinematics (reverse mapping).

## Repository Structure

The codebase has been reorganized to separate the mathematically correct, production-ready pipeline from the legacy experimental scripts.

### `/correct_pipeline/` (Active Development)
Contains the mathematically rigorous, dual-head neural network pipeline:
- `DataPipeline_Fixed.py`: Derives exact tendon actuations using base-marker displacement and performs SLERP data augmentation.
- `Train_ForwardNN_v2.py`: Dual-head NN with quaternion-aware geometric loss function.
- `Train_InverseNN_v2.py`: Properly structured inverse kinematics model.
- `Evaluate.py`: Computes rigorous spatial metrics (mm position error, geodesic angular error).

### `/raw_mocap_data/`
- Original unmodified `.csv` datasets from the Qualisys system (`Case1.csv` through `Case8.csv`).

### `/legacy_scripts/`
- Old preprocessing, analytical constant-curvature (`Arm_Kinematics.py`), and LMA optimizer (`LMA_weights*.py`) scripts. Kept for reference.

### `/legacy_data_and_models/`
- Generated artifacts, `.h5` models, and Excel sheets produced by the legacy scripts.


