import numpy as np
import csv
from scipy.spatial.transform import Rotation as R

def generate_synthetic_dataset(num_points_per_motor=50):
    """
    Generate synthetic dataset for a 5-segment soft robotic arm with 2 motor actuations.
    Outputs end-effector position (x,y,z) and orientation (quaternion qx,qy,qz,qw).
    
    Assumptions:
    - 5 segments, with bending increasing from proximal (least) to distal (most)
    - 2 motors actuate 4 strings in a "+" configuration (diametrically opposite pairs)
    - act1 controls bending in XZ plane; act2 controls bending in YZ plane
    - Each actuation value ranges between 0 (no bend) and 1 (max bend)
    """
    
    # Parameters
    num_segments = 5
    max_bend_per_motor = np.pi / 6  # max bend angle per motor per segment (30 degrees)
    segment_length = 0.05  # 5 cm per segment
    
    # Generate motor actuation sweep [0..1]
    act1_vals = np.linspace(0, 1, num_points_per_motor)
    act2_vals = np.linspace(0, 1, num_points_per_motor)
    
    dataset = []
    
    for a1 in act1_vals:
        for a2 in act2_vals:
            # Compute bending angles per segment with distal segments bending more
            # Weight bending: proximal segment bends least, last segment bends most
            weights = np.linspace(0.2, 1.0, num_segments)  # increasing weights
            
            # Bend angles per segment for each motor
            bend_angles_motor1 = a1 * max_bend_per_motor * weights  # XZ plane bending
            bend_angles_motor2 = a2 * max_bend_per_motor * weights  # YZ plane bending
            
            # Initialize position and orientation
            pos = np.array([0.0, 0.0, 0.0])
            rot = R.from_quat([0, 0, 0, 1])  # identity quaternion
            
            # For each segment, apply bending from motor1 and motor2
            for i in range(num_segments):
                # Rotation around Y axis by motor1 bending (left-right bend)
                rot_motor1 = R.from_euler('y', bend_angles_motor1[i])
                # Rotation around X axis by motor2 bending (front-back bend)
                rot_motor2 = R.from_euler('x', bend_angles_motor2[i])
                
                # Combined rotation for this segment
                segment_rot = rot_motor1 * rot_motor2
                
                # Update cumulative rotation
                rot = rot * segment_rot
                
                # Update position: move forward along local Z by segment_length
                step = rot.apply([0, 0, segment_length])
                pos += step
            
            # Extract quaternion
            qx, qy, qz, qw = rot.as_quat()  # scipy returns [x,y,z,w]
            
            # Append data: act1, act2, x, y, z, qx, qy, qz, qw
            dataset.append([a1, a2, pos[0], pos[1], pos[2], qx, qy, qz, qw])
    
    return dataset

def save_dataset_to_csv(dataset, filename='robot_data.csv'):
    header = ['act1', 'act2', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dataset)
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    data = generate_synthetic_dataset(num_points_per_motor=50)  # 2500 samples
    save_dataset_to_csv(data)
