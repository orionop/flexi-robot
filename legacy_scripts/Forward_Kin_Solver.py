import numpy as np
import csv
from scipy.spatial.transform import Rotation as R

def forward_kinematics(act1, act2, num_segments=5, segment_length=0.05, max_bend_per_motor=np.pi/6):
    weights = np.linspace(0.2, 1.0, num_segments)
    bend_angles_motor1 = act1 * max_bend_per_motor * weights
    bend_angles_motor2 = act2 * max_bend_per_motor * weights

    pos = np.array([0.0, 0.0, 0.0])
    rot = R.from_quat([0, 0, 0, 1])  

    for i in range(num_segments):
        rot_motor1 = R.from_euler('y', bend_angles_motor1[i])
        rot_motor2 = R.from_euler('x', bend_angles_motor2[i])
        segment_rot = rot_motor1 * rot_motor2
        rot = rot * segment_rot
        step = rot.apply([0, 0, segment_length])
        pos += step
        print(segment_rot.as_quat())
        print(rot.as_quat())
        print(step)
        print(pos)

    qx, qy, qz, qw = rot.as_quat()
    return pos[0], pos[1], pos[2], qx, qy, qz, qw

def export_to_csv(actuation_inputs, filename='C:/Users/Aaditya/Desktop/SysCon 2025/Tendon Driven Model/Neural Network/fk_outputs.csv'):
    header = ['act1', 'act2', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    rows = []

    for act1, act2 in actuation_inputs:
        x, y, z, qx, qy, qz, qw = forward_kinematics(act1, act2)
        rows.append([act1, act2, x, y, z, qx, qy, qz, qw])

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"CSV file has been saved...")

if __name__ == "__main__":
    test_inputs = [
        # (0.0, 0.0),
        # (0.5, 0.5),
        (1.0, 0.0),
        # (0.0, 1.0),
        # (1.0, 1.0)
        #(0.65,0.35)
    ]

    export_to_csv(test_inputs)
