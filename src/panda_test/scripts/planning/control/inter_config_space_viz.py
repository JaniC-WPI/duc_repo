import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from Robot import RobotTest, PandaReal2D
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

jt_file_paths_template = [
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/no_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/no_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/no_obs/{}/save_distances.csv'                                                            
]

# Function to read keypoints and joints from a CSV file
def read_keypoints_and_joints(file_path):
    """Reads the keypoints and joints from a CSV file."""
    df = pd.read_csv(file_path)

    # Extract keypoints
    x_keypoints = df.iloc[:, 1:18:2].to_numpy()  # Odd columns
    y_keypoints = df.iloc[:, 2:19:2].to_numpy()  # Even columns

    # print("X kp", x_keypoints)
    # print("Y kp", y_keypoints)

    x_kp_indexed = x_keypoints[:, [1, 3, 4, 6, 7, 8]]
    y_kp_indexed = y_keypoints[:, [1, 3, 4, 6, 7, 8]]

    # print("X kp indexed", x_kp_indexed)
    # print("Y kp indexed", y_kp_indexed)

    # Select specific keypoints (new indices: 0, 3, 4, 5, 6, 7)
    selected_keypoints = np.dstack((x_kp_indexed, y_kp_indexed)).reshape(-1,6,2)

    # print(selected_keypoints.shape)

    # Extract joint angles
    joints = df[["Joint 1", "Joint 2", "Joint 3"]].to_numpy()

    return selected_keypoints, joints

def create_jt_matrix(joints):
    joint_positions = [0.007195404887023141, 0, -0.008532170082858044, 0, 0.0010219530727038648, 0, 0.8118303423692146]
    joint_positions[1] = joints[0]
    joint_positions[3] = joints[1]
    joint_positions[5] = joints[2]

    return joint_positions

def compute_distance(config1, config2):
    """Compute the Euclidean distance between two configurations."""
    return sum(np.linalg.norm(np.array(kp1) - np.array(kp2)) for kp1, kp2 in zip(config1, config2))

# Define the DH transformation matrix
def dh_transformation(a, d, alpha, theta):
    """Compute the DH transformation matrix."""
    return np.array([
        [math.cos(theta), -math.sin(theta),  0, a],
        [math.sin(theta) * math.cos(alpha), math.cos(theta) * math.cos(alpha), -math.sin(alpha), -d * math.sin(alpha)],
        [math.sin(theta) * math.sin(alpha), math.cos(theta) * math.sin(alpha), math.cos(alpha), d * math.cos(alpha)],
        [0, 0, 0, 1]
    ])

# Define the forward kinematics function
def forward_kinematics(joint_angles):
    """Compute the forward kinematics for the Franka Emika Panda Arm."""
    # DH Parameters
    DH_params_home = [
        [0,      0.333,             0,   0],
        [0,          0,  -(math.pi/2),   0],
        [0,      0.316,   (math.pi/2),   0],
        [0.0825,     0,   (math.pi/2),   0],
        [-0.0825, 0.384,  -(math.pi/2),  0],
        [0,          0,   (math.pi/2),   0],
        [0.088,      0,   (math.pi/2),   0]
    ]
    
    # Initialize the transformation matrix
    T = np.eye(4)
    transformations = []

    # Compute the total transformation matrix
    for i, (a, d, alpha, _) in enumerate(DH_params_home):
        theta = joint_angles[i]
        T_i = dh_transformation(a, d, alpha, theta)
        T = np.dot(T, T_i)
        transformations.append(T)

    return T, transformations  # Return the final transformation and all intermediate ones

def transform(tvec, quat):

    r = R.from_quat(quat).as_matrix()

    ht = np.array([[r[0][0], r[0][1], r[0][2], tvec[0]],
                   [r[1][0], r[1][1], r[1][2], tvec[1]],
                   [r[2][0], r[2][1], r[2][2], tvec[2]],
                   [0, 0, 0, 1]])

    return ht

def camera_intrinsics(K):
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]
    camera_K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    
    return camera_K

def remove_duplicates(points, precision=5):
    unique_points = []
    seen = set()
    for x, y in points:
        rounded_point = (round(x, precision), round(y, precision))
        if rounded_point not in seen:
            seen.add(rounded_point)
            unique_points.append([x, y])
    return unique_points

def camera_extrinsics():
    camera_ext_trans = [-0.18, 0.49,  1.55]#[-0.1406737, 0.51277635, 1.82787528]
    camera_ext_rot =  [0.66974827,  0.03, -0.03,  0.74] #[0.69402915, -0.17008132,  0.15281551,  0.68267365]
    camera_ext = transform(camera_ext_trans, camera_ext_rot)

    return camera_ext

def world_coords_tf(joints):
        """
        Calculates positions of frames in world frame using FK.
        """
        T, transformations = forward_kinematics(joints)

        tmp = np.transpose([0, 0, 0, 1])
        # Using local variable to prevent shared data bug
        world_coords = [np.dot(np.eye(4), tmp)]
        world_coords.extend([np.dot(T, tmp) for T in transformations])

        return world_coords

def image_pixels(camera_K, camera_ext, world_coords):
        """
        Calculates pixel coordinates of the keypoints given extrinsics and
        keypoints' positions in world frame.
        """
        proj_model = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        # make intrinsics matrix 4x4
        camera_K_4 = np.dot(camera_K, proj_model)
        # raw (unscaled) keypoint pixels
        image_pixs = [
            np.dot(camera_K_4, np.dot(camera_ext, world_coords[i]))
            for i in range(len(world_coords))]
        
        # print("Image PIxels before keypoints", [image_pixs[i] for i in range(len(world_coords))])
        # scale the pixel coordinates
        img_pixels = [
            [
                image_pixs[i][0]/image_pixs[i][2],  # u
                image_pixs[i][1]/image_pixs[i][2],  # v
         ] for i in range(len(world_coords))]
        return img_pixels

def extend_and_drop_perpendicular(point_a, point_b):
    """
    Computes a perpendicular from point_b (6th keypoint) using the direction and length of the vector from point_a to point_b.
    The perpendicular will be of the same length as the vector from point_a to point_b.
    """
    vector = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
    length = np.linalg.norm(vector)
    perpendicular_vector = np.array([-vector[1], vector[0]])  # Rotate 90 degrees to get the perpendicular
    final_point = np.array([point_b[0], point_b[1]]) + perpendicular_vector / np.linalg.norm(perpendicular_vector) * (length * 1.25)
    return [final_point[0], final_point[1]] 

def update_keypoints(keypoints):
    # Extend and drop perpendicular from 5th to 6th keypoints
    point_c = keypoints[4]
    point_d = keypoints[5]
    new_keypoint = extend_and_drop_perpendicular(point_c, point_d)
    keypoints.append(new_keypoint)

    return keypoints

def interpolate_joints(joint_start, joint_end, num_points):
    """Interpolates joint angles between two configurations."""
    return np.linspace(joint_start, joint_end, num_points)

def interpolate_keypoints(kp_start, kp_end, num_points):
    kp_start = np.array(kp_start)
    kp_end = np.array(kp_end)

    actual_interp_kp = np.linspace(kp_start, kp_end, num_points, axis=0)

    return actual_interp_kp


def calculate_deviations(interpolated_keypoints, actual_keypoints):
    """Calculate deviations between actual and interpolated keypoints."""
    deviations = []
    deviations_per_keypoint = {i: [] for i in range(1, 6)}

    for act_kp, interp_kp in zip(actual_keypoints, interpolated_keypoints):
        for i in range(1, 6):  # Keypoints from index 1 to 5
            actual_point = np.array(act_kp[i])
            interpolated_point = np.array(interp_kp[i])
            manhattan_dist = np.sum(np.abs(actual_point - interpolated_point))
            # manhattan_dist = np.linalg.norm(np.array(actual_point) - np.array(interpolated_point))
            deviations_per_keypoint[i].append(manhattan_dist)
            deviations.append(manhattan_dist)

    return deviations, deviations_per_keypoint

def visualize_interpolated_keypoints(original_keypoints, interpolated_keypoints, actual_interpolated_keypoints):
    """Visualizes the original and interpolated keypoints."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for i, og_config in enumerate(original_keypoints):
        # Plot original keypoints
        ax.scatter(og_config[:, 0], -og_config[:, 1], color='red', label='Original Keypoints', s=10)
        # Draw lines between original keypoints
        ax.plot(og_config[:, 0], -og_config[:, 1], color='red', linestyle='-', linewidth=1, label='Original Keypoint Chain')

    # Plot interpolated keypoints
    for i, ip_config in enumerate(interpolated_keypoints):
        ip_config = np.array(ip_config)
        ax.scatter(ip_config[:, 0], -ip_config[:, 1], color='blue', label='Interpolated Keypoints in Config Space', s=2)
        # Label the keypoints in order
        # for idx, (x, y) in enumerate(ip_config):
        #     ax.text(
        #         x, -y, f'{idx}', color='black', fontsize=8, ha='center', va='center'
        #     )

    for i, act_ip_config in enumerate(actual_interpolated_keypoints):
        act_ip_config = np.array(act_ip_config)
        ax.scatter(act_ip_config[:, 0], -act_ip_config[:, 1], color='green', label='Actual Interpolated Keypoints', s=1)
        # Label the keypoints in order
        # for idx, (x, y) in enumerate(act_ip_config):
        #     ax.text(
        #         x, -y, f'{idx}', color='black', fontsize=8, ha='center', va='center'
        #     )
    
    # Labels and legend
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Visualization of Interpolated Keypoints', fontsize=14)
    # ax.legend()
    ax.set_xlim(0, 640)
    ax.set_ylim(-480, 0)
    plt.show()
    
labels = ["Ground Truth", "Learned", "Image Space"]
    
# exp_no = [1,2,3,4,6,8,9,10,13,14,15,16,17,18,19,20] # exp with no obs
# exp_no = [1, 2, 8, 9, 10, 14, 17, 18, 19, 20] # exps for with obs
exp_no = [17]

num_interpolations = 500
# Store deviation statistics
deviation_stats = []

for path_template, label in zip(jt_file_paths_template, labels):
    for exp in exp_no:
        file_path = path_template.format(exp)
        if not os.path.exists(file_path):
            continue

        # Read keypoints and joints
        keypoints, joints = read_keypoints_and_joints(file_path)

        interpolated_keypoints = []
        actual_interpolated_keypoints = []

        # Step 1: Compute distances between consecutive configurations
        distances = [compute_distance(keypoints[i], keypoints[i + 1]) for i in range(len(keypoints) - 1)]
        total_distance = sum(distances)

        # Step 2: Compute relative weights
        weights = [dist / total_distance for dist in distances]

        # Step 3: Distribute interpolation points
        interpolation_points = [int(num_interpolations * weight) for weight in weights]

        # Step 4: Adjust for rounding
        interpolation_points[-1] += num_interpolations - sum(interpolation_points)

        for i, num_points in enumerate(interpolation_points):
            print("actual_number_of_interpolation between config pairs", num_points)
            segment = interpolate_joints(joints[i], joints[i + 1], num_points)
            for j, interp_joint in enumerate(segment):            
                joint_angles = create_jt_matrix(interp_joint)
                three_d_proj = world_coords_tf(joint_angles)
                camera_ext = camera_extrinsics()
                K = [605.2972412109375, 0.0, 320.8614807128906, 0.0, 604.324951171875, 251.25967407226562, 0.0, 0.0, 1.0]
                camera_K = camera_intrinsics(K)
                initial_keypoints = image_pixels(camera_K, camera_ext,three_d_proj)
                unique_kp = remove_duplicates(initial_keypoints)
                updated_kp = update_keypoints(unique_kp)
                updated_kp = np.array(updated_kp)[1:, :]

                # Set the first and last points to match the start and end configs
                if j == 0:
                    updated_kp = keypoints[i]  # First point matches start config
                elif j == len(segment) - 1:
                    updated_kp = keypoints[i + 1]  # Last point matches next config


                interpolated_keypoints.append(updated_kp)

            actual_interp_kp = interpolate_keypoints(keypoints[i], keypoints[i+1], num_points)
            # print("interpolation between configs", len(actual_interp_kp))
            actual_interpolated_keypoints.extend(actual_interp_kp)  

visualize_interpolated_keypoints(keypoints, interpolated_keypoints, actual_interpolated_keypoints)
        








