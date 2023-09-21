import pandas as pd
import numpy as np

# Assuming the data is stored in a CSV file called "keypoints.csv"
# data = pd.read_csv("/home/jc-merlab/Pictures/Data/kp_prior_poses/gt_poses/dream_gt_pred_comp.csv")
data = pd.read_csv("/home/jc-merlab/Pictures/panda_data/lama_gt_pred_comp.csv")

print(data.columns)

# Compute the euclidean distances for each keypoint
data['distance'] = np.sqrt((data['x_gt'] - data['x_pred'])**2 + (data['y_gt'] - data['y_pred'])**2)

# Get the mean distance across all keypoints
mean_distance = data['distance'].mean()

print(f"Mean Euclidean Distance: {mean_distance:.2f} pixels")

grouped = data.groupby('pose')
mean_distances_per_pose = grouped['distance'].mean()

print(mean_distances_per_pose)

grouped = data.groupby('pose')
mean_distances_per_pose = grouped['distance'].mean()

print(mean_distances_per_pose)

# Set a threshold value, for instance, 5 pixels
T = 5

# Compute the accuracy2.19
accuracy = (data['distance'] <= T).mean() * 100

print(f"Accuracy (within {T} pixels): {accuracy:.2f}%")