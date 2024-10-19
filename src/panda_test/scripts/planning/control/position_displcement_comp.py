
import os
import json
import numpy as np
import matplotlib.pyplot as plt


json_folder = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/regression_rearranged_all_double_corrected/'

# Prepare lists for storing axis-specific and overall magnitude data
# Prepare lists for storing axis-specific and overall magnitude data
position_displacement_j1 = []
position_displacement_j2 = []
position_displacement_j3 = []
actual_displacement_j1 = []
actual_displacement_j2 = []
actual_displacement_j3 = []
position_displacement_magnitude = []
actual_displacement_magnitude = []

# Function to calculate the Euclidean magnitude of a vector
def calculate_magnitude(vector):
    return sum([v**2 for v in vector]) ** 0.5

# Iterate through all files in the folder and collect data
for filename in os.listdir(json_folder):
    if filename.endswith('_combined.json'):
        file_path = os.path.join(json_folder, filename)

        # Open and load the JSON data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

            # Extract the position and actual joint displacement
            position = data["position"]  # [x, y, z]
            actual_joint_displacement = data["actual_joint_displacement"]  # [x, y, z]

            # Append axis-specific displacement data
            position_displacement_j1.append(position[0])
            position_displacement_j2.append(position[1])
            position_displacement_j3.append(position[2])

            actual_displacement_j1.append(actual_joint_displacement[0])
            actual_displacement_j2.append(actual_joint_displacement[1])
            actual_displacement_j3.append(actual_joint_displacement[2])

            # Append magnitude (overall displacement)
            position_displacement_magnitude.append(calculate_magnitude(position))
            actual_displacement_magnitude.append(calculate_magnitude(actual_joint_displacement))

# Convert lists to numpy arrays for easier manipulation
position_displacement_magnitude = np.array(position_displacement_magnitude)
actual_displacement_magnitude = np.array(actual_displacement_magnitude)

position_displacement_j1 = np.array(position_displacement_j1)
position_displacement_j2 = np.array(position_displacement_j2)
position_displacement_j3 = np.array(position_displacement_j3)

actual_displacement_j1 = np.array(actual_displacement_j1)
actual_displacement_j2 = np.array(actual_displacement_j2)
actual_displacement_j3 = np.array(actual_displacement_j3)

# Plot histograms for each axis with darker colors, side-by-side, and same bin width

def plot_histograms_side_by_side(data1, data2, label1, label2, xlabel, ylabel, title):
    # Calculate the bin edges to ensure same bin width
    min_value = min(min(data1), min(data2))
    max_value = max(max(data1), max(data2))
    bin_width = 0.1 # Define your bin width
    bins = np.arange(min_value, max_value, bin_width)  # Create bins with equal width

    # Calculate the histograms using np.histogram to get counts and bin edges
    counts1, bin_edges1 = np.histogram(data1, bins=bins)
    counts2, bin_edges2 = np.histogram(data2, bins=bins)

    # Find max bin count and corresponding range for each dataset
    max_count1 = np.max(counts1)
    max_bin_idx1 = np.argmax(counts1)
    range1 = (bin_edges1[max_bin_idx1], bin_edges1[max_bin_idx1 + 1])

    max_count2 = np.max(counts2)
    max_bin_idx2 = np.argmax(counts2)
    range2 = (bin_edges2[max_bin_idx2], bin_edges2[max_bin_idx2 + 1])

    range1_formatted = f"({range1[0]:.6f}, {range1[1]:.6f})"
    range2_formatted = f"({range2[0]:.6f}, {range2[1]:.6f})"

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot histograms
    # ax1.hist(data1, bins=bins, alpha=0.9 , label=f'{label1}\nMax Count: {max_count1}\nRange: {range1_formatted}', color='blue')  # Darker blue
    ax1.hist(data1, bins=bins, alpha=1.0, label='', color='blue')  # Darker blue
    # ax1.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    # ax1.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    # ax1.set_title(f'{label1} {title}', fontsize=14, fontweight='bold')
    # ax1.grid(True)
    # ax1.legend(prop={'size': 12, 'weight': 'bold'})

    # ax1.set_xticklabels([str(int(tick)) for tick in ax1.get_xticks()], fontsize=8, weight='bold')
    # ax1.set_yticklabels([str(int(tick)) for tick in ax1.get_yticks()], fontsize=10, weight='bold')

    # ax2.hist(data2, bins=bins, alpha=0.9, label=f'{label2}\nMax Count: {max_count2}\nRange: {range2_formatted}', color='green')  # Darker green
    ax2.hist(data2, bins=bins, alpha=1.0, label='', color='green')  # Darker green

    # ax2.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    # ax2.set_title(f'{label2} {title}', fontsize=14, fontweight='bold')
    # ax2.grid(True)
    # ax2.legend(prop={'size': 12, 'weight': 'bold'})

    # Make tick label values bold for both axes
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # Set tick label fonts to bold manually
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')

    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')

    plt.tight_layout()
    plt.show()

# Plot X-axis histograms (side by side)
plot_histograms_side_by_side(
    position_displacement_j1, actual_displacement_j1,
    'Estimated Joint 1 Displacement', 'Actual Joint 1 Displacement',
    'Joint 1 Displacement', 'Frequency', 'Comparison')

# Plot Y-axis histograms (side by side)
plot_histograms_side_by_side(
    position_displacement_j2, actual_displacement_j2,
    'Estimated Joint 2 Displacement', 'Actual Joint 2 Displacement',
    'Joint 2 Displacement', 'Frequency', 'Comparison')

# Plot Z-axis histograms (side by side)
plot_histograms_side_by_side(
    position_displacement_j3, actual_displacement_j3,
    'Estimated Joint 3 Displacement', 'Actual Joint 3 Displacement',
    'Joint 3 Displacement', 'Frequency', 'Comparison')

# Plot Overall displacement magnitude histograms (side by side)
plot_histograms_side_by_side(
    position_displacement_magnitude, actual_displacement_magnitude,
    'Estimated Joint Displacement Magnitude', 'Actual Joint Displacement Magnitude',
    'Displacement Magnitude', 'Frequency', 'Comparison')
