import os
import shutil

# Define paths
main_folder = '/media/jc-merlab/Crucial X9/control_experiments/exps/'
op_folder = '/media/jc-merlab/Crucial X9/Control_Experiments_RA_L_v3/'

# Experiment lists
experiments = {
    "joint_space": {
        "no_obs": ["1", "2", "3", "4", "6", "8", "9", "10", "13", "14", "15", "16", "17", "18", "19", "20"],
        "with_obs": ["1", "2", "8", "9", "10", "14", "17", "18", "19", "20"]
    },
    "learned": {
        "no_obs": ["1", "2", "3", "4", "6", "8", "9", "10", "13", "14", "15", "16", "17", "18", "19", "20"],
        "with_obs": ["1", "2", "8", "9", "10", "14", "17", "18", "19", "20"]
    },
    "image_space": {
        "no_obs": ["1", "2", "3", "4", "6", "8_a", "9", "10", "13", "14", "15", "16_a", "17_a", "18", "19", "20"],
        "with_obs": ["1_a", "2_a", "8_a", "9", "10", "14_a", "17", "18_a", "19", "20"]
    }
}

# Function to determine if an experiment is failed
def is_failed_experiment(exp_name):
    return exp_name.endswith("_a")

# Function to process experiments
def process_experiments(subfolder, obs_type):
    source_path = os.path.join(main_folder, subfolder, f"astar_latest_{obs_type}")
    exp_list = experiments[subfolder][obs_type]

    for exp in exp_list:
        exp_folder = os.path.join(source_path, exp)
        video_file = os.path.join(exp_folder, "exp_vid.avi")

        if os.path.exists(video_file):
            # Determine if the experiment is failed or successful
            if subfolder == "image_space":
                is_failed = is_failed_experiment(exp)
            else:
                # Check if the corresponding image_space experiment is failed
                corresponding_exp = f"{exp}_a"
                is_failed = corresponding_exp in experiments["image_space"][obs_type]

            destination = "Failed_Experiments" if is_failed else "Successful_Experiments"
            target_folder = os.path.join(op_folder, destination, exp)
            os.makedirs(target_folder, exist_ok=True)

            # Rename file according to its type
            renamed_file = f"{subfolder}_{'no_obstacle' if obs_type == 'no_obs' else 'with_obstacle'}.avi"
            target_path = os.path.join(target_folder, renamed_file)

            shutil.copy(video_file, target_path)
            print(f"Copied {video_file} to {target_path}")
        else:
            print(f"Video file not found: {video_file}")

# Process each subfolder and obstacle type
for sub in ["joint_space", "learned", "image_space"]:
    for obs in ["no_obs", "with_obs"]:
        process_experiments(sub, obs)