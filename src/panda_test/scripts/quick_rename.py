import os

# Path to the directory containing the images
directory = "/home/jc-merlab/Pictures/Dl_Exps/lama_vs/servoing/exps_for_video/2/raw"

# Loop over all the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        # Extract the number from the filename
        number = int(filename.split(".")[0])

        # Create the new filename with the format 000001.png, 000002.png, etc.
        new_filename = f"{number+1:06}.png"

        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))