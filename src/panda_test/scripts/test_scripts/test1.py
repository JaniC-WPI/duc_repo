import pandas as pd
from PIL import Image, ImageDraw

# Load the CSV data
data = pd.read_csv('/home/jc-merlab/Pictures/panda_data/aruco_pose/aruco_centers.csv')  # Replace 'your_csv_file.csv' with the path to your CSV file

print(data)

# Loop through each unique pose in the dataframe
for pose in data['pose'].unique():
    # Filter the dataframe for the current pose
    pose_data = data[data['pose'] == pose]
    
    # Construct the image file name based on the pose number
    # This assumes your images are named '000001.jpg', '000002.jpg', etc.
    image_file = f'/home/jc-merlab/Pictures/panda_data/aruco_pose/{pose:06d}.jpg'  # This formats the pose number as a six-digit string
    
    # Open the image
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    
    # Draw each point for the current pose
    for _, row in pose_data.iterrows():
        # The points are drawn as small circles for visibility
        # You might need to adjust the circle radius or use a different shape based on your needs
        draw.ellipse((row['x'] - 3, row['y'] - 3, row['x'] + 3, row['y'] + 3), fill='red')
    
    # Save the modified image
    # You might want to save this in a separate directory to keep the original images unchanged
    img.save(f'/home/jc-merlab/Pictures/panda_data/aruco_pose/modified_{pose:06d}.jpg')  # Prepends 'modified_' to the original file name for differentiation