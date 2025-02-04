from PIL import Image
import numpy as np

# Load the newly uploaded images
image1_three_path = '/mnt/data/frame_0074.png'
image2_three_path = '/mnt/data/sim_published_goal_image.jpg'
image3_three_path = '/mnt/data/frame_1262.png'

image1_three = Image.open(image1_three_path).convert("RGBA")
image2_three = Image.open(image2_three_path).convert("RGBA")
image3_three = Image.open(image3_three_path).convert("RGBA")

# Convert images to numpy arrays for manipulation
image1_three_array = np.array(image1_three, dtype=float)
image2_three_array = np.array(image2_three, dtype=float)
image3_three_array = np.array(image3_three, dtype=float)

# Define the alpha values for blending
alpha_image1_three = 0.9
alpha_image2_three = 1.5
alpha_image3_three = 1.5

# Blend the first two images
blended_array_step1 = (image1_three_array * alpha_image1_three + image2_three_array * alpha_image2_three) / (alpha_image1_three + alpha_image2_three)

# Blend the result with the third image
blended_array_final = (blended_array_step1 + image3_three_array * alpha_image3_three) / (1 + alpha_image3_three)
blended_array_final = np.clip(blended_array_final, 0, 255).astype('uint8')  # Ensure valid range

# Convert back to an image
blended_image_three = Image.fromarray(blended_array_final.astype('uint8'), "RGBA")

# Save the result
three_image_output_path = "/mnt/data/final_blended_three_images.png"
blended_image_three.save(three_image_output_path)
three_image_output_path