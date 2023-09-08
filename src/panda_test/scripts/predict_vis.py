import cv2
import glob
import torch
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import os

# Initialize model and load weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_weights_origami_b1_e30_v1.pth'
model = torch.load(weights_path).to(device)
model.eval()

# Create destination folder if doesn't exist
os.makedirs('/home/jc-merlab/Pictures/Data/origami_output', exist_ok=True)

# Loop through each image in the folder
image_paths = sorted(glob.glob("/home/jc-merlab/Pictures/origami_data/2023-09-01_00_37_51/*.jpg"))

for idx, image_path in enumerate(image_paths):
    # Load and preprocess image
    image = Image.open(image_path)
    image_tensor = F.to_tensor(image).to(device)
    image_list = [image_tensor]
    
    # Run the model
    with torch.no_grad():
        output = model(image_list)
        
    # Draw keypoints and bounding boxes
    image_np = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    scores = output[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > 0.7)[0].tolist()
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()


    keypoints = []
    key_points = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append(list(map(int, kps[0,0:2])))
    
    print("Keypoints", keypoints)

    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_copy = image_np.copy()

    for keypoints, bbox in zip(output[0]['keypoints'][high_scores_idxs][post_nms_idxs], output[0]['boxes'][high_scores_idxs][post_nms_idxs]):
        keypoints = keypoints.detach().cpu().numpy()
        print("Keypoints", keypoints)
        bbox = bbox.detach().cpu().numpy().astype(int)
        for kp in keypoints:
            x, y = kp[0:2].astype(int)
            cv2.circle(image_copy, (x, y), 3, (0, 255, 0), -1)
        cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    # Save the new image
    output_path = f"/home/jc-merlab/Pictures/Data/origami_output/{idx}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
    print(f"Saved visualized output to {output_path}")

# Create a video from the saved images
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (image_np.shape[1], image_np.shape[0]))

for idx in range(len(image_paths)):
    frame = cv2.imread(f"/home/jc-merlab/Pictures/Data/origami_output/{idx}.jpg")
    out.write(frame)

out.release()
print("Video created")