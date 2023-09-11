import os
import cv2
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = '/home/jc-merlab/lama/predict_data/2023-09-09/trained_models/keypointsrcnn_lama_b4_e25_v1.pth'

model = torch.load(weights_path).to(device)
model.eval()

image_folder_path = '/home/jc-merlab/lama/train_data/dataset/train/'
output_folder_path = "/home/jc-merlab/Pictures/Data/panda_lama_output/"

image_files = sorted(os.listdir(image_folder_path))
# image_files.sort()

def visualize(image, bboxes, keypoints, labels):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert image from RGB to BGR format
    for kps, label in zip(keypoints, labels):
        cv2.circle(image, tuple(kps), 5, (0, 0, 255), -1)
        cv2.putText(image, str(label), tuple(kps), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    return image

for i, img_file in enumerate(image_files):
    print(i)
    frame = cv2.imread(os.path.join(image_folder_path, img_file))
    image_pil = Image.fromarray(frame)

    image_tensor = F.to_tensor(image_pil).to(device)
    image_list = [image_tensor]

    with torch.no_grad():
        output = model(image_list)

    image_np = (image_list[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    scores = output[0]['scores'].detach().cpu().numpy()

    high_scores_idxs = np.where(scores > 0.7)[0].tolist()
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

    keypoints = [list(map(int, kps[0,0:2])) for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()]
    labels = [label for label in output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()]

    # Visualize the keypoints on the image
    img = visualize(image_np, None, keypoints, labels)

    cv2.imwrite(os.path.join(output_folder_path, img_file), img)

# Create a video from the saved images
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(output_folder_path, 'output_video_1.avi'), fourcc, 5.0, (img.shape[1], img.shape[0]))

output_images = [img for img in sorted(os.listdir(output_folder_path)) if not img.endswith(".avi")]

for image_file in output_images:
    image_path = os.path.join(output_folder_path, image_file)
    img = cv2.imread(image_path)
    out.write(img)
    # Deleting the image after it has been written to the video
    os.remove(image_path)

out.release()