import os
import cv2
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import csv

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# weights_path = '/home/jc-merlab/Pictures/panda_data/trained_models/keypointsrcnn_lama_b4_e25_v1.pth'

# model = torch.load(weights_path).to(device)
# model.eval()

# image_folder_path = '/home/jc-merlab/Pictures/panda_data/kp_poses/'
# output_folder_path = "/home/jc-merlab/Pictures/panda_data/kp_poses/kp_pose_op/"

# image_files = sorted(os.listdir(image_folder_path))
# # image_files.sort()

# def save_keypoints_to_csv(csv_filename, img_id, keypoints, labels):
#     with open(csv_filename, 'a', newline='') as file:
#         writer = csv.writer(file)
#         for label, kps in sorted(zip(labels, keypoints), key=lambda x: x[0]):
#             writer.writerow([img_id, label, kps[0], kps[1]])

# def visualize(image, bboxes, keypoints, labels):
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert image from RGB to BGR format
#     for kps, label in zip(keypoints, labels):
#         cv2.circle(image, tuple(kps), 5, (0, 0, 255), -1)
#         cv2.putText(image, str(label), tuple(kps), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
#     return image

# # Specify path to your csv file
# csv_path = '/home/jc-merlab/Pictures/panda_data/kp_poses/keypoints.csv'

# # Create the CSV file and write headers
# with open(csv_path, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['image_id', 'label', 'x', 'y'])

# for i, img_file in enumerate(image_files):
#     print(i)
#     frame = cv2.imread(os.path.join(image_folder_path, img_file))
#     image_pil = Image.fromarray(frame)

#     image_tensor = F.to_tensor(image_pil).to(device)
#     image_list = [image_tensor]

#     with torch.no_grad():
#         output = model(image_list)

#     image_np = (image_list[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
#     scores = output[0]['scores'].detach().cpu().numpy()

#     high_scores_idxs = np.where(scores > 0.7)[0].tolist()
#     post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

#     keypoints = [list(map(int, kps[0,0:2])) for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()]
#     labels = [label for label in output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()]

#     # Visualize the keypoints on the image
#     img = visualize(image_np, None, keypoints, labels)

#     cv2.imwrite(os.path.join(output_folder_path, img_file), img)





# # Create a video from the saved images
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(os.path.join(output_folder_path, 'output_video.avi'), fourcc, 5.0, (img.shape[1], img.shape[0]))

# output_images = [img for img in sorted(os.listdir(output_folder_path)) if not img.endswith(".avi")]

# for image_file in output_images:
#     image_path = os.path.join(output_folder_path, image_file)
#     img = cv2.imread(image_path)
#     out.write(img)
#     # Deleting the image after it has been written to the video
#     # os.remove(image_path)

# out.release()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = '/home/jc-merlab/Pictures/panda_data/trained_models/keypointsrcnn_ur_lama_b4_e100_v1.pth'
model = torch.load(weights_path).to(device)
model.eval()

image_folder_path = '/home/jc-merlab/Pictures/panda_data/ur10/ur_pred/'
# image_folder_path = '/home/jc-merlab/Pictures/Dl_Exps/lama_vs/servoing/exps_latest_2023_09_14/15/raw'
# aruco_op_folder_path = "/home/jc-merlab/Pictures/panda_data/kp_poses/aruco_poses_kp/"
output_folder_path = "/home/jc-merlab/Pictures/panda_data/ur10/ur_pred/ur_pred_op/"
# output_folder_path = '/home/jc-merlab/Pictures/Dl_Exps/lama_vs/servoing/exps_latest_2023_09_14/15/img_op/'
csv_path = '/home/jc-merlab/Pictures/panda_data/ur10/ur_pred/ur_pred_op/ur_keypoints.csv'
# csv_path = '/home/jc-merlab/Pictures/Dl_Exps/lama_vs/servoing/exps_latest_2023_09_14/15/keypoints.csv'
# aruco_folder_path = '/home/jc-merlab/Pictures/panda_data/aruco_pose/'

image_files = sorted(os.listdir(image_folder_path))
# aruco_files = sorted(os.listdir(aruco_folder_path))

# Prediction and Saving keypoints to CSV
def save_keypoints_to_csv(data, csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'label', 'x', 'y'])
        writer.writerows(data)

keypoint_data = []

for i, img_file in enumerate(image_files):
    img_path = os.path.join(image_folder_path, img_file)

    # Check if the constructed image path is valid
    if not os.path.exists(img_path):
        print(f"Image {img_path} does not exist.")
        continue

    frame = cv2.imread(img_path)

    # Check if the image has been read correctly
    if frame is None:
        print(f"Failed to read image at {img_path}. Continuing with next image.")
        continue

    image_pil = Image.fromarray(frame)

    image_tensor = F.to_tensor(image_pil).to(device)
    image_list = [image_tensor]

    with torch.no_grad():
        output = model(image_list)

    print(output)
    scores = output[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > 0.15)[0].tolist()
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

    keypoints = [list(map(int, kps[0,0:2])) for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()]
    labels = [label for label in output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()]

    # Sort keypoints and labels based on label values
    sorted_kps_labels = sorted(zip(keypoints, labels), key=lambda x: x[1])

    # Appending sorted keypoints to keypoint_data
    for kps, label in sorted_kps_labels:
        keypoint_data.append([os.path.splitext(img_file)[0], label, kps[0], kps[1]])

# Save keypoints data to CSV
save_keypoints_to_csv(keypoint_data, csv_path)


# Visualization using keypoints from CSV
def get_keypoints_from_csv(csv_filename, img_id):
    keypoints = []
    labels = []
    with open(csv_filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            if row[0] == img_id:
                labels.append(int(row[1]))
                keypoints.append([int(row[2]), int(row[3])])
    return keypoints, labels

def visualize(image, keypoints, labels):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for kps, label in zip(keypoints, labels):
        cv2.circle(image, tuple(kps), 5, (0,97,230), -1)
        # cv2.putText(image, str(label), tuple(kps), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 4, cv2.LINE_AA)
    return image

for i, img_file in enumerate(image_files):
    frame = cv2.imread(os.path.join(image_folder_path, img_file))

    if frame is None:
        print(f"Failed to read image at {img_path}. Continuing with next image.")
        continue

    img_id = os.path.splitext(img_file)[0]
    keypoints, labels = get_keypoints_from_csv(csv_path, img_id)

    img = visualize(frame, keypoints, labels)

    cv2.imwrite(os.path.join(output_folder_path, img_file), img)

# for i, img_file in enumerate(aruco_files):
#     frame = cv2.imread(os.path.join(aruco_folder_path, img_file))

#     if frame is None:
#         print(f"Failed to read image at {img_path}. Continuing with next image.")
#         continue

#     img_id = os.path.splitext(img_file)[0]
#     keypoints, labels = get_keypoints_from_csv(csv_path, img_id)

#     img = visualize(frame, keypoints, labels)

#     cv2.imwrite(os.path.join(aruco_op_folder_path, img_file), img)

# Create a video from the saved images
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(output_folder_path, 'output_video.avi'), fourcc, 30.0, (img.shape[1], img.shape[0]))

output_images = [img for img in sorted(os.listdir(output_folder_path)) if not img.endswith(".avi")]

for image_file in output_images:
    image_path = os.path.join(output_folder_path, image_file)
    img = cv2.imread(image_path)
    out.write(img)

out.release()