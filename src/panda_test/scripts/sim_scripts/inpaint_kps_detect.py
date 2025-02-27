
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision.transforms import functional as F
from torchvision import models
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from att_unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_inpaint_model(inpaint_path):
    inpaint_model = UNet().to(device)
    inpaint_model.load_state_dict(torch.load(inpaint_path, map_location=device))
    inpaint_model.eval()
    return inpaint_model

def load_kps_models(keypoint_path, model):
    keypoint_model = model.to(device)
    keypoint_model = torch.load(keypoint_path, map_location=device)
    keypoint_model.eval()
    return keypoint_model

def get_model(num_keypoints, weights_path=None):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=False,
                                                                   weights_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 7, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)
    return model

def denormalize(tensor, means, stds):
    means = torch.tensor(means).view(-1, 1, 1)
    stds = torch.tensor(stds).view(-1, 1, 1)

    if tensor.is_cuda:
        means = means.to(tensor.device)
        stds = stds.to(tensor.device)
    tensor = tensor * stds + means

    return tensor

def load_and_preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    preprocess = T.Compose([
        T.Resize((480, 640)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.209555113170385, 0.22507974363977162, 0.20982026500023962],
            std=[0.20639409678896012, 0.19208633033458372, 0.20659148273508857]
        )
    ])
    return preprocess(image).unsqueeze(0).to(device)

def predict_keypoints(frame, keypoint_model):
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    if frame.max() > 1.0:
        frame = frame / 255.0

    if frame.dim() == 4:
        frame = frame.squeeze(0)
    frame = frame.to(device)
    
    with torch.no_grad():
        outputs = keypoint_model([frame]) 
       
        kps_per_frame = []
        kp_nums = []
        seen_keypoints = set()
        
        for i in range(len(outputs[0]['keypoints'])):
            kp = outputs[0]['keypoints'][i]
            kp_score = outputs[0]['keypoints_scores'][i]
            kp_num = outputs[0]['labels'][i].item()

            if kp_num not in seen_keypoints:
                seen_keypoints.add(kp_num)
                max_score = kp_score.max()
                kp_score_index = torch.where(kp_score == max_score)[0]
                final_kps = kp[kp_score_index]
                kps_per_frame.append(final_kps.cpu().numpy())
                kp_nums.append(kp_num)
                
        kps_per_frame = np.array(kps_per_frame).squeeze(1)
        
    return kps_per_frame, kp_nums

def draw_keypoints(image, keypoints, nums, color=(0, 255, 0), radius=5, thickness=-1):
    for i, (kp, num) in enumerate(zip(keypoints, nums)):
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius, color, thickness)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text = str(num)  #.cpu().numpy()
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        text_x = x + 5
        text_y = y - 5
        
        cv2.rectangle(image, (text_x, text_y - text_size[1]), 
                      (text_x + text_size[0], text_y + 5), 
                      (0, 0, 0), -1)
        
        cv2.putText(image, text, (text_x, text_y), 
                    font, font_scale, (255, 255, 255), font_thickness)
    
    return image

def process_video(input_path, output_path, inpaint_model, keypoint_model, batch_size=1):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames) as pbar:
        while True:
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if not frames:
                break

            input_tensors = load_and_preprocess_frame(frame)
            with torch.no_grad():
                output_image_inpainted = inpaint_model(input_tensors)
            output_image_inpainted = denormalize(output_image_inpainted, 
                                       [0.209555113170385, 0.22507974363977162, 0.20982026500023962],
                                       [0.20639409678896012, 0.19208633033458372, 0.20659148273508857])

            out_frame = (output_image_inpainted).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            out_frame = np.clip(out_frame, 0, 1)
            out_frame = (out_frame * 255).astype(np.uint8)
            out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            output, nums = predict_keypoints(out_frame, keypoint_model)   
            # frame_kps = draw_keypoints(out_frame, output, nums)
            frame_kps = draw_keypoints(frame, output, nums)

            
            out.write((frame_kps).astype(np.uint8)) 
            pbar.update(1)


    cap.release()
    out.release()

def main():
    print("Starting the inpainting and keypoint detection process...")
    print("Base Directory =", BASE_DIR)
    # inpaint_path = os.path.join(BASE_DIR, 'trained_models/generator.pth') 
    inpaint_path = '/home/jc-merlab/Pictures/Data/trained_models/generator.pth'  
    #'/home/venk/Downloads/inpainting/inpainting/Saved_models/turing_models_checkpts/Sep_6/generator.pth'
    # input_video = os.path.join(BASE_DIR, 'input_videos/ycb_test_01.avi')  
    input_video = '/home/jc-merlab/Pictures/Test_Data/occ_vids/exp_01/pred/pred.avi'
    # output_video = os.path.join(BASE_DIR, 'output_videos/output01.avi')
    output_video = '/home/jc-merlab/Pictures/Test_Data/occ_vids/exp_01/pred/output_video_exp_01.avi'
    # keypoint_path = os.path.join(BASE_DIR, 'trained_models/keypointsrcnn_planning_b1_e50_v8.pth')
    keypoint_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'

    inpaint_model = load_inpaint_model(inpaint_path)
    model = get_model(num_keypoints = 9, weights_path=keypoint_path)
    keypoint_model = load_kps_models(keypoint_path, model)
    process_video(input_video, output_video, inpaint_model, keypoint_model, batch_size=1)

if __name__ == "__main__":
    main()