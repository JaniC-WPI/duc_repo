#!/usr/bin/env python3

import cv2
from os.path import expanduser
# from PIL import Image

home = expanduser("~")
exps = list(range(1,11))
exps = [str(int) for int in exps]

exps = ["13", "14", "15", "16", "17", "18", "19", "20", "21"]

for exp in exps:
    exp_no = exp

    # file_path = home + "/Pictures/adaptive_vs/servoing/exps-2-10-2022-window20/"
# file_path = home + "/Desktop"
# traj_path = file_path + exp_no + "/traj.jpg"
# img_path = file_path + exp_no + "/imgs/0.png"
# save_path = file_path + exp_no + "/final_img.jpg"

# traj_path = file_path + "/final4237.tiff"
# img_path = file_path + "/151.png"
# save_path = file_path + "/vsbot_elbow_left.png"
# img1 = cv2.imread(traj_path)

img1 = cv2.imread("/home/jc-merlab/Pictures/Dl_Exps/lama_vs/servoing/exps/20/raw/0.png")
img2 = cv2.imread("/home/jc-merlab/Pictures/Dl_Exps/lama_vs/servoing/exps/20/overlay_1.png")
# out = cv2.addWeighted(img1, 1.0, img2, 1.0,0)
out = cv2.addWeighted(img1, 0.5, img2, 0.8, 0)

# RGBImage = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
# PILimage = Image.fromarray(RGBImage)

# PILimage.save(save_path, dpi=(300,300))
cv2.imwrite('/home/jc-merlab/Pictures/Dl_Exps/lama_vs/servoing/exps/20/overlay.png', out)