import json
import numpy as np
import cv2
import heapq
import os

if __name__ == "__main__": 
    # goal_config = np.array([[267, 432], [271, 315], [317, 218], [342, 231], [463, 293], [494, 281]])
    goal_config = np.array([[316, 225], [399, 279], [413, 295]])

    goal_image =  np.zeros((480,640,3), dtype=np.int8)

    for point in goal_config:
        cv2.circle(goal_image, tuple(point.astype(int)), radius=9, color=(0, 0, 255), thickness=-1)

    for i in range(len(goal_config) - 1):
        cv2.line(goal_image, tuple(goal_config[i].astype(int)), tuple(goal_config[i+1].astype(int)), (0, 0, 255), 4)

    cv2.imwrite('/home/jc-merlab/.ros/sim_published_goal_image_orig.jpg', goal_image)