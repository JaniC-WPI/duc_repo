import json
import numpy as np
import cv2
import heapq
import os

if __name__ == "__main__": 
    # goal_config = np.array([[267, 432], [271, 315], [317, 218], [342, 231], [463, 293], [494, 281]])
    # goal_config = np.array([[235, 206], [255, 200], [294, 170], [333, 140], [354, 143], [352, 164]])
    # goal_config = np.array([[235, 206], [255, 200], [304, 198], [354, 197], [376, 195], [377, 218]])
    # goal_config = np.array([[259, 203], [280, 203], [315, 238], [351, 272], [366, 288], [351, 303]])
    # goal_config = np.array([[301, 216], [319, 227], [365, 244], [412, 260], [434, 265], [429, 287]])
    # goal_config = np.array([[260, 203], [280, 204], [329, 211], [378, 219], [392, 203], [409, 217]])
    # goal_config = np.array([[303, 217], [320, 229], [366, 248], [413, 266], [434, 274], [426, 295]])

    # goal_config = np.array([[320, 229], [413, 266], [434, 274], [426, 295]])
    # goal_config = np.array([[280, 203], [351, 272], [366, 288], [351, 303]])
    # goal_config = np.array([[180, 240], [218, 145], [230, 127], [247, 141]])
    # goal_config = np.array([[255, 200], [333, 140], [352, 149], [344, 168]])

    # goal_config = np.array([[235, 206], [255, 200], [333, 140], [354, 143], [352, 164]])
    # goal_config = np.array([[235, 206], [255, 200], [354, 197], [376, 195], [377, 218]])
    # goal_config = np.array([[260, 203], [280, 204], [378, 219], [392, 203], [409, 217]])
    # goal_config = np.array([[303, 217], [320, 229], [413, 266], [434, 274], [426, 295]])
    # goal_config = np.array([[259, 203], [280, 203], [351, 272], [366, 288], [351, 303]])

    # goal_config = np.array([[320, 208], [411, 329], [444, 344], [426, 382]])   
    # [184.0, 257.0, 188.0, 239.0, 220.0, 144.0, 231.0, 127.0, 251.0, 139.0]
    # goal_config = np.array([[278, 203], [362, 151], [379, 163], [367, 180]])

    goal_config = np.array([[250, 442], [252, 311], [199, 287], [146, 263], [159, 235], [229, 214], [299, 194], [325, 172], [352, 204]])
    

    goal_image =  np.zeros((480,640,3), dtype=np.int8)

    for point in goal_config:
        cv2.circle(goal_image, tuple(point.astype(int)), radius=9, color=(0, 0, 255), thickness=-1)

    for i in range(len(goal_config) - 1):
        cv2.line(goal_image, tuple(goal_config[i].astype(int)), tuple(goal_config[i+1].astype(int)), (0, 0, 255), 4)

    cv2.imwrite('/home/jc-merlab/.ros/sim_published_goal_image_orig.jpg', goal_image)