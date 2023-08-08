#!/usr/bin/env python3

import os
import cv2
import numpy as np
# from shapely import affinity
# from shapely.geometry import Polygon
import json


class SimMaskGen:
    """
    Generates masks at keypoints for occlusion applications.
    """

    def __init__(self, raw_folder, dest_folder,
                 keypoints,
                 mask_half_width=5,
                 save=True, visualize=False):
        """
        Generate masks at specified keypoints.

        raw_folder: path to raw image folder
        dest_folder: path to destination results folder

        keypoints: List of kp indices.

        mask_half_width: Half of width of masks
        save: if True, save the results to files
        visualize: if True, display the result for each image
        """
        self.raw_folder = raw_folder
        all_files = os.listdir(raw_folder)  # all files in the data folder
        self.img_files = [f for f in all_files
                          if (f.endswith('jpg') or f.endswith('png'))]
        self.json_files = [f for f in all_files if f.endswith('json')]
        self.dest_folder = dest_folder

        self.keypoints = keypoints
        self.mask_hw = mask_half_width

        self.visualize = visualize

        self.save = save

    def gen_mask(self, img, kps):
        """
        Generate mask of markers.
        """
        mask = np.zeros(img.shape[:2], np.uint8)  # black by default
        for i in self.keypoints:
            corners = [
                [kps[i][0]-self.mask_hw, kps[i][1]-self.mask_hw],
                [kps[i][0]-self.mask_hw, kps[i][1]+self.mask_hw],
                [kps[i][0]+self.mask_hw, kps[i][1]+self.mask_hw],
                [kps[i][0]+self.mask_hw, kps[i][1]-self.mask_hw]]
            cv2.fillPoly(mask, [np.array(corners, dtype=int)], 255)
        # Enlarge the mask to account for the white border
        # Method 1: dilation
        # cv2.fillPoly(mask, corners, 255)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        # mask = cv2.dilate(mask, kernel, iterations=1)
        # Method 2: scale the bounding polygon
        # for i in range(len(corners)):
        #     polygon = Polygon(corners[i][0])
        #     scaled_polygon = affinity.scale(polygon, xfact=1.35, yfact=1.35)
        #     corners[i][0] = \
        #         np.array(scaled_polygon.exterior.coords[:-1], np.int32)
        #     # keypoint [[x,y,visibility]]
        #     kps.append([list(scaled_polygon.centroid.coords[0]) + [0]])
        #     cv2.fillPoly(mask, corners, 255)

        if self.visualize:
            masked_img = np.copy(img)
            mask_3ch = cv2.merge((mask, mask, mask))  # 3 channel
            # cv2.fillPoly(masked_img, corners, color=(255, 255, 255))
            masked_img = cv2.bitwise_and(masked_img,
                                         cv2.bitwise_not(mask_3ch))
            # Visualizes original, mask, masked image
            vis_img = np.concatenate((img, mask_3ch, masked_img), axis=1)

            cv2.imshow('visualize', vis_img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        return mask

    def run(self):
        for f in self.json_files:
            json_path = os.path.join(self.raw_folder, f)
            with open(json_path, 'r') as f_json:
                data = json.load(f_json)

                raw_file_name, extension = os.path.splitext(data['image_rgb'])
                mask_file_name = raw_file_name + '_mask' + extension
                img = cv2.imread(
                    os.path.join(self.raw_folder, data['image_rgb']))
                keypoints = [kp[0] for kp in data['keypoints']]
                mask = self.gen_mask(img, keypoints)
                if self.save:
                    cv2.imwrite(
                        os.path.join(self.dest_folder, mask_file_name), mask)
                    print(f'Saved mask: {mask_file_name}')
            if self.save:
                with open(os.path.join(self.dest_folder, f), 'w') as f_json:
                    # Update json
                    data['keypoints'] = [data['keypoints'][i]
                                         for i in self.keypoints]
                    data['bboxes'] = [data['bboxes'][i]
                                      for i in self.keypoints]
                    json_obj = json.dumps(data, indent=4)
                    f_json.write(json_obj)


if __name__ == "__main__":
    raw_folder = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/sim_marker/raw/'
    dest_folder = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/sim_marker/raw_mask/'

    SimMaskGen(raw_folder,
               dest_folder,
               keypoints=[1, 5],
               mask_half_width=8,
               save=False,
               visualize=True).run()
