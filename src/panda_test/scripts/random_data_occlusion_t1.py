import os
import cv2
import random
import math
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

from shapely import geometry
from shapely.ops import unary_union
import geopandas as gpd

"""
Randomly put occlusion to robot image. The occlusion will be in proximity of
keypoints. Number of occlusion and size of occlusion are adjustable.
"""

int_stream = '000000'


def load_images(folder_path):
    images = []
    keypoints = []
    image_files = []
    data_files = os.listdir(folder_path)  # all files in the data folder
    # filter for json files
    json_files = sorted([f for f in data_files if f.endswith('.json')])

    # use cv2 to plot each image with keypoints and bounding boxes
    for j in range(len(json_files)):
        # process file names
        new_stream = int_stream[0:-len(str(j))]
        json_path = folder_path + new_stream + str(j) + '.json'

        with open(json_path, 'r') as f_json:
            data = json.load(f_json)
            image = cv2.imread(folder_path + data['image_rgb'])
            # Load image and keypoints
            images.append(image)
            image_files.append(data['image_rgb'])
            keypoints.append(data['keypoints'])

    return images, image_files, keypoints


def occlude_random_parts(image, min_occlusion_size, max_occlusion_size,
                         num_occlusions,
                         keypoints, w1, w2,
                         visualize=False):

    def random_points_in_polygon(polygon, number) -> geometry.Point:
        """
        Returns [number] random points that lie inside [polygon].
        Source: https://www.matecdev.com/posts/random-points-in-polygon.html
        """
        points = []
        minx, miny, maxx, maxy = polygon.bounds
        while len(points) < number:
            pnt = geometry.Point(
                np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if polygon.contains(pnt):
                points.append(pnt)
        return points

    link_lengths = []
    sin_a = []
    cos_a = []
    polygons = []
    for i in range(1, len(keypoints)):
        # Coordinates of 2 adjacent keypoints
        x1, y1 = keypoints[i-1][0][0], keypoints[i-1][0][1]
        x2, y2 = keypoints[i][0][0], keypoints[i][0][1]
        # ignore duplicate frames
        if x1 == x2 and y1 == y2:
            continue
        link_lengths.append(math.sqrt((x1-x2)**2 + (y1-y2)**2))
        sin_a.append((y2-y1)/link_lengths[-1])
        cos_a.append((x2-x1)/link_lengths[-1])
        # First 2 vertices of the bounding box
        v1 = np.array([[x1 + math.copysign(w2*cos_a[-1], x1-x2)],
                       [y1 + math.copysign(w2*sin_a[-1], y1-y2)]]) + \
             w1*np.array([[-sin_a[-1], sin_a[-1]],
                          [cos_a[-1], -cos_a[-1]]])
        # Last 2 vertices of the bounding box
        v2 = np.array([[x2 + math.copysign(w2*cos_a[-1], x2-x1)],
                       [y2 + math.copysign(w2*sin_a[-1], y2-y1)]]) + \
             w1*np.array([[sin_a[-1], -sin_a[-1]],
                          [-cos_a[-1], cos_a[-1]]])
        # Combined vertices of the bounding box
        vertices = np.concatenate((v1, v2), axis=1)
        polygons.append(geometry.Polygon(vertices.T))

    # Take union of bounding boxes to get ROI
    roi = unary_union(polygons)

    # Generate random vertices of occlusions
    occlusion_rectangles = []
    occlusion_origins = random_points_in_polygon(roi, num_occlusions)
    for o in occlusion_origins:
        # Random radius
        r = random.uniform(
            min_occlusion_size*math.sqrt(2), max_occlusion_size*math.sqrt(2))
        # Random angle
        phi = random.uniform(-math.pi, math.pi)
        # Calculate occlusion regions...
        x_min = int(min(o.x, o.x + r*math.cos(phi)))
        x_max = int(max(o.x, o.x + r*math.cos(phi)))
        y_min = int(min(o.y, o.y + r*math.sin(phi)))
        y_max = int(max(o.y, o.y + r*math.sin(phi)))
        # ...then turn into black pixels
        image[y_min:y_max, x_min:x_max] = 0
        occlusion_rectangles.append([x_min, y_min, x_max, y_max])

    # Visualize on image
    if visualize:
        roi_vis = np.array(roi.exterior.coords[:-1], np.int32)
        roi_vis = roi_vis.reshape((-1, 1, 2))
        img = cv2.polylines(image, [roi_vis], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.imshow('Occlusion visualization', img)
        cv2.waitKey(0)

    return image, occlusion_rectangles

# def save_images(images, filenames, output_folder):
#     os.makedirs(output_folder, exist_ok=True)

#     for filename, image in zip(filenames, images):
#         output_path = os.path.join(output_folder, f'occluded_{filename}')
#         output_path_plt = os.path.join(output_folder, f'occluded_plt_{filename}')
#         cv2.imwrite(output_path, image)
#         plt.imshow(image)
#         plt.show()
#         # plt.savefig(output_path_plt, image)

def save_images(images, filenames, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename, image in zip(filenames, images):
        print(f'Saving image: {filename}')
        output_path = os.path.join(output_folder, f'{filename}')
        output_path_plt = os.path.join(output_folder, 'plt', f'occluded_plt_{filename}')
        cv2.imwrite(output_path, image)
        # plt.imshow(image)
        # plt.grid(True)  # Turns on the grid
        # plt.savefig(output_path_plt)  # Saves the image with the grid
        # plt.show()
        # plt.clf()

def load_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

def draw_occlusion_rectangles(image, occlusion_rectangles):
    for occlusion_rectangle in occlusion_rectangles:
        x_min, y_min, x_max, y_max = occlusion_rectangle
        cv2.circle(image, (x_min, y_min), 3, (0, 255, 0), -1)
        cv2.circle(image, (x_max, y_max), 3, (0, 0, 255), -1)
        # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image

def is_point_occluded(x, y, occlusion_rectangles):
    # [x_min, y_min, x_max, y_max] = occlusion_rectangles
    # print(x_min, x, x_max)
    # print(y_min, y, y_max)
    # print(x, y)
    # print(occlusion_rectangles)
    for x_min, y_min, x_max, y_max in occlusion_rectangles:
        if x_min < x < x_max and y_min < y < y_max:
            print(f"Point ({x}, {y}) is occluded by rectangle ({x_min}, {y_min}, {x_max}, {y_max})")            
            # x, y = #keypoint from step 2
            # x_min, y_min, x_max, y_max = #occlusion rectangle from step 2   
            # print(x_min < x < x_max and y_min < y < y_max)         
            return True
        # print(x_min < x < x_max and y_min < y < y_max)
    return False

def update_keypoints_visibility(json_data, occlusion_rectangles):
    for k_index, keypoints in enumerate(json_data['keypoints']):
        # print(keypoints)
        for kp_index, keypoint in enumerate(keypoints):
            x, y, visibility = keypoint
            if is_point_occluded(x, y, occlusion_rectangles):
                print(f"Updating visibility for point ({x}, {y})")
                json_data['keypoints'][k_index][kp_index][2] = 0

    return json_data


def save_json(json_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(json_data, f)


def main():
    # Parameters
    visualize = False  # if True, display the results

    # Increase the occlusion size
    min_occlusion_size = 20
    max_occlusion_size = 80
    num_occlusions = 3

    input_folder = '/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/kp_test_images/8/'
    output_folder = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/occluded_results_mi{min_occlusion_size}_ma{max_occlusion_size}_n{num_occlusions}/'

    images, filenames, all_keypoints = load_images(input_folder)

    occluded_images = []
    occlusion_rectangles_list = []
    for (image, filename, keypoints) in zip(images, filenames, all_keypoints):
        print(f'Processing {filename}')
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        occluded_image, occlusion_rectangles = occlude_random_parts(
            image_cv, min_occlusion_size, max_occlusion_size, num_occlusions,
            keypoints, 20, 5, visualize=visualize)

        occluded_images.append(occluded_image)
        occlusion_rectangles_list.append(occlusion_rectangles)

    save_images(occluded_images, filenames, output_folder)

    json_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.json')])

    for i, json_file in enumerate(json_files):
        json_path = os.path.join(input_folder, json_file)
        json_data = load_json(json_path)

        # The occlusion rectangles for the current image
        occlusion_rectangles = occlusion_rectangles_list[i]
        updated_json_data = update_keypoints_visibility(json_data, occlusion_rectangles)

        output_json_path = os.path.join(output_folder, f'{json_file}')
        save_json(updated_json_data, output_json_path)


if __name__ == '__main__':
    main()
