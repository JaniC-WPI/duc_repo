#!/usr/bin/env python3
import json
import numpy as np
import cv2
import heapq
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from inference.models.utils import get_roboflow_model
import torchvision
from PIL import Image
import torchvision.transforms as T

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 20  # Safe distance from the obstacle
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        # if filename.endswith('.json'):
        if filename.endswith('.json') and not filename.endswith('_combined.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                # Convert keypoints to integers
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
    # print(configurations)
    return configurations

def load_and_sample_configurations(directory, num_samples):
    # Load configurations from JSON files
    configurations = load_keypoints_from_json(directory)

    # If there are more configurations than needed, sample a subset
    if len(configurations) > num_samples:
        sampled_indices = np.random.choice(len(configurations), size=num_samples, replace=False)
        sampled_configurations = [configurations[i] for i in sampled_indices]
    else:
        sampled_configurations = configurations

    return sampled_configurations

def build_lazy_roadmap_with_kdtree(configurations, k_neighbors):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    """
    # Flatten each configuration to work with KDTree
    flattened_configs = [config.flatten() for config in configurations]
    
    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(flattened_configs)
    
    # Initialize the roadmap as a graph
    G = nx.Graph()
    
    # Add configurations as nodes
    for i, config in enumerate(configurations):
        G.add_node(i, configuration=config)
    
    # Use KDTree to find k nearest neighbors for each node and add edges
    for i, config in enumerate(flattened_configs):
        # Ensure the query point (config) is in a 2D array
        query_point = np.array([config])  # Reshape config into a 2D array
        distances, indices = tree.query(query_point, k=k_neighbors + 1)  # +1 because query includes the point itself
        for j in indices[1:]:  # Skip the first index to exclude self
            G.add_edge(i, j)
    
    return G, tree

# Add a new configuration to the roadmap
def add_config_to_roadmap(config, G, tree, k_neighbors=5):
    flattened_config = config.flatten()
    distances, indices = tree.query([flattened_config], k=k_neighbors)
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config)
    
    for i in indices[0]:
        G.add_edge(node_id, i)
        
    return node_id

# Find a path between start and goal configurations
def find_path(G, tree, start_config, goal_config, k_neighbors=5):
    start_node = add_config_to_roadmap(start_config, G, tree, k_neighbors)
    goal_node = add_config_to_roadmap(goal_config, G, tree, k_neighbors)
    path = nx.shortest_path(G, source=start_node, target=goal_node)
    return path

if __name__ == "__main__":
    directory = '/path/to/your/json/files'  # Update this path
    configurations = load_keypoints_from_json(directory)
    num_neighbors = 5  # Adjust based on your dataset size and density
    
    start_time = time.time()
    roadmap, tree = build_lazy_roadmap_with_kdtree(configurations, num_neighbors)
    end_time = time.time()
    print("Roadmap built in:", end_time - start_time, "seconds")
    
    # Define start and goal configurations as numpy arrays
    start_config = np.array([[279, 414], [278, 298], [305, 197], [331, 203], [444, 269], [472, 280]])
    goal_config = np.array([[279, 414], [278, 298], [186, 253], [198, 229], [232, 106], [230, 72]])
    
    # Find and print the path from start to goal
    path = find_path(roadmap, tree, start_config, goal_config, num_neighbors)
    print("Path from start to goal:", path)


# def object_detection_yolov8(image_path):
#     model = YOLO("yolov5xu.pt")
#     results = model.predict(image_path)
    
#     return results[0]

# def object_detection_ycb_roboflow(image_path):
#     model = get_roboflow_model(model_id="ycb-object-dataset/2", api_key="Uvh7kNIxrhp3Ftky3FpA")

#     results = model.infer(image_path, confidence=0.05, iou_threshold=0.5)
#     print(results[0])

#     # Extract bounding boxes and class names
#     boxes = []
#     classes = []
#     for prediction in results.predictions:
#         # Convert (x, y, width, height) to (x_min, y_min, x_max, y_max)
#         x_min = prediction.x - prediction.width / 2
#         y_min = prediction.y - prediction.height / 2
#         x_max = prediction.x + prediction.width / 2
#         y_max = prediction.y + prediction.height / 2
#         boxes.append([(x_min, y_min), (x_max, y_max)])
#         classes.append(prediction.class_name)
    
#     return boxes, classes

#     return results

# def object_detection_rcnn(image_path, threshold):
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
#     model.eval()
#     img = Image.open(image_path)
#     transform = T.Compose([T.ToTensor()])
#     img = transform(img)
#     pred = model([img])
#     print(pred)
#     pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
#     pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
#     pred_score = list(pred[0]['scores'].detach().numpy())
#     pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
#     pred_boxes = pred_boxes[:pred_t+1]
#     pred_class = pred_class[:pred_t+1]
#     return pred_boxes, pred_class

# def draw_bounding_box(image_path, box, obj_class):
#     # Load the image
#     img = cv2.imread(image_path)
#     # Define the color and thickness of the box
#     color = (0, 255, 0)  # Green
#     thickness = 2
    
#     # Convert box coordinates to integer
#     # x_min, y_min, x_max, y_max = map(int, box)
#     x_min, y_min, x_max, y_max = int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])
#     cv2.putText(img, obj_class, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  
    
#     # Draw the bounding box
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    
#     # Display the image with bounding box
#     cv2.imshow("Image with Bounding Box", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def draw_bounding_boxes(image_path, boxes):
#     # Load the image
#     img = cv2.imread(image_path)
    
#     # Define the color and thickness of the box
#     color = (0, 255, 0)  # Green
#     thickness = 2

#     # Iterate through all boxes and draw them on the image
#     for box in boxes:
#         x_min, y_min, x_max, y_max = int(box[0][0]), int(box[0][1]), int(box[1][0]), int(box[1][1])
#         cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    
#     # Display the image with bounding boxes
#     cv2.imshow("Image with Bounding Boxes", img)
#     # Wait for key press to proceed
#     while True:
#         # If 'q' is pressed, break from the loop
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             break

#     # Close the current window to proceed to the next one
#     cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    num_samples = 500
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/regression_dataset_panda/'  # Replace with the path to your JSON files
    configurations = load_and_sample_configurations(directory, num_samples)
    # Parameters for PRM
    num_neighbors = 50  # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap = build_lazy_roadmap_with_kdtree(configurations, num_neighbors)   
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)  

    # image_path = '/home/jc-merlab/Pictures/panda_data/images_for_occlusion/1/image_864.jpg'
    # # results = object_detection_yolov8(image_path)
    # # results = object_detection_ycb_roboflow(image_path)
    # results = object_detection_rcnn(image_path, threshold=0.7)
    # print(results)

    # boxes, classes = results
    # # print(boxes)
    
    # # # draw_bounding_boxes(image_path, boxes)
    # for box, obj_class in zip(boxes, classes):
    #     draw_bounding_box(image_path, box, obj_class)

    # # draw_bounding_box(image_path, boxes)

    # # boxes = results.boxes.xyxy[0].cpu().numpy()
    # # # boxes = np.array(boxes)
    # # print(boxes)
    # # draw_bounding_box(image_path, boxes)

    # # end_time = time.time()


