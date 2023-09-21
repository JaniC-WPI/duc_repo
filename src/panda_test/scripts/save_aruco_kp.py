import cv2
import csv
import os

def detect_aruco_markers(image_path):
    # Load the predefined dictionary
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)

    centers = {}
    if ids is not None:
        for i, corner in enumerate(corners):
            if ids[i][0] in [30, 32, 34]:  # Only consider specified IDs
                c = corner[0]
                # Calculate the center of the marker
                cx = int((c[0][0] + c[2][0]) / 2)
                cy = int((c[0][1] + c[2][1]) / 2)
                centers[ids[i][0]] = (cx, cy)
    return centers

def save_centers_to_csv(centers, csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'x', 'y'])
        
        # Loop through the desired order
        for id in [30, 32, 34]:
            if id in centers:
                x, y = centers[id]
                writer.writerow([id, x, y])

# def process_image_folder(folder_path, output_csv_filename):
#     image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]  # you can add more extensions if needed

#     # Open CSV file once and write header
#     with open(output_csv_filename, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['image', 'id', 'x', 'y'])

#         for image_file in image_files:
#             image_path = os.path.join(folder_path, image_file)
#             centers = detect_aruco_markers(image_path)
            
#             # Loop through the desired order
#             for id in [30, 32, 34]:
#                 if id in centers:
#                     x, y = centers[id]
#                     writer.writerow([image_file, id, x, y])

#     print(f"Centers saved to '{output_csv_filename}'")

# if __name__ == "__main__":
#     folder_path = '/home/jc-merlab/Pictures/panda_data/ur10/aruco'
#     output_csv_filename = '/home/jc-merlab/Pictures/panda_data/ur10/aruco/aruco_centers.csv'
#     process_image_folder(folder_path, output_csv_filename)

if __name__ == "__main__":
    image_path = '/home/jc-merlab/Pictures/panda_data/ur10/aruco17/000055.jpg'
    centers = detect_aruco_markers(image_path)
    save_centers_to_csv(centers, '/home/jc-merlab/Pictures/panda_data/ur10/aruco17/aruco_centers.csv')
    print(f"Centers saved to 'centers.csv'")