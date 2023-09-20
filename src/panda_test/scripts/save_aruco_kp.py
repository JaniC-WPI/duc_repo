import cv2
import csv

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
            if ids[i][0] in [28, 30, 32, 34]:  # Only consider specified IDs
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
        for id in [28, 30, 32, 34]:
            if id in centers:
                x, y = centers[id]
                writer.writerow([id, x, y])

if __name__ == "__main__":
    image_path = '/home/jc-merlab/Pictures/panda_data/aruco_pose_9/000037.jpg'
    centers = detect_aruco_markers(image_path)
    save_centers_to_csv(centers, '/home/jc-merlab/Pictures/panda_data/aruco_pose_9/aruco_centers.csv')
    print(f"Centers saved to 'centers.csv'")