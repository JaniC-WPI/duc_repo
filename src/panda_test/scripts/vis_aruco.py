import cv2
import csv

def read_aruco_centers_from_csv(csv_path):
    """Read ArUco centers from the CSV file."""
    centers = {}
    current_pose = None

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header

        for row in reader:
            # Check if we have a new pose
            if row[0]:  # If first column (pose) is not empty
                current_pose = int(row[0])
            
            # Only proceed if there's a valid pose
            if current_pose:
                aruco_id, x, y = int(row[1]), int(row[2]), int(row[3])
                if current_pose not in centers:
                    centers[current_pose] = {}
                centers[current_pose][aruco_id] = (x, y)

    return centers

def visualize_aruco_centers_on_images(image_folder, centers):
    """Visualize ArUco centers on the images."""
    for pose, coords in centers.items():
        # Construct the image filename based on the pose
        image_path = f"{image_folder}/{str(pose).zfill(6)}.jpg"
        
        image = cv2.imread(image_path)
        for aruco_id, (x, y) in coords.items():
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Draw red circle for each ArUco center
            cv2.putText(image, str(aruco_id), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)  # Add ArUco ID next to the circle
        
        cv2.imshow(f"Image {pose}", image)
        cv2.waitKey(0)  # Wait for a key press to show the next image

    cv2.destroyAllWindows()

if __name__ == "__main__":
    csv_path = "/home/jc-merlab/Pictures/panda_data/aruco_pose/aruco_centers.csv"
    image_folder = "/home/jc-merlab/Pictures/panda_data/aruco_pose/"

    aruco_centers = read_aruco_centers_from_csv(csv_path)
    visualize_aruco_centers_on_images(image_folder, aruco_centers)