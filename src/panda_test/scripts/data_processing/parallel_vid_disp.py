#!/usr/bin/env python3

import cv2

# Open the video files
cap1 = cv2.VideoCapture('/home/jc-merlab/Pictures/panda_data/images_for_occlusion/1/occ_test_1.avi')
cap2 = cv2.VideoCapture('/home/jc-merlab/Pictures/panda_data/images_for_occlusion/1/occ_test_1_op.avi')
cap3 = cv2.VideoCapture('/home/jc-merlab/Pictures/panda_data/images_for_occlusion/1/occ_test_1_kp.avi')

# Get the height of the videos to resize if necessary
height = max(int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while cap1.isOpened() and cap2.isOpened() and cap3.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    # Check if all frames are read successfully
    if not ret1 or not ret2 or not ret3:
        print("Failed to grab all the frames")
        break

    # Resize frames to have the same height
    frame1 = cv2.resize(frame1, (int(frame1.shape[1] * height / frame1.shape[0]), height))
    frame2 = cv2.resize(frame2, (int(frame2.shape[1] * height / frame2.shape[0]), height))
    frame3 = cv2.resize(frame3, (int(frame3.shape[1] * height / frame3.shape[0]), height))

    # Concatenate the frames horizontally
    combined_frame = cv2.hconcat([frame1, frame2, frame3])

    # Display the concatenated frame
    cv2.imshow('Videos Side by Side', combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video captures and close OpenCV windows
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()