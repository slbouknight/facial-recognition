# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:19:00 2023

@author: slbouknight
"""

# Import required libraries
import cv2
import face_recognition

# Get default webcam video stream
video_stream = cv2.VideoCapture('images/testing/modi.mp4')

# Array to store face locations
all_face_locations = []


# Loop through each video frame until user exits
while True:
    ret, current_frame = video_stream.read()

    # Lets use a smaller version (0.25x) of the image for faster processing
    scale_factor = 4
    current_frame_small = cv2.resize(
        current_frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)

    # Find total number of faces
    all_face_locations = face_recognition.face_locations(
        current_frame_small, number_of_times_to_upsample=2, model='hog')

    # Let's print the location of each of the detected faces
    for index, current_face_location in enumerate(all_face_locations):
        # Splitting up tuple of face location
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        # Correct positions based on scale factor
        top_pos *= scale_factor
        right_pos *= scale_factor
        bottom_pos *= scale_factor
        left_pos *= scale_factor
        
        print(
            f'Found face {index + 1} at location Top: {top_pos}, Left: {left_pos}, Bottom: {bottom_pos}, Right: {right_pos}')

        # Draw rectangle around each face in video frame
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        
    # Show current face with rectangle
    cv2.imshow('Video', current_frame)
        
    # Press 'enter' key to exit loop
    if cv2.waitKey(1) == 13:
        break
        
video_stream.release()
cv2.destroyAllWindows()
