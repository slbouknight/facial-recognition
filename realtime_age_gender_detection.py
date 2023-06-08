# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:35:39 2023

@author: slbouknight
"""

# Import required libraries
import cv2
import face_recognition

# Get default webcam video stream
webcam_video_stream = cv2.VideoCapture(0)

# Array to store face locations
all_face_locations = []


# Loop through each video frame until user exits
while True:
    ret, current_frame = webcam_video_stream.read()

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

        # print(
        # f'Found face {index + 1} at location Top: {top_pos}, Left: {left_pos}, Bottom: {bottom_pos}, Right: {right_pos}')

        # Now we'll slice our image array to isolate the faces
        current_face_image = current_frame[top_pos: bottom_pos,
                                           left_pos:right_pos]

        # The 'AGE_GENDER_MODEL_MEAN_VALUES' calculated by using numpy mean()
        AGE_GENDER_MODEL_MEAN_VALUES = (
            78.4263377603, 87.7689143744, 114.895847746)

        # Create a blob of the current face slice
        current_face_image_blob = cv2.dnn.blobFromImage(
            current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES)

        # USING GENDER MODEL
        # Declaring labels
        gender_label_list = ['Male', 'Female']

        # Specify file paths
        gender_protext = 'dataset/gender_deploy.prototxt'
        gender_caffemodel = 'dataset/gender_net.caffemodel'

        # Model creation and input
        gender_conv_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        gender_conv_net.setInput(current_face_image_blob)

        # Get gender predictions and get label of max value
        gender_predictions = gender_conv_net.forward()
        gender = gender_label_list[gender_predictions[0].argmax()]

        # USING AGE MODEL
        # Declaring labels
        age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                          '(25-32)', '(38-43)', '(48-53)', '(60-100)']

        # Specify file paths
        age_protext = 'dataset/age_deploy.prototxt'
        age_caffemodel = 'dataset/age_net.caffemodel'

        # Model creation and input
        age_conv_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        age_conv_net.setInput(current_face_image_blob)

        # Get gender predictions and get label of max value
        age_predictions = age_conv_net.forward()
        age = age_label_list[age_predictions[0].argmax()]

        # Draw rectangle around each face in video frame
        cv2.rectangle(current_frame, (left_pos, top_pos),
                      (right_pos, bottom_pos), (0, 0, 255), 2)

        # Display label as text over image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, gender + ' ' + age + ' years',
                    (left_pos, bottom_pos + 20), font, 0.5, (255, 255, 255), 1)

    # Show current face with rectangle
    cv2.imshow('Webcam Video', current_frame)

    # Press 'enter' key to exit loop
    if cv2.waitKey(1) == 13:
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
