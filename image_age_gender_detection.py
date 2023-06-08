# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:25:31 2023

@author: slbouknight
"""

# Import required libraries
import cv2
import face_recognition

# Image we want to detect
image_to_detect = cv2.imread('images/testing/trump-modi.jpg')

# Detect all faces present in image
all_face_locations = face_recognition.face_locations(
    image_to_detect, model='hog')

# Print the number of detected faces
print(f'There are {len(all_face_locations)} faces in this image')

# Let's print the location of each of the detected faces
for index, current_face_location in enumerate(all_face_locations):
    # Splitting up tuple of face location
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print(
        f'Found face {index + 1} at location Top: {top_pos}, Left: {left_pos}, Bottom: {bottom_pos}, Right: {right_pos}')

    # Now we'll slice our image array to isolate the faces
    current_face_image = image_to_detect[top_pos: bottom_pos, left_pos:right_pos]
    
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
    cv2.rectangle(image_to_detect, (left_pos, top_pos),
                  (right_pos, bottom_pos), (0, 0, 255), 2)

    # Display label as text over image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_detect, gender + ' ' + age + ' years',
                (left_pos, bottom_pos + 20), font, 0.5, (255, 255, 255), 1)

# Show current face with rectangle
cv2.imshow('Image', image_to_detect)
    
# Press any key to exit and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
