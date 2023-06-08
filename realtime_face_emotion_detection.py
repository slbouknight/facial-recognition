# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:40:24 2023

@author: slbouknight
"""

# Import required libraries
import numpy as np
import cv2
from keras import utils
from keras.models import model_from_json
import face_recognition

# Get default webcam video stream
webcam_video_stream = cv2.VideoCapture(0)

# Initialize facial expression recognition model
face_exp_model = model_from_json(open('dataset/facial_expression_model_structure.json', 'r').read())

# Load model weights
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')

# Labels for each emotion
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

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
        
        #print(
            #f'Found face {index + 1} at location Top: {top_pos}, Left: {left_pos}, Bottom: {bottom_pos}, Right: {right_pos}')
        
        # Now we'll slice our image array to isolate the faces
        current_face_image = current_frame[top_pos: bottom_pos, left_pos:right_pos]
        
        # Draw rectangle around each face in video frame
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        
        # IMAGE PRE-PROCESSING
        # Convert image to grayscale
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize image to 48x48 px
        current_face_image = cv2.resize(current_face_image, (48, 48))
        
        # Convert PIL image into 3D numpy array
        img_pixels = utils.img_to_array(current_face_image)
        
        # Expand array shape into single row multiple columns
        img_pixels = np.expand_dims(img_pixels, axis=0)
        
        # Normalize pixel values to range [0, 1]
        img_pixels /= 255
        
        # EMOTION PREDICTION USING MODEL
        exp_predictions = face_exp_model.predict(img_pixels)
        
        # Get max indexed prediction value (range from 0-7)
        max_index = np.argmax(exp_predictions[0])
        
        # Obtain corresponding label using array of emotions
        emotion_label = emotions_label[max_index]
        
        # Display label as text over image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
        
        
    # Show current face with rectangle
    cv2.imshow('Webcam Video', current_frame)
        
    # Press 'enter' key to exit loop
    if cv2.waitKey(1) == 13:
        break
        
webcam_video_stream.release()
cv2.destroyAllWindows()
