# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:20:27 2023

@author: slbouknight
"""

# Import required libraries
import cv2
import numpy as np
from keras import utils
from keras.models import model_from_json
import face_recognition

# Image we want to detect
image_to_detect = cv2.imread('images/testing/trump-modi.jpg')

# Initialize facial expression recognition model
face_exp_model = model_from_json(open('dataset/facial_expression_model_structure.json', 'r').read())

# Load model weights
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')

# Labels for each emotion
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

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
    
    
    # Draw rectangle around each face in video frame
    cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
    
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
    cv2.putText(image_to_detect, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)
    
    
# Show current face with rectangle
cv2.imshow('Image face emotions', image_to_detect)
    
# Press any key to exit and close image window
cv2.waitKey(0)
cv2.destroyAllWindows()

