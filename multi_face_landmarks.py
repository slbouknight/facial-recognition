# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:59:17 2023

@author: slbouknight
"""

import face_recognition
from PIL import Image, ImageDraw

# Load image into numpy array
face_image = face_recognition.load_image_file('images/testing/trump-modi.jpg')

# Obtain list of face landmarks
face_landmarks_list = face_recognition.face_landmarks(face_image)

# Display landmarks
print(face_landmarks_list)

# Convert np array to pil image and make a DrawObject
pil_image = Image.fromarray(face_image)
d = ImageDraw.Draw(pil_image)

# Loop through every face
index = 0
while index < len(face_landmarks_list):
    
    # Iterate through all face landmarks in a given face
    for face_landmarks in face_landmarks_list:
        
        # Connect each face landmark point using a white line
        d.line(face_landmarks['chin'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['left_eyebrow'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['right_eyebrow'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['nose_bridge'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['nose_tip'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['left_eye'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['right_eye'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['top_lip'], fill=(255, 255, 255), width=2)
        d.line(face_landmarks['bottom_lip'], fill=(255, 255, 255), width=2)
    
    index += 1
    
    # Show final image
    pil_image.show()
    
    # Save image
    pil_image.save('images/samples/multi_landmarks.jpg')
