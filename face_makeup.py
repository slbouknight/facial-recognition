# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:06:38 2023

@author: slbouknight
"""

import face_recognition
from PIL import Image, ImageDraw

# Load image into numpy array
face_image = face_recognition.load_image_file('images/samples/abhi.jpg')

# Obtain list of face landmarks
face_landmarks_list = face_recognition.face_landmarks(face_image)

# Display landmarks
print(face_landmarks_list)

# Iterate through all face landmarks
for face_landmarks in face_landmarks_list:
    # Convert np array to pil image and make a DrawObject
    pil_image = Image.fromarray(face_image)
    d = ImageDraw.Draw(pil_image, 'RGBA')
    
    # Makeup Art
    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128), width=5)
    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128), width=5)
    d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

# Show final image
pil_image.show()

# Save image
pil_image.save('images/samples/abhi_makeup.jpg')