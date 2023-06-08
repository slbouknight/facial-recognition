# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:06:36 2023

@author: slbouknight
"""

# Import required libraries
import cv2
import face_recognition

image_to_recognize_path = 'images/testing/trump-modi-unknown.jpg'

# Image we want to detect
original_image = cv2.imread(image_to_recognize_path)

# Load samples and retrieve 128 face encodings for each
face_1 = face_recognition.load_image_file('images/samples/modi.jpg')
face_1_encodings = face_recognition.face_encodings(face_1)[0]
face_1_name = 'Narendra Modi'

face_2 = face_recognition.load_image_file('images/samples/trump.jpg')
face_2_encodings = face_recognition.face_encodings(face_2)[0]
face_2_name = 'Donald Trump'

# Save encodings and corresponding labels to separate arrays in same order
known_face_encodings = [face_1_encodings, face_2_encodings]
known_face_names = [face_1_name, face_2_name]

# Now lets load an unknown image to test against and get its encodings
image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
image_to_recognize_encoding = face_recognition.face_encodings(image_to_recognize)[0]

# Find distance of current coding with all known encodings
face_distances = face_recognition.face_distance(known_face_encodings, image_to_recognize_encoding)

# Print face distance for each known sample to the unknown image
for i, face_distance in enumerate(face_distances):
    print(f'The calculated face distance is {face_distance:.2} against the sample {known_face_names[i]} ')
    print('The matching percentage is {}% against the sample {}'.format(round(((1-float(face_distance))*100), 2), known_face_names[i]))

