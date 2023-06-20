# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:41:52 2023

@author: slbouknight
"""

# Import required libraries
import cv2
import face_recognition

# Image we want to detect
original_image = cv2.imread('images/testing/bey-jay.jpg')

# Load samples and retrieve 128 face encodings for each
face_1 = face_recognition.load_image_file('images/samples/bey.jpg')
face_1_encodings = face_recognition.face_encodings(face_1)[0]
face_1_name = 'Beyonce'

face_2 = face_recognition.load_image_file('images/samples/jay.jpg')
face_2_encodings = face_recognition.face_encodings(face_2)[0]
face_2_name = 'Jay-Z'

# Save encodings and corresponding labels to separate arrays in same order
known_face_encodings = [face_1_encodings, face_2_encodings]
known_face_names = [face_1_name, face_2_name]

# Now lets load an unknown image to test against
image_to_recognize = face_recognition.load_image_file(
    'images/testing/bey-jay.jpg')

# Find all the faces/encodings in our test image
all_face_locations = face_recognition.face_locations(
    image_to_recognize, model='hog')

all_face_encodings = face_recognition.face_encodings(
    image_to_recognize, all_face_locations)

# Iterate through each face location and encoding in our test image
for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
    # Splitting up tuple of face location
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print(
        f'Found face at location Top: {top_pos}, Left: {left_pos}, Bottom: {bottom_pos}, Right: {right_pos}')

    # Now we'll slice our image array to isolate the faces
    current_face_image = image_to_recognize[top_pos: bottom_pos,
                                            left_pos:right_pos]

    # Compare to known faces to check for matches
    all_matches = face_recognition.compare_faces(
        known_face_encodings, current_face_encoding)

    # Initialize name string as unknown face
    name_of_person = 'Unknown Face'

    # Check if all_matches isn't empty
    # If yes get the index number corresponding to the face in the first index
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]

    # Draw rectangle around face
    cv2.rectangle(original_image, (left_pos, top_pos),
                  (right_pos, bottom_pos), (255, 255, 255), 2)

    # Write corresponding name below face
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos,
                bottom_pos + 20), font, 0.5, (255, 255, 255), 1)

    # Display image with rectangle and text
    cv2.imshow('Identified Faces', original_image)

# Press any key to exit and close image window
cv2.waitKey(0)
cv2.destroyAllWindows()
