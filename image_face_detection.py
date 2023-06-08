# -*- coding: utf-8 -*-
"""

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
    
    # Display each face
    cv2.imshow(f'Face {index + 1}', current_face_image)
    
# Press any key to exit and close image window
cv2.waitKey(0)
cv2.destroyAllWindows()