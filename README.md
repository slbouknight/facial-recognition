# facial-recognition
A series of programs coded in Python using OpenCV, Tensorflow backend, and open source neural networks to complete tasks including facial location, expression classification, age/gender estimation, and real-time facial recognition.

## Major Libraries & Dependencies
   - Python 3.10 or later
   - NumPy
   - Tensorflow
   - Keras
   - Scikit-learn
   - OpenCV
   - face-recognition https://pypi.org/project/face-recognition/
   - dlib (install from .whl file found in env folder)
     
## How to Use this Project
1. Clone this repository
2. Install all of the major libraries and dependecies
3. Run desired file(s) in an IDE

## Neural Networks 
This project utilizes pre-trained neural network models for expression, age, and gender classification functionalities. All of this information can be found in the dataset folder. The sources for these models and their data is also included below:
1. The 'Kaggle Facial Expression Recognition Challenge' Dataset https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
    - Consists of 48x48 pixel grayscale face images
    - Each image corresponds to 1 of 7 expression categories
    - Total dataset contains approximately 36,000 images

2. Age and Gender Detection Deep Learning Model
    - Created and trained by Gil Levi and Tal Hassner using the Adience datset https://talhassner.github.io/home/projects/Adience/Adience-data.html
    - The trained model files can be downloaded from https://talhassner.github.io/home/publication/2015_CVPR

 ## Project Functionality
 The main functionalities of the project include facial location, expression classification, age/gender estimation, and facial mapping (landmarks). All of which can be performed on images, videos, or real-time via webcam. All image/video data can be edited by adding to the images folder then adjusting the specific pathway. Here are a few screenshots demonstrating these functionalities on images:
### Facial Recognition
<img src="https://github.com/slbouknight/facial-recognition/blob/main/images/demo/facial-recognition.png" width="400" height="300" />

### Expression Classification
<img src="https://github.com/slbouknight/facial-recognition/blob/main/images/demo/expression.png" width="400" height="300" />

### Age/Gender Estimation
<img src="https://github.com/slbouknight/facial-recognition/blob/main/images/demo/age-gender.png" width="400" height="300" />

### Facial Mapping
<img src="https://github.com/slbouknight/facial-recognition/blob/main/images/demo/landmark.png" width="400" height="300" />
