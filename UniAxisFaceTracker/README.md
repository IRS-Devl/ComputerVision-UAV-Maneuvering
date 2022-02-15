# Single Axis Face Tracker

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com) 


![Img](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/depthai.png)
![Img2](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/opencv.png)

## Hardware requirements:
- Raspberry pi 3 Model B(and all its accessories)
- 1 OAK-D Lite Camera
- 1 MG90S Micro Servo motor(and its accessories)
- USB to USB-C cable for the camera
- 3 Male to female jumper wires

## About the code
YuNet face detection model is being used to detection faces in the frame. The neural network inference is carried out by oak d lite camera mounted upon a servo motor which in turn is controlled by the raspberry pi. The aim of this code is to detect the face and to track and keep it in the center of the camera frame. Simultaenously, computer vision based feedback instructions for motor actuation are displayed on the frame itself while the code is running.

## Tips to run the code
In order to deploy the code, certain steps have to be followed. A virtual environment for python has to be setup, in which the dependencies given in the facetrackreq.txt file has to be downloaded. Once this is done the drone.py file can be executed directly to view the results.

While deploying the code, make sure to give the correct path to the blob file here in the Drone.py according to the file structured followed.
line 50: parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='/home/pi/Desktop/Facetracker/face_detection_yunet_120x160.blob', type=str)
