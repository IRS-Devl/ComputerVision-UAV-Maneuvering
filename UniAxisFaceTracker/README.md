# Single Axis Face Tracker

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com) ![Img](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/made-with-depthai.svg)![Img2](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/made-with-opencv.svg)![Img3](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/oak-d-lite-camera.svg)

## Hardware requirements:
- Raspberry pi 3 Model B(and all its accessories)
- 1 OAK-D Lite Camera
- 1 MG90S Micro Servo motor(and its accessories)
- USB to USB-C cable for the camera
- 3 Male to female jumper wires

If powering up the motor is not desired from the raspberry pi then additional components can be used:
- 15V battery
- Buck Convertor(15V to 5V)
- Breadboard
- Extra jumper wires 


## About the code
YuNet face detection model is used to detect faces in the frame. The neural network inference is carried out by oak d lite camera mounted upon a servo motor which in turn is controlled by the raspberry pi. The aim of this code is to detect a face, track and keep it in the center of the camera frame within a certain threshold range of pixels values. Simultaenously, computer vision based feedback instructions for motor actuation, bounding boxes, relevant coordinates for reference and tracking are displayed in the frame itself while the code is running to make it more comprehensive and intuitive. As only one motor is controlled, the camera tracks faces about z axis(yaw movement). Also, the code is designed to track only one object at a time and the workspace of the camera is a semi-circle due to the servo motor rotation limitations.


## Tips to run the code
In order to deploy the code, certain steps have to be followed. A virtual environment for python has to be setup, in which the dependencies given in the facetrackreq.txt file has to be downloaded. Once this is done the drone.py file can be executed directly to view the results.

While deploying the code, make sure to give the correct path to the blob file here in the Drone.py according to the file structured followed.
line 50: parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='/home/pi/Desktop/Facetracker/face_detection_yunet_120x160.blob', type=str)

## Prerequisites
Complete setup of raspbian on raspberrypi 3B/3B+/4B. It should be up and running connected via monitor,keyboard,mouse,wifi/ethernet.
Note: The live face detection feed and face tracking won't be displayed if the raspberrypi is connected via ssh.

## Face Tracking setup and Results
![GIF facetrack](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/FaceGIF.gif)
![Setup](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/FaceTrackingSetup.jpeg)

## Steps

## Contributors
* [Anmol Singh](https://github.com/28anmol)
* [Luxonis Depthai Experiments](https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-detection)
(The face detection model and its deployment code on oak d lite camera is taken from the above given link.)
* https://github.com/OlanrewajuDada (Credits for Face Tracking Results GIF)
