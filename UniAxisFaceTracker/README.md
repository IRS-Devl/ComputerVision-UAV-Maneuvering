# Single Axis Face Tracker

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com) ![Img](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/made-with-depthai.svg)![Img2](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/made-with-opencv.svg)![Img3](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/oak-d-lite-camera.svg)

# Table of Contents
1. [Aim](#aim)
2. [Hardware requirements and setup](#hardware-requirements-and-setup)
3. [About the code](#about-the-code)
4. [Tips to run the code](#tips-to-run-the-code)
5. [Prerequisites](#prerequisites)
6. [Steps](#steps)
7. [Challenges](#challenges)
8. [Improvements](#improvements)
9. [Face tracking setup and results](#face-tracking-setup-and-results)
10. [Contributors](#contributors)



## Aim
The aim of this code is based on the theme: "Computer Vision based feedback on maneuvering of unmanned aerial vehicles". The camera detects an object in the frame and actuates the motors in a way such that the object is tracked and centered in the frame at all times. This is crucial to keep the test object in the frame when the UAV has to follow the particular test object and accomplish some mission autonomously.

## Hardware requirements and setup
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

Regarding the hardware setup, a few things needs te be taken care of before actually executing the code
- The servo motor should be mounted such that it is free to rotate in a semicircle(0-180deg)
- The camera is mounted on the motor shaft parallel to when the motor shaft head is pointing at 90deg and it has +/- 90 deg of free rotation on both sides. So setting up and mounting the camera at 90deg is a reference.
    - In order to calibrate the motor and mount the camera in the correct orientation, following steps can be followed. Run the following commands on raspberrypi terminal(assuming the signalpin of the servo is connected to GPIO 11 and a head is connected on top of the shaft)   
        Opens up the python shell
        ```bash
        sudo python                         
        ```
        ```bash
        >>> import RPi.GPIO as GPIO
        ```
        ```bash
        >>> import time
        ```
        ```bash
        >>> GPIO.setmode(GPIO.BOARD)
        ```
        ```bash
        >>> GPIO.setwarnings(False)
        ```
        ```bash
        >>> signalpin = 11
        ```
        ```bash
        >>> GPIO.setup(signalpin,GPIO.OUT)
        ```
        ```bash
        >>> pwm = GPIO.PWM(signalpin,50)
        ```
        Rotates the servo to 0 degree position
        ```bash
        >>> pwm.start(2) 
        ```                    
        Rotates the servo to 90 degree position
        ```bash
        >>> pwm.ChangeDutyCycle(7)
        ```            
        Temporarily stops the pwm signals from the raspberry pi
        ```bash
        >>> pwm.ChangeDutyCycle(0)               
        ```
        Stops the pwm signals from the raspberrypi GPIO pin
        ```bash
        >>> pwm.stop()                           
        ```
        Cleans up the GPIO pins
        ```bash
        >>> GPIO.cleanup()                      
        ```
        Quit the python shell
        ```bash
        >>> quit()
        ```                               
    - Once these steps are done, you can mount the camera and then follow with the final steps of executing the Drone.py file)

## About the code
YuNet face detection model is used to detect faces in the frame. The neural network inference is carried out by oak d lite camera mounted upon a servo motor which in turn is controlled by the raspberry pi. The aim of this code is to detect a face, track and keep it in the center of the camera frame within a certain threshold range of pixels values. Simultaenously, computer vision based feedback instructions for motor actuation, bounding boxes, relevant coordinates for reference and tracking are displayed in the frame itself while the code is running to make it more comprehensive and intuitive. As only one motor is controlled, the camera tracks faces about z axis(yaw movement). Also, the code is designed to track only one object at a time and the workspace of the camera is a semi-circle due to the servo motor rotation limitations.

Note: If more than one face are in the frame then the model would get confused and as a result it would follow only one face out of all present in the frame.

## Tips to run the code
In order to deploy the code, certain steps have to be followed. A virtual environment for python has to be setup, in which the dependencies given in the facetrackreq.txt file has to be installed. Once this is done the drone.py file can be executed directly to view the results.

While deploying the code, make sure to give the correct path to the blob file here in the Drone.py according to the file structured followed.
line 50: 
```python
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='/home/pi/Desktop/Facetracker/face_detection_yunet_120x160.blob', type=str)
```

## Prerequisites
Complete setup of raspbian on raspberrypi 3B/3B+/4B. It should be up and running connected via monitor,keyboard,mouse,wifi/ethernet.
Note: The live face detection feed and face tracking won't be displayed if the raspberrypi is connected via ssh.

Additionally run the following commands on the raspberrypi terminal:
```bash
    sudo apt-get update
```
```bash
    sudo apt-get upgrade
```

## Steps
Run the following commands on the rapberrypi terminal:
- Create a folder on Desktop named FaceTracker where the codes and virtual environment exists, then clone the following files in the folder
    - Clone the folder utilcode (other modules imported in the master python file)
    - Clone the file Drone.py (master python file to execute the face tracking code)
    - Clone the text file facetrackreq.txt (requirements.txt file - dependencies)
    - Clone the face detection model: face_detection_yunet_120x160.blob (face detection model)
- Create a virtual environment named "Drone" in the above folder
    ```bash
        python3 -m Drone <path_to_virtualenv>
    ```
    Activate the virtual environment
    ```bash
        source ./Drone/bin/activate 
    ```
    Deactivate the virtual environment
    ```bash
        deactivate 
    ```
- Install the dependencies for the code to work.
    ```bash
        python -m pip3 install -r facetrackreq.txt
    ```
   - Additionally run the following commands in the activated python environment
        ```bash
            sudo apt-get update
        ```
        ```bash
            sudo apt-get upgrade
        ```
- Make changes to blob file path accordingly to file directory structure
- Execute the code with all hardware connected as well as the correct file path for the model has been mentioned in the code
    ```bash
        python3 Drone.py
    ```
- Terminate the code (Ctrl + C : Keyboard Interrupt)
    ```bash
        ^C
    ```

## Challenges
The biggest challenge encountered in the making of this project was finding the correct face detection model,configuring the oak d camera and deploying the precompiled blob file on the hardware. The deployment code and the blob file for the above hardware was already ready which eased life to an extent. Finding this was the challenge.

## Improvements
- Fine tuning the servo motor rotation parameters to get a smooth rotation of the camera.
- Setting up the parameters such that the motor rotation copes up with the rate of pixel change of the detections in the frame.
- Extending the setup to three servo motor actuations in order to get yaw,pitch and roll movements of the camera (3 degrees of freedom).

## Face tracking setup and results
![GIF facetrack](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/FaceGIF.gif)
![CamMotion](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/CamMotion.gif)
![Setup](https://github.com/IRS-Devl/ComputerVision-UAV-Maneuvering/blob/main/UniAxisFaceTracker/FaceTrackingSetup.jpeg)

## Contributors
* [Anmol Singh](https://github.com/28anmol)
* [Luxonis Depthai Experiments](https://github.com/luxonis/depthai-experiments/tree/master/gen2-face-detection)
(The face detection model and its deployment code on oak d lite camera is taken from the above given link.)
* https://github.com/OlanrewajuDada (Credits for participating as a volunteer test object for face tracking GIF)
