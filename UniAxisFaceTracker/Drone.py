#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time
from utilcode.utils import draw
from utilcode.priorbox import PriorBox
import RPi.GPIO as GPIO


'''
YuNet face detection demo running on device with video input from host.
https://github.com/ShiqiYu/libfacedetection
Run as:
python3 -m pip install -r requirements.txt
python3 main.py
Blob is taken from:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/144_YuNet
'''


# Take care to keep the duty cycle within a range of 0 and 12 as well the agle rotation of the servo
# within 0 and 180 degrees. The rotation should be very smooth. The code should also show face detections with all the points plotted.



# Raspberry pi pins setup for servo as well start of the pulsewidth modulation signal.
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
servopin = 11
GPIO.setup(servopin,GPIO.OUT)
myservo = GPIO.PWM(servopin,50)
myservo.start(7)                       # Starting reference position of the servo is 90 degrees. Just for record keeping to keep a track of the shaft angle of the motor.
time.sleep(0.5)
reference_ang = 90
min_ang = 0
max_ang = 180
min_dc = 2             # minimum duty cycle(0 deg)
max_dc = 12            # maximum duty cycle(180 deg)
temp_stop_dc = 0

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='/home/pi/Desktop/Facetracker/face_detection_yunet_120x160.blob', type=str)
parser.add_argument("-conf", "--confidence_thresh", help="set the confidence threshold", default=0.6, type=float)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.3, type=float)
parser.add_argument("-topk", "--keep_top_k", default=750, type=int, help='set keep_top_k for results outputing.')


args = parser.parse_args()

nn_path = args.nn_model

# resize input to smaller size for faster inference
NN_WIDTH, NN_HEIGHT = 160, 120
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480


# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)


# Define a neural network that will detect faces
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Define camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(VIDEO_WIDTH, VIDEO_HEIGHT)
cam.setInterleaved(False)
cam.setFps(40)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Define manip
manip = pipeline.createImageManip()
manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
manip.setWaitForConfigInput(False)

# Create outputs
xout_cam = pipeline.createXLinkOut()
xout_cam.setStreamName("cam")

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")


cam.preview.link(manip.inputImage)
cam.preview.link(xout_cam.input)
manip.out.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)


# Function defining facetracking feedback mechanism to the drone motors
# This code is designed to make a face tracker for a single axis

current_ang = reference_ang  # To keep a record of the current position of the motor with respect to the starting angle

def horizontalfacetracker(midpointx,boxcenterx):
    global current_ang
    threshold = 50    # A range of vertical lines to keep the object in between
    if (abs(midpointx - boxcenterx) > threshold):
        if (midpointx > boxcenterx):
            cv2.putText(frame,'Rotate Camera Clockwise',(250,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.arrowedLine(frame,(300,60),(340,60),(0,0,255),3)
            for angle in range(current_ang+1,current_ang+4,1):      # Rotating the motor 1-3 degrees ahead of its current angle value
                DC = (1.0/18.0)*(angle) + 2      # calculating duty cycle to rotate the motor from 1 to 3 degrees.
                if(DC >= 3.67 and DC <= 12):      # min_dc and max_dc values to stop the servo motor from exceeding its limits.
                        myservo.ChangeDutyCycle(DC)   # rotation of the servo to adjust the camera position for tracking face
                        time.sleep(0.01)
                        current_ang = angle               # Updating the current angle value after every for loop iteration
                else:
                        myservo.ChangeDutyCycle(temp_stop_dc)
        elif (midpointx < boxcenterx):
            cv2.putText(frame,'Rotate Camera Counterclockwise',(250,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.arrowedLine(frame,(340,60),(300,60),(0,0,255),3)
            for angle in range(current_ang-1,current_ang-4,-1):
                DC = (1.0/18.0)*(angle) + 2
                if(DC >= 3.67 and DC <= 12):        # min_dc and max_dc to stopthe motor from exceeding its limits.
                        myservo.ChangeDutyCycle(DC)
                        time.sleep(0.01)
                        current_ang = angle
                else:
                        myservo.ChangeDutyCycle(temp_stop_dc)
    else:
        myservo.ChangeDutyCycle(temp_stop_dc)
        cv2.putText(frame,'Test Object Centered',(250,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)



'''
def verticalfacetracker(midpointy,boxcentery):
    threshold = 50
    if (abs(midpointy - boxcentery) > threshold):
        if (midpointy > boxcentery):
            cv2.putText(frame,'Move Object Down',(10,240),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.arrowedLine(frame,(165,210),(165,250),(0,0,255),3)
        else:
            cv2.putText(frame,'Move Object Up',(10,240),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.arrowedLine(frame,(145,250),(145,210),(0,0,255),3)
    elif (abs(midpointy - boxcentery) < threshold):
        cv2.putText(frame,'Object Centered',(10,240),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

'''

# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue("cam", 4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    while True:
        in_frame = q_cam.get()
        in_nn = q_nn.get()

        frame = in_frame.getCvFrame()


        # get all layers
        conf = np.array(in_nn.getLayerFp16("conf")).reshape((1076, 2))
        iou = np.array(in_nn.getLayerFp16("iou")).reshape((1076, 1))
        loc = np.array(in_nn.getLayerFp16("loc")).reshape((1076, 14))


        # decode
        pb = PriorBox(input_shape=(NN_WIDTH, NN_HEIGHT), output_shape=(frame.shape[1], frame.shape[0]))
        dets = pb.decode(loc, conf, iou, args.confidence_thresh)
        bbox = dets[:, 0:4]             #Getting the boundary box coordinates from the neural network.
        width, height = 640,480
        mpx,mpy,mx1,my1,mx2,my2 = int(width/2),int(height/2),int(width/2),0,int(width/2),height
        cv2.line(frame, (mx1,my1),(mx2,my2),(0,255,0),2)               #Draw a center line on the face detection frame.
        cv2.line(frame, (mx1+50,my1),(mx2+50,my2),(0,255,0),1)         #Threshold for the object being in the centre.
        cv2.line(frame, (mx1-50,my1),(mx2-50,my2),(0,255,0),1)
        cv2.line(frame, (0,mpy-50),(width,mpy-50),(0,255,0),1)         #Threshold for the object being in the centre.
        cv2.line(frame, (0,mpy+50),(width,mpy+50),(0,255,0),1)
        cv2.line(frame, (0,mpy),(width,mpy),(0,255,0),2)               #Draw a center line on the face detection frame.
        cv2.circle(frame,(mpx,mpy), 2, (255,0,255),5)                    #Draw the midpoint of the center line frame


        # NMS
        if dets.shape[0] > 0:
            # NMS from OpenCV
            bboxes = dets[:, 0:4]
            scores = dets[:, -1]


            keep_idx = cv2.dnn.NMSBoxes(
                bboxes=bboxes.tolist(),
                scores=scores.tolist(),
                score_threshold=args.confidence_thresh,
                nms_threshold=args.iou_thresh,
                eta=1,
                top_k=args.keep_top_k)  # returns [box_num, class_num]
            keep_idx = np.squeeze(keep_idx)  # [box_num, class_num] -> [box_num]
            dets = dets[keep_idx]

        # Draw
        if dets.shape[0] > 0:

            if dets.ndim == 1:
                dets = np.expand_dims(dets, 0)

            lan = np.reshape(dets[:, 4:14], (-1,5,2))

            img_res = draw(img=frame,bboxes=dets[:, :4],landmarks=np.reshape(dets[:, 4:14], (-1, 5, 2)),scores=dets[:, -1])


            lan = np.reshape(dets[:, 4:14], (-1,5,2))
            cx, cy = lan[0,2,:]        # To get the coordinates of the nose point from the neural network.
            cv2.circle(frame,(int(cx),int(cy)),2,(255,28,255),5)
            horizontalfacetracker(mpx,cx)
            #verticalfacetracker(mpy,cy)

        # show fps
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)
        cv2.imshow("Detections", frame)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            GPIO.cleanup()
            break



