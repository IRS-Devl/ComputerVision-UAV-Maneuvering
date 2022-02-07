'''
Face tracking code to keep the face withing a certain range of pixels by sending feedbacks to the drone.
Developed by: Anmol Singh in collaboration with the code and models provided by OpenCV courses.
'''

import numpy as np
import cv2


modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)      #This particular deep learning neural network model has been trained on tensorflow framework.


def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    #print(frameWidth, frameHeight)
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            if x1<0:
                x1=0        #New changes to the code.
            elif x2>640:
                x2=640
            elif y1<0:
                y1=0
            elif y2>480:
                y2=480

            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes, x1, y1, x2, y2


conf_threshold = 0.6


cap = cv2.VideoCapture(0)
cap.set(3,300)
cap.set(4,300)
#cap.set(10,100)
width = int(cap.get(3))
height = int(cap.get(4))
#print(width,height)


def leftrightcenteredfacetracker(midpointx,boxcenterx):
    threshold = 50
    if (abs(midpointx - boxcenterx) > threshold):
        if (midpointx > boxcenterx):
            cv2.putText(output,'Move Object Right',(250,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.arrowedLine(output,(300,60),(340,60),(0,0,255),3)
        else:
            cv2.putText(output,'Move Object Left',(250,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.arrowedLine(output,(340,60),(300,60),(0,0,255),3)
    elif (abs(midpointx - boxcenterx) < threshold):
        cv2.putText(output,'Object Centered',(250,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)


def updowncenteredfacetracker(midpointy,boxcentery):
    threshold = 50
    if (abs(midpointy - boxcentery) > threshold):
        if (midpointy > boxcentery):
            cv2.putText(output,'Move Object Down',(10,240),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.arrowedLine(output,(165,210),(165,250),(0,0,255),3)
        else:
            cv2.putText(output,'Move Object Up',(10,240),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            cv2.arrowedLine(output,(145,250),(145,210),(0,0,255),3)
    elif (abs(midpointy - boxcentery) < threshold):
        cv2.putText(output,'Object Centered',(10,240),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)


while True:
    ret,imageframe = cap.read()
    output,bboxes,x1,y1,x2,y2 = detectFaceOpenCVDnn(net, imageframe)
    #print(type(x1))
    #dict1 = {'x1':x1,'x2':x2,'y1':y1,'y2':y2}
    #print(dict1)
    mpx,mpy,mx1,my1,mx2,my2,cx,cy = int(width/2),int(height/2),int(width/2),0,int(width/2),height,int((x1+x2)/2),int((y1+y2)/2)
    cv2.line(output, (mx1,my1),(mx2,my2),(0,255,0),2)               #Draw a center line on the face detection frame.
    cv2.line(output, (mx1+50,my1),(mx2+50,my2),(0,255,0),1)         #Threshold for the object being in the centre.
    cv2.line(output, (mx1-50,my1),(mx2-50,my2),(0,255,0),1)
    cv2.line(output, (0,mpy-50),(width,mpy-50),(0,255,0),1)         #Threshold for the object being in the centre.
    cv2.line(output, (0,mpy+50),(width,mpy+50),(0,255,0),1)
    cv2.circle(output,(mpx,mpy), 2, (255,0,255),5)                    #Draw the midpoint of the center line frame
    cv2.circle(output,(cx,cy), 2, (255,0,255),5)                      #draw midpoint of face detection bounding box
    leftrightcenteredfacetracker(mpx,cx)
    updowncenteredfacetracker(mpy,cy)
    cv2.imshow('FaceDetectionFrame',output[:,:,::1])                #Write a 1 instead of -1 to put it on color mode from grayscale mode.
    cv2.imshow('OutputFrame',imageframe)
    #print(imageframe.shape[0])
    #print(height)
    if cv2.waitKey(1) == ord('q'):
        break


#centeredFaceTracker(cap)
cap.release()
cv2.destroyAllWindows()


















