#!/usr/bin/python3

import argparse
import glob
import os
import time
from recognition import recognize
import cv2
import imutils
from imutils.object_detection import non_max_suppression

font = cv2.FONT_HERSHEY_SIMPLEX
cascade_path = "face_cascades/haarcascade_profileface.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
count = 0
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
pickleFilePath="./recognition/encodings.pickle"
cascadeFilePath="./recognition/haarcascade_frontalface_default.xml"
def detect_people(frame):
    """
    detect humans using HOG descriptor
    Args:
        frame:
    Returns:
        processed frame
    """
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.06)
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (x, y, w, h) in rects:
        end_cord_x = x + w
        end_cord_y = y + h
        color = (0, 0, 255) #BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    return frame

if __name__ == '__main__':
    """
    main function
    """

camera = cv2.VideoCapture(0)
frame_object = recognize.recognize(pickleFilePath, cascadeFilePath)
while True:
    
    
    grabbed, frame = camera.read()
    frame_processed = detect_people(frame)
    frame_new = frame_object.recognize_face(frame_processed)
    cv2.imshow("Detected Human and face", frame_new)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

