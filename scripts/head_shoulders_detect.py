from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
import sys
cascade_path = "HS.xml"
image_path = sys.argv[1]
# image = cv2.imread(image_path)
cascade = cv2.CascadeClassifier(cascade_path)
camera = cv2.VideoCapture(sys.argv[1])
# start = time.time()
while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, height=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start = time.time()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10),
        flags=0
    )
    end = time.time()
    total_time_taken = end - start
    print(total_time_taken)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("face and Shoulders", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
