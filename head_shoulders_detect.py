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
image = cv2.imread(image_path)
image1 = imutils.resize(image,height=300)
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier(cascade_path)
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
    cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Faces found", image1)
cv2.waitKey(0)
