import numpy as np
import cv2
import imutils
import sys
from collections import deque
from scipy.misc import imread
import argparse
from scipy import signal
bgs = cv2.createBackgroundSubtractorMOG2()
camera = cv2.VideoCapture(sys.argv[1])
grabbed, frame1 = camera.read()
while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = bgs.apply(frame)
    cv2.imshow('window', f)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
camera.release()
cv2.destryallwindows()
