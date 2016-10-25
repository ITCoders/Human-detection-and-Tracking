import numpy as np
import cv2
import imutils
import sys
from collections import deque
from scipy.misc import imread
import argparse
from scipy import signal
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])
while True:
    # grab the current frame
    (grabbed, image2) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
    image2 = imutils.resize(image2, height=500)
    image1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image1, height=500)
    gaussian = np.ones((5, 5), np.float32) / 25
    laplacian = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    dst = cv2.filter2D(image, -1, gaussian)
    dst1 = cv2.filter2D(dst, -1, laplacian)
    # invert image
    dst1 = (255 - dst1)
    th2 = cv2.filter2D(dst1, -1, gaussian)
    th3 = cv2.adaptiveThreshold(
        th2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    res1 = cv2.bitwise_and(image2, image2, mask=th3)
    cv2.imshow('wine', res1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
