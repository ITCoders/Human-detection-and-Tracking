# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
import sys
# funciton to draw head and shoulders of human


def draw_Head_shoulders(frame):
    # cascade_path = "HS.xml"
    cascade_path = "haarcascade_profileface.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
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
    key = cv2.waitKey(1)
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# for imagePath in paths.list_images(args["images"]):
# load the image and resize it to (1) reduce detection time
# and (2) improve detection accuracy
fgbg = cv2.createBackgroundSubtractorMOG2()
camera = cv2.VideoCapture(sys.argv[1])
while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break
    image = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = image.copy()
    start = time.time()
    fgbg.apply(image)
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(16, 16), scale=1.06)
    if type(rects) is not tuple:
        cropped = image[rects[0][1]:rects[0][1] + rects[0]
                        [3], rects[0][0]:rects[0][0] + rects[0][2]]
        draw_Head_shoulders(cropped)
        # print(cropped)
    # draw the original bounding boxes
    # print(rects)
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    end = time.time()
    totalTimeTaken = end - start
    # print(totalTimeTaken)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    # print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))

    # show the output images
    # cv2.imshow("Before NMS", orig)
    cv2.imshow("window", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
