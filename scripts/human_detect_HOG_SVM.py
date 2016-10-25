# import the necessary packages
from __future__ import print_function
import argparse
import datetime
import imutils
import cv2
from imutils.object_detection import non_max_suppression
cascade = cv2.CascadeClassifier('lbpcascade_profileface.xml')
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--win-stride", type=str,
                default="(8, 8)", help="window stride")
ap.add_argument("-p", "--padding", type=str,
                default="(16, 16)", help="object padding")
ap.add_argument("-s", "--scale", type=float,
                default=1.05, help="image pyramid scale")
ap.add_argument("-m", "--mean-shift", type=int,
                default=-1, help="whether or not mean")
args = vars(ap.parse_args())
# evaluate the command line arguments (using the eval function like
# this is not good form, but let's tolerate it for the example)
winStride = eval(args["win_stride"])
padding = eval(args["padding"])
meanShift = True if args["mean_shift"] > 0 else False

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# load the image and resize it
image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(400, image.shape[1]))
# detect people in the image
start = datetime.datetime.now()
(rects, weights) = hog.detectMultiScale(image, hitThreshold=0, winStride=winStride,
                                        padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)
print(len(rects))
face = cascade.detectMultiScale(image)
print(len(face))
print("[INFO] detection took: {}s".format(
    (datetime.datetime.now() - start).total_seconds()))
pick = non_max_suppression(rects, probs=None, overlapThresh=10)
# draw the original bounding boxes
for (x, y, w, h) in pick:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Detections", image)
cv2.waitKey(0)
