# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import sys
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
camera = cv2.VideoCapture(sys.argv[1])
while True:
	grabbed,frame = camera.read()
	if not grabbed:
		break
	image = imutils.resize(frame, width=min(400, frame.shape[1]))
	orig = image.copy()
	
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
	padding=(8, 8), scale=1.05)
	
	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
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
	cv2.imshow("Before NMS", orig)
	cv2.imshow("After NMS", image)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
camera.release()
cv2.destroyAllWindows()