import numpy as np
import cv2
import imutils
import sys
from scipy.misc import imread
from scipy import signal


def auto_canny(image, sigma=0.25):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

image2 = cv2.imread(sys.argv[1],)
image2 = imutils.resize(image2, height=500)
gaussian = np.ones((5, 5), np.float32) / 25
laplacian = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
dst = cv2.filter2D(image2, -1, gaussian)
cv2.imshow('dsd', dst)
cv2.waitKey(0)
dst = cv2.Canny(dst, 99, 100)
_, th4, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image2, th4, -1, 255, 2)
cv2.imshow('window', image2)
cv2.waitKey(0)
auto = auto_canny(image2)
_, th5, _ = cv2.findContours(auto, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image2, th5, -1, 255, 2)
cv2.imshow('windas', image2)
cv2.waitKey(0)
