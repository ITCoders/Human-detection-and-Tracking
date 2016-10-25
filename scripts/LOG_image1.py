import numpy as np
import cv2


import imutils
import sys
from scipy.misc import imread
from scipy import signal
image2 = cv2.imread(sys.argv[1],)
image2 = imutils.resize(image2, height=500)
cv2.imshow('image', image2)
cv2.waitKey(0)
image1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image = imutils.resize(image1, height=500)
cv2.imshow('gdh', image)
cv2.waitKey(0)
gaussian = np.ones((5, 5), np.float32) / 25
laplacian = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
dst = cv2.filter2D(image, -1, gaussian)
cv2.imshow('dsd', dst)
cv2.waitKey(0)
dst1 = cv2.filter2D(dst, -1, laplacian)
cv2.imshow('jh', dst1)
cv2.waitKey(0)
# invert image
dst1 = (255 - dst1)
cv2.imshow('dhgf', dst1)
cv2.waitKey(0)
th2 = cv2.filter2D(dst1, -1, gaussian)
th3 = cv2.adaptiveThreshold(
    th2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
cv2.imshow('wind', th3)
cv2.waitKey(0)
res1 = cv2.bitwise_and(image2, image2, mask=th3)
cv2.imwrite('processed_img.jpeg', res1)
cv2.imshow('wine', res1)
cv2.waitKey(0)
_, th4, _ = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image2, th4, -1, 255, 2)
cv2.imshow('window', image2)
cv2.waitKey(0)
output = cv2.connectedComponentsWithStats(th3, 4, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]
print(num_labels)
print(labels)
print(stats)
print(centroids)
# (_,cnts, _) = cv2.findContours(th3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(image2, cnts, -1, 255, 2)
# cv2.imshow('win4',image2)
# cv2.waitKey(0)
