import cv2
import numpy as np
import imutils
# from matplotlib import pyplot as plt
import sys
# loading image
#img0 = cv2.imread('SanFrancisco.jpg',)
img2 = cv2.imread(sys.argv[1],)
img0 = imutils.resize(img2, height=500)
# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.bilateralFilter(gray, 9, 75, 75)
# img = cv2.GaussianBlur(gray,(3,3),0)
# convolute with proper kernels
laplacian = cv2.Laplacian(img, cv2.CV_8U)
sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)  # x
sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=5)  # y
edged = cv2.Canny(img, 30, 200)
# cv2.imshow('new',edged)
# cv2.waitKey(0)
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')

# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
(_, cnts, _) = cv2.findContours(
    edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(cnts)
# plt.show()
# On this output, draw all of the contours that we have detected
# in white, and set the thickness to be 3 pixels
cv2.drawContours(img0, cnts, -1, 255, 2)

# Spawn new windows that shows us the donut
# (in grayscale) and the detected contour
cv2.imshow('Output Contour', img0)

# Wait indefinitely until you push a key.  Once you do, close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
