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
# (_,cnts, _) = cv2.findContours(dst1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(imag,e cnts, -1, 255, 2)
# cv2.imshow('hello',image)
# cv2.waitKey(0)
# print(dst1)
dst1 = (255 - dst1)
cv2.imshow('dhgf', dst1)
cv2.waitKey(0)
res = cv2.bitwise_and(image2, image2, mask=dst1)
cv2.imshow('win', res)
cv2.waitKey(0)
print(dst1.shape)
th2 = cv2.filter2D(dst1, -1, gaussian)
th3 = cv2.adaptiveThreshold(
    th2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('wind', th3)
cv2.waitKey(0)
# kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
# closed = cv2.morphologyEx(th3,cv2.MORPH_CLOSE,kernal)
# cv2.imshow('win3',closed)
# cv2.waitKey(0)
res1 = cv2.bitwise_and(image2, image2, mask=th3)
cv2.imshow('final', res1)
cv2.waitKey(0)
# a = 0
# b = 0
# count = 0
# for i in th3:
# 	for j in i:
# 		if j == 0:
# 			count = count+1
# 			image2[i][j] = [0,255,0]
# 	a = a+1
# print(count)
# cv2.imshow('final',image2)
# cv2.waitKey(0)
