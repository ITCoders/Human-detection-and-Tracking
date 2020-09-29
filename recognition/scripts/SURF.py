import cv2
import imutils
import sys
img = cv2.imread(sys.argv[1])
img = imutils.resize(img, width=500, height=500)
surf = cv2.xfeatures2d.SURF_create(500)
kp, des = surf.detectAndCompute(img, None)
f = open('a.txt', 'w')
print(kp)
print(len(kp))
img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
cv2.imshow('window', img2)
cv2.waitKey(0)
