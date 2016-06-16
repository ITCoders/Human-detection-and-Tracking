import cv2
import numpy
from PIL import Image
import sys
import imutils
import Surv_final
import os	
path = "/home/arpit/Projects/Survelliance_System/data/extra2/"
camera = cv2.VideoCapture(sys.argv[1])
i=12000
while True:
	grabbed,frame = camera.read()
	if not grabbed:
		break
	rects = Surv_final.detect_face(frame)
	if len(rects) > 0:
		for x,y,w,h in rects:
			cropped = frame[y:y+h,x:x+w]
			cv2.imwrite(os.path.join(path,str(i)+".jpg"),cropped)
		# for i in range(1,5):
		# 	frame = camera.read()
	i = i+1