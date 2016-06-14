#import necessary packages
from __future__ import print_function
import cv2
import os
from imutils.object_detection import non_max_suppression
import time	
import sys
import numpy as np
from PIL import Image	
import imutils
import time	
from imutils.object_detection import non_max_suppression

subject_label = 1
total_count = 0
subject_one_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
list_of_videos = []
#haar-cascade to detect faces
cascade_path = "haarcascade_profileface.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
#HOG descriptor object
hog = cv2.HOGDescriptor()
# setting HOG decriptor with pretrained support vector machine to dectect humans
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# Local Binary Pattern Histogram face recognizer
recognizer = cv2.face.createLBPHFaceRecognizer()

###################################################
#function to detect human using HOG descriptor
def detect_people(frame):
	(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(16, 16), scale=1.06)
	rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
	return frame

###################################################
#function to detect human faces using haar-cascades
def detect_face(frame):
	faces = face_cascade.detectMultiScale(frame)
	return faces

###################################################
#function to recognize human faces using LBPH face recognizer and pretrained model file
def recognize_face(frame_orginal,faces):
	predict_label = []
	predict_conf = []
	for x,y,w,h in faces:
		frame_orginal_grayscale = cv2.cvtColor(frame_orginal[y: y + h, x: x + w],cv2.COLOR_BGR2GRAY)
		cv2.imshow("cropped",frame_orginal_grayscale)
		predict_tuple = recognizer.predict(frame_orginal_grayscale)
		a,b = predict_tuple
		predict_label.append(a)
		predict_conf.append(b)
		print(predict_tuple)
	return predict_label
		# if predict_conf < 100.0:
			# print("person is recognized as {} with precision {}".format(predict_label,predict_conf))

###################################################
#function to draw rectangle around the human faces detected by detect_face function
def draw_faces(frame,faces):
	for (x, y, w, h) in faces:
		xA = x
		yA = y
		xB = x+w
		yB = y+h
		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
	return frame

###################################################
#function to put labels on faces predicted by LBPH face recognizer
def put_label_on_face(frame,faces,labels):
	i=0
	for x,y,w,h in faces:
		cv2.putText(frame,str(labels[i]),(x,y), font, 1,(255,255,255),2)
		i = i+1
	return frame

####################################################
#main function
if __name__=='__main__':	
	path = sys.argv[1]
	for f in os.listdir(path):
		if os.path.isfile(os.path.join(path,f)):
			list_of_videos.append(f)
	print(list_of_videos)
	if os.path.exists("model.yaml"):
		recognizer.load("model.yaml")
		for video in list_of_videos:
			camera = cv2.VideoCapture(os.path.join(path,video))
			while True:
				starttime = time.time()
				grabbed,frame = camera.read()
				if not grabbed:
					break
				frame_orginal = imutils.resize(frame, width=min(500, frame.shape[1]))
				frame_orginal1 = cv2.cvtColor(frame_orginal,cv2.COLOR_BGR2GRAY)
				frame_processed = detect_people(frame_orginal1)
				faces = detect_face(frame_orginal)
				if len(faces) > 0:
					frame_processed = draw_faces(frame_processed,faces)
					label = recognize_face(frame_orginal,faces)
					frame_processed = put_label_on_face(frame_processed,faces,label)
					for i in label:
						total_count = total_count+1
						if i == 1:
							subject_one_count = subject_one_count+1
				cv2.imshow("window",frame_processed)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
			camera.release()
			cv2.destroyAllWindows()
			endtime = time.time()
			print("accuracy is ",subject_one_count*100/total_count)
		# print("total_count is {} and subject one count is {}".format(total_count,subject_one_count))
	else:
		print("model file not found")